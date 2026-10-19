#!/usr/bin/env python3
"""
Collect successful RB-Y1 CloseDrawerLeft source demos across RoboCasa layouts, styles and seeds.

Runs auto_close_drawer.py once per attempt. Successful recordings are renamed to
demo_NN_layout_LL_style_SS_seed_XXXX and listed in manifest.csv; failed
recordings are deleted. Scenes without room for the robot to the right of the
drawer (exit code 3) are skipped. Every attempt and its outcome is listed in
attempts.csv, and its output is in logs/.

By default the demos are randomized: the initial drawer opening is sampled from
a range, and the scripted motion parameters given as two values (MIN MAX) are
sampled uniformly per attempt (one value keeps them fixed). The sampled values
are in attempts.csv. Base placement noise is off by default: even a few cm / deg
puts the end of the push out of the left arm's reach.
"""
import argparse
import csv
import itertools
import shutil
import sys
from pathlib import Path

import numpy as np

from collect_close_door_demos import all_style_ids, run_and_log, spread_order, write_manifest

DEFAULT_SCRIPT = Path(__file__).resolve().with_name("auto_close_drawer.py")
EXIT_SUCCESS, EXIT_SKIPPED = 0, 3
# Scripted motion parameters of auto_close_drawer.py that can be randomized per attempt
MOTION_PARAMS = ["push_edge_inset", "contact_z_offset", "tool_pitch", "push_lead"]


def sample_param(values, rng):
    """One value: fixed. Two values: uniform in [MIN, MAX]."""
    if len(values) == 1:
        return values[0]
    if len(values) != 2 or values[0] > values[1]:
        raise ValueError(f"Expected VALUE or MIN MAX, got {values}")
    return float(rng.uniform(*values))


def write_attempts(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attempt", "layout", "style", "seed", *MOTION_PARAMS, "outcome", "exit", "log"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Drawer-closing script to run")
    parser.add_argument("--output", default="rby1_drawer_demos_10", help="Output directory")
    parser.add_argument("--count", type=int, default=10, help="Number of successful demonstrations")
    parser.add_argument("--max-attempts", type=int, default=150, help="Attempts in total, skipped scenes included")
    parser.add_argument("--layouts", type=int, nargs="+", default=[1], help="RoboCasa layout ids to cycle through")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--controller", default=None, help="Optional controller JSON override")
    parser.add_argument("--base-pos-noise", type=float, default=0.0, help="Std of the base x / y placement noise [m]")
    parser.add_argument("--base-yaw-noise", type=float, default=0.0, help="Std of the base yaw placement noise [deg]")
    parser.add_argument("--robot-side-gap", type=float, default=0.3, help="Gap between the drawer's right edge and the robot [m]")
    parser.add_argument("--robot-front-offset", type=float, default=-0.18,
                        help="Robot offset along the drawer's front direction [m] (RoboCasa: -0.23, which leaves "
                             "the end of the push at the edge of the left arm's reach)")
    parser.add_argument("--robot-yaw", type=float, default=5.0,
                        help="Extra robot base yaw toward the drawer (left +) [deg]. At 0 the left shoulder hides a fully "
                             "open drawer's front from the head camera; 10 shows it but costs reach (37%% vs 60%% success)")
    parser.add_argument("--drawer-open", type=float, nargs=2, default=[0.5, 1.0], metavar=("MIN", "MAX"),
                        help="Initial drawer opening range (fraction of fully open)")
    parser.add_argument("--push-edge-inset", type=float, nargs="+", default=[0.06, 0.14], help="VALUE or MIN MAX [m]")
    parser.add_argument("--contact-z-offset", type=float, nargs="+", default=[-0.03, 0.01], help="VALUE or MIN MAX [m]")
    parser.add_argument("--tool-pitch", type=float, nargs="+", default=[0.0, 6.0], help="VALUE or MIN MAX [deg]")
    parser.add_argument("--push-lead", type=float, nargs="+", default=[0.02, 0.04], help="VALUE or MIN MAX [m]")
    parser.add_argument("--torso-squat", type=float, default=None)
    parser.add_argument("--steps-per-waypoint", type=int, default=4)
    parser.add_argument("--max-push-steps", type=int, default=1500)
    parser.add_argument("--render", action="store_true", help="Show the MuJoCo viewer for every attempt")
    args = parser.parse_args()

    script = Path(args.script).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"Could not find demo script: {script}")

    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    manifest_path = root / "manifest.csv"
    attempts_path = root / "attempts.csv"

    styles = all_style_ids()
    style_order = spread_order(styles, min(args.count, len(styles)))
    # All layouts with the evenly spread styles first, then the remaining styles
    scene_order = [(layout, style) for style, layout in itertools.product(style_order, args.layouts)]
    print(f"Demo script: {script}")
    print(f"Layouts: {args.layouts}")
    print(f"Detected {len(styles)} RoboCasa styles:")
    print(styles)
    print("First-pass style order:")
    print(style_order[:min(args.count, len(style_order))])

    successful = failed = skipped = 0
    attempt = 0
    rows, attempt_rows = [], []

    while successful < args.count and attempt < args.max_attempts:
        layout, style = scene_order[attempt % len(scene_order)]
        seed_round = attempt // len(scene_order)
        seed = args.seed_start + attempt + seed_round * 1000
        attempt += 1

        tag = f"attempt_{attempt:03d}_layout_{layout:02d}_style_{style:02d}_seed_{seed:04d}"
        attempt_dir = root / tag
        log_path = logs_dir / f"{tag}.log"
        shutil.rmtree(attempt_dir, ignore_errors=True)
        rng = np.random.default_rng(seed)
        motion = {name: sample_param(getattr(args, name), rng) for name in MOTION_PARAMS}

        cmd = [
            sys.executable, str(script),
            "--environment", "CloseDrawerLeft",
            "--layout", str(layout),
            "--style", str(style),
            "--seed", str(seed),
            "--base-pos-noise", str(args.base_pos_noise),
            "--base-yaw-noise", str(args.base_yaw_noise),
            f"--robot-side-gap={args.robot_side_gap}",
            f"--robot-front-offset={args.robot_front_offset}",
            f"--robot-yaw={args.robot_yaw}",
            "--drawer-open-min", str(args.drawer_open[0]),
            "--drawer-open-max", str(args.drawer_open[1]),
            *[a for name, value in motion.items() for a in ("--" + name.replace("_", "-"), f"{value:.4f}")],
            "--steps-per-waypoint", str(args.steps_per_waypoint),
            "--max-push-steps", str(args.max_push_steps),
            "--record-dir", str(attempt_dir),
        ]
        if args.torso_squat is not None:
            cmd += ["--torso-squat", str(args.torso_squat)]
        if args.controller is not None:
            cmd += ["--controller", str(Path(args.controller).expanduser())]
        if not args.render:
            cmd.append("--no-render")

        rc = run_and_log(cmd, log_path)

        if rc == EXIT_SUCCESS and attempt_dir.exists():
            successful += 1
            outcome = "success"
            final_dir = root / f"demo_{successful:02d}_layout_{layout:02d}_style_{style:02d}_seed_{seed:04d}"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            attempt_dir.rename(final_dir)
            rows.append({
                "demo": successful,
                "layout": layout,
                "style": style,
                "seed": seed,
                "arm": "left",
                "directory": str(final_dir),
                "log": str(log_path),
            })
            print(f"\n[SAVED] demo {successful}/{args.count}: layout={layout}, style={style}, seed={seed}")
            print(f"        {final_dir}")
        else:
            shutil.rmtree(attempt_dir, ignore_errors=True)
            if rc == EXIT_SUCCESS:
                failed += 1
                outcome = "missing_recording"
                print(f"\n[ERROR] Child process returned success but recording directory does not exist: {attempt_dir}")
            elif rc == EXIT_SKIPPED:
                skipped += 1
                outcome = "skipped"
                print(f"\n[SKIPPED] layout={layout}, style={style}, seed={seed}")
            else:
                failed += 1
                outcome = "failed"
                print(f"\n[FAILED] layout={layout}, style={style}, seed={seed}, exit={rc}")

        attempt_rows.append({
            "attempt": attempt, "layout": layout, "style": style, "seed": seed,
            **{name: round(value, 4) for name, value in motion.items()},
            "outcome": outcome, "exit": rc, "log": str(log_path),
        })
        write_manifest(manifest_path, rows)
        write_attempts(attempts_path, attempt_rows)

    print("\n===== Collection summary =====")
    print(f"Successful demos: {successful}/{args.count}")
    print(f"Attempts:         {attempt}/{args.max_attempts} ({failed} failed, {skipped} skipped)")
    print(f"Manifest:         {manifest_path}")
    print(f"Attempts log:     {attempts_path}")
    print(f"Logs:             {logs_dir}")

    if successful < args.count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
