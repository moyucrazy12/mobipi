#!/usr/bin/env python3
"""
Collect successful RB-Y1 CloseDrawerLeft source demos across RoboCasa layouts, styles and seeds.

Runs auto_close_drawer.py once per attempt. Successful recordings are renamed to
demo_NN_layout_LL_style_SS_seed_XXXX and listed in manifest.csv; failed
recordings are deleted. Scenes without room for the robot to the right of the
drawer (exit code 3) are skipped. Every attempt and its outcome is listed in
attempts.csv, and its output is in logs/.
"""
import argparse
import csv
import itertools
import shutil
import sys
from pathlib import Path

from collect_close_door_demos import all_style_ids, run_and_log, spread_order, write_manifest

DEFAULT_SCRIPT = Path(__file__).resolve().with_name("auto_close_drawer.py")
EXIT_SUCCESS, EXIT_SKIPPED = 0, 3


def write_attempts(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["attempt", "layout", "style", "seed", "outcome", "exit", "log"])
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
    parser.add_argument("--push-edge-inset", type=float, default=0.10)
    parser.add_argument("--contact-z-offset", type=float, default=0.0)
    parser.add_argument("--torso-squat", type=float, default=None)
    parser.add_argument("--tool-pitch", type=float, default=0.0)
    parser.add_argument("--push-lead", type=float, default=0.03)
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

        cmd = [
            sys.executable, str(script),
            "--environment", "CloseDrawerLeft",
            "--layout", str(layout),
            "--style", str(style),
            "--seed", str(seed),
            "--push-edge-inset", str(args.push_edge_inset),
            "--contact-z-offset", str(args.contact_z_offset),
            "--tool-pitch", str(args.tool_pitch),
            "--push-lead", str(args.push_lead),
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
