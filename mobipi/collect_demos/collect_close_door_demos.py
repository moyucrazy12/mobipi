#!/usr/bin/env python3
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
import robocasa.models.scenes.scene_registry as SceneRegistry

DEFAULT_SCRIPT = Path(__file__).resolve().with_name("auto_close_microwave_door.py")

def all_style_ids():
    styles = [int(x) for x in SceneRegistry.unpack_style_ids(None)]
    if not styles:
        raise RuntimeError("RoboCasa reported no available style IDs.")
    return list(dict.fromkeys(styles))

def spread_order(values, first_count):
    """Put evenly spaced values first, then append the remaining values."""
    values = list(values)
    if first_count <= 0:
        return values
    if first_count == 1:
        return values
    if len(values) <= first_count:
        return values
    idxs = [round(i * (len(values) - 1) / (first_count - 1)) for i in range(first_count)]
    idxs = list(dict.fromkeys(idxs))
    first = [values[i] for i in idxs]
    return first + [x for x in values if x not in first]

def run_and_log(cmd, log_path):
    print("\n$", " ".join(map(str, cmd)), flush=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()

def write_manifest(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["demo", "layout", "style", "seed", "arm", "directory", "log"])
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser(description="Collect successful RBY1 CloseSingleDoor source demos across RoboCasa styles and seeds.")
    parser.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Surface-closing script to run")
    parser.add_argument("--output", default="rby1_seed_demos_10", help="Output directory")
    parser.add_argument("--count", type=int, default=10, help="Number of successful demonstrations")
    parser.add_argument("--max-attempts", type=int, default=80)
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--arm", choices=["auto", "left", "right"], default="left")
    parser.add_argument("--controller", default=None, help="Optional controller JSON override")
    parser.add_argument("--robot-x", type=float, default=0.8)
    parser.add_argument("--robot-y", type=float, default=-1.45)
    parser.add_argument("--robot-z", type=float, default=0.0)
    parser.add_argument("--robot-yaw", type=float, default=1.5707963267948966)
    parser.add_argument("--push-x-direction", type=int, choices=[-1, 0, 1], default=1)
    parser.add_argument("--push-lead", type=float, default=0.010)
    parser.add_argument("--steps-per-waypoint", type=int, default=4)
    parser.add_argument("--handle-clearance", type=float, default=0.053)
    parser.add_argument("--finish-angle", type=float, default=0.12)
    parser.add_argument("--contact-z-offset", type=float, default=0.0)
    args = parser.parse_args()

    script = Path(args.script).expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(f"Could not find demo script: {script}")

    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    logs_dir = root / "logs"
    logs_dir.mkdir(exist_ok=True)
    manifest_path = root / "manifest.csv"

    styles = all_style_ids()
    style_order = spread_order(styles, min(args.count, len(styles)))
    print(f"Demo script: {script}")
    print(f"Detected {len(styles)} RoboCasa styles:")
    print(styles)
    print("First-pass style order:")
    print(style_order[:min(args.count, len(style_order))])

    successful = 0
    attempt = 0
    rows = []

    while successful < args.count and attempt < args.max_attempts:
        style = style_order[attempt % len(style_order)]
        seed_round = attempt // len(style_order)
        seed = args.seed_start + attempt + seed_round * 1000
        attempt += 1

        tag = f"attempt_{attempt:03d}_layout_{args.layout:02d}_style_{style:02d}_seed_{seed:04d}"
        attempt_dir = root / tag
        log_path = logs_dir / f"{tag}.log"
        shutil.rmtree(attempt_dir, ignore_errors=True)

        cmd = [
            sys.executable, str(script),
            "--environment", "CloseSingleDoor",
            "--layout", str(args.layout),
            "--style", str(style),
            "--seed", str(seed),
            "--arm", args.arm,
            "--robot-x", str(args.robot_x),
            "--robot-y", str(args.robot_y),
            "--robot-z", str(args.robot_z),
            "--robot-yaw", str(args.robot_yaw),
            "--push-x-direction", str(args.push_x_direction),
            "--push-lead", str(args.push_lead),
            "--contact-z-offset", str(args.contact_z_offset),
            "--steps-per-waypoint", str(args.steps_per_waypoint),
            "--handle-clearance", str(args.handle_clearance),
            "--finish-angle", str(args.finish_angle),
            "--record-dir", str(attempt_dir),
        ]
        if args.controller is not None:
            cmd += ["--controller", str(Path(args.controller).expanduser())]

        rc = run_and_log(cmd, log_path)

        if rc == 0:
            successful += 1
            final_dir = root / f"demo_{successful:02d}_layout_{args.layout:02d}_style_{style:02d}_seed_{seed:04d}"
            if final_dir.exists():
                shutil.rmtree(final_dir)
            if not attempt_dir.exists():
                print(f"\n[ERROR] Child process returned success but recording directory does not exist: {attempt_dir}")
                successful -= 1
                continue
            attempt_dir.rename(final_dir)
            rows.append({
                "demo": successful,
                "layout": args.layout,
                "style": style,
                "seed": seed,
                "arm": args.arm,
                "directory": str(final_dir),
                "log": str(log_path),
            })
            print(f"\n[SAVED] demo {successful}/{args.count}: layout={args.layout}, style={style}, seed={seed}, arm={args.arm}")
            print(f"        {final_dir}")
        else:
            shutil.rmtree(attempt_dir, ignore_errors=True)
            print(f"\n[FAILED] layout={args.layout}, style={style}, seed={seed}, arm={args.arm}, exit={rc}")

        write_manifest(manifest_path, rows)

    print("\n===== Collection summary =====")
    print(f"Successful demos: {successful}/{args.count}")
    print(f"Attempts:         {attempt}/{args.max_attempts}")
    print(f"Manifest:         {manifest_path}")
    print(f"Logs:             {logs_dir}")

    if successful < args.count:
        raise SystemExit(1)

if __name__ == "__main__":
    main()