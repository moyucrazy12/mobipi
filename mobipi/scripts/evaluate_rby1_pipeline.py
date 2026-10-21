r"""Run the complete RB-Y1 pipeline over seeds and save resumable results.

Other options are forwarded to run_rby1_pipeline; --scene-dir is required.
For example, from the repository root in the mobipi_pipeline environment:

    python -m mobipi.scripts.evaluate_rby1_pipeline \
        --scene-dir scene_data/close_single_door/layout1_style1 \
        --skip-detection --num-seeds 20 --output-dir logs/rby1_pipeline/seeds_0_19

Repeat the command with --resume to skip finished seeds and retry interrupted
ones. Each seed runs in a separate process using this Python environment.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mobipi.scripts import run_rby1_pipeline as pipeline


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = result.add_mutually_exclusive_group()
    selection.add_argument("--seeds", nargs="+", type=int, help="Explicit seeds, e.g. --seeds 0 1 42 123")
    selection.add_argument("--num-seeds", type=int, default=20, help="Number of consecutive seeds (default: 20)")
    result.add_argument("--start-seed", type=int, default=0, help="First consecutive seed (default: 0)")
    result.add_argument("--output-dir", type=Path, required=True, help="New sweep directory, or existing one with --resume")
    result.add_argument("--timeout-minutes", type=float, default=20, help="Wall-time limit per seed (default: 20)")
    result.add_argument("--resume", action="store_true", help="Skip finished seeds; retry interrupted or unfinished seeds")
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def fingerprint(path):
    path = path.expanduser().resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def stop_process(process):
    """Give the pipeline time to save progress, then terminate an unresponsive run."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=15)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        process.wait()


def run_seed(seed, output, pipeline_args, timeout_minutes):
    seed_dir = output / f"seed_{seed}"
    seed_dir.mkdir(exist_ok=True)
    attempt = 1
    while (seed_dir / f"attempt_{attempt:03d}.log").exists() or (seed_dir / f"attempt_{attempt:03d}").exists():
        attempt += 1
    run_dir = seed_dir / f"attempt_{attempt:03d}"
    log_path = run_dir.with_suffix(".log")
    command = [sys.executable, "-u", "-m", "mobipi.scripts.run_rby1_pipeline", *pipeline_args,
               "--seed", str(seed), "--output-dir", str(run_dir)]
    started = time.perf_counter()
    status, reason, returncode = "", "", None
    with log_path.open("w") as log:
        log.write("Command: " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, cwd=pipeline.ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout_minutes * 60)
        except subprocess.TimeoutExpired:
            status, reason = "timeout", f"Exceeded {timeout_minutes:g} minutes"
            stop_process(process)
        except KeyboardInterrupt:
            status, reason = "interrupted", "Batch interrupted by user"
            stop_process(process)
        returncode = process.returncode

    result = {}
    try:
        result = json.loads((run_dir / "result.json").read_text())
        if not isinstance(result, dict):
            raise ValueError("Expected a result object")
    except (OSError, ValueError) as error:
        result = {}
        reason = reason or f"No readable pipeline result: {error}"
    policy = result.get("policy") or {}
    arm = result.get("arm_return") or {}
    if not status:
        if result.get("interrupted"):
            status = "interrupted"
        elif returncode != 0 or result.get("error") or not result:
            status = "error"
        elif result.get("success"):
            status = "success"
        elif not result.get("navigation_success"):
            status = "navigation_failed"
        elif not policy.get("success"):
            status = "policy_failed"
        elif not arm.get("success"):
            status = "arm_return_failed"
        else:
            status = "failed"
    reason = reason or result.get("error") or result.get("stop_reason") or arm.get("stop_reason") or ""
    if status == "policy_failed" and not reason:
        reason = policy.get("failure_classification", "")
    if status == "error" and not reason:
        reason = f"Pipeline exited with code {returncode}; see {log_path}"
    row = dict(seed=seed, status=status, success=status == "success", stage=result.get("stage", ""),
               navigation_success=result.get("navigation_success"), policy_success=policy.get("success"),
               arm_return_success=arm.get("success"), policy_steps=policy.get("steps_executed"),
               success_timestep=policy.get("success_timestep"), wall_seconds=round(time.perf_counter() - started, 2),
               returncode=returncode, reason=reason, run_dir=str(run_dir), log_path=str(log_path))
    atomic_json(seed_dir / "evaluation.json", row)
    return row


def save_summary(output, seeds, rows):
    ordered = [rows[seed] for seed in seeds if seed in rows]
    finished = [row for row in ordered if row["status"] != "interrupted"]
    successful = [row["seed"] for row in finished if row["success"]]
    by_status = {}
    for row in ordered:
        by_status.setdefault(row["status"], []).append(row["seed"])
    summary = dict(requested_seeds=seeds, completed=len(finished), successful_seeds=successful,
                   success_rate=len(successful) / len(finished) if finished else None,
                   seeds_by_status=by_status, pending_seeds=[seed for seed in seeds if seed not in rows],
                   success_definition="Navigation, stable door closure, and arm return all succeed",
                   success_rate_definition="Successful seeds / finished seeds, including errors and timeouts")
    atomic_json(output / "summary.json", summary)
    if ordered:
        temporary = output / "results.csv.tmp"
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ordered[0]))
            writer.writeheader()
            writer.writerows(ordered)
        temporary.replace(output / "results.csv")
    return summary


def main(argv=None):
    arguments = parser()
    args, forwarded = arguments.parse_known_args(argv)
    if any(item.split("=")[0] == "--seed" for item in forwarded):
        arguments.error("Use --seeds or --start-seed/--num-seeds instead of --seed")
    if args.seeds is not None and args.start_seed != 0:
        arguments.error("Use --start-seed with --num-seeds, or use explicit --seeds")
    if args.num_seeds < 1 or not math.isfinite(args.timeout_minutes) or args.timeout_minutes <= 0:
        arguments.error("num-seeds and timeout-minutes must be positive")
    seeds = args.seeds if args.seeds is not None else list(range(args.start_seed, args.start_seed + args.num_seeds))
    if len(set(seeds)) != len(seeds) or any(seed < 0 or seed * 10000 >= 2**32 for seed in seeds):
        arguments.error("Use unique seeds between 0 and 429496 (the pose sampler seeds NumPy with seed * 10000)")
    run_parser = pipeline.parser()
    run_parser.allow_abbrev = False
    settings = vars(run_parser.parse_args(forwarded))
    settings.pop("seed")
    settings.pop("output_dir")
    pipeline_args = []
    for name, value in settings.items():
        if isinstance(value, Path):
            value = str(value.expanduser().resolve())
            settings[name] = value
        if value is False or value is None:
            continue
        pipeline_args.append("--" + name.replace("_", "-"))
        if value is not True:
            pipeline_args.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    scene = Path(settings["scene_dir"])
    scene_models = sorted(scene.glob(f"model/**/nerfstudio_models/step-{settings['scene_model_step']:09d}.ckpt"))
    if not scene_models:
        arguments.error(f"No trained scene checkpoint under {scene}")
    try:
        inputs = {name: fingerprint(Path(settings[name])) for name in ("checkpoint", "dataset")}
        inputs.update(scene_cloud=fingerprint(scene / "pc.ply"), scene_model=fingerprint(scene_models[-1]))
    except OSError as error:
        arguments.error(str(error))
    protocol = dict(pipeline_settings=settings, inputs=inputs, python=sys.executable,
                    pipeline_sha256=hashlib.sha256(Path(pipeline.__file__).read_bytes()).hexdigest(),
                    timeout_minutes=args.timeout_minutes)
    output = args.output_dir.expanduser().resolve()
    if args.resume:
        manifest = output / "metadata.json"
        if not manifest.is_file() or json.loads(manifest.read_text())["protocol"] != protocol:
            arguments.error("Resume settings or inputs changed; use the original settings or a new output directory")
    elif output.exists():
        arguments.error("Output directory already exists; add --resume or choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            arguments.error("Another evaluation is already using this output directory")
        atomic_json(output / "metadata.json", dict(protocol=protocol, requested_seeds=seeds))
        rows = {}
        for seed in seeds:
            saved = output / f"seed_{seed}" / "evaluation.json"
            if args.resume and saved.is_file():
                rows[seed] = json.loads(saved.read_text())
        save_summary(output, seeds, rows)
        for index, seed in enumerate(seeds, 1):
            if seed in rows and rows[seed]["status"] != "interrupted":
                print(f"[{index}/{len(seeds)}] seed={seed}: skip finished ({rows[seed]['status']})", flush=True)
                continue
            print(f"[{index}/{len(seeds)}] seed={seed}: starting; logs in {output / f'seed_{seed}'}", flush=True)
            rows[seed] = run_seed(seed, output, pipeline_args, args.timeout_minutes)
            summary = save_summary(output, seeds, rows)
            row = rows[seed]
            print(f"seed={seed}: {row['status']} ({row['wall_seconds']:.1f}s); "
                  f"successes={len(summary['successful_seeds'])}/{summary['completed']}", flush=True)
            if row["status"] == "interrupted" or row["returncode"] == 2:
                if row["returncode"] == 2:
                    print("Pipeline rejected the configuration; stopping the batch. See the seed log.", flush=True)
                break
        summary = save_summary(output, seeds, rows)
        print(f"Successful seeds: {summary['successful_seeds']}\nResults: {output}", flush=True)


if __name__ == "__main__":
    # Handle normal `kill PID` and Ctrl+C even when launched with nohup in the background.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    main()
