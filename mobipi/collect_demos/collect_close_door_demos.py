"""Collect balanced, successful microwave demos with reproducible random scene seeds."""

import argparse
from collections import Counter
import csv
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import secrets
import shlex
import shutil
import signal
import subprocess
import sys


LAYOUT_Y_OFFSETS = {1: -0.35, 2: -0.28, 3: -0.35, 4: -0.28, 5: -0.28,
                    6: -0.35, 7: -0.28, 8: -0.35, 9: -0.35}
FIELDS = ("demo", "attempt", "layout", "style", "angle_deg", "seed", "spawn_offset_y",
          "success", "failure_reason", "phase", "detail", "exit_code", "actual_initial_angle_deg",
          "final_angle_deg", "base_pose", "directory", "raw_demo", "log", "command")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def write_progress(output, state, args, quotas):
    """The atomic state file is authoritative; CSVs and summaries are derived views."""
    atomic_json(output / "progress.json", state)
    saved = [row for row in state["attempts"] if row["success"]]
    atomic_json(output / "summary.json", saved)
    for name, rows in (("manifest.csv", saved), ("attempts.csv", state["attempts"])):
        path = output / name
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    counts = {key: {str(value): sum(row[key] == value for row in saved) for value in values}
              for key, values in (("layout", args.layouts), ("style", args.styles), ("angle_deg", args.angles))}
    failures = Counter((row["layout"], row["style"], row["angle_deg"]) for row in state["attempts"]
                       if not row["success"] and row["failure_reason"] != "interrupted")
    atomic_json(output / "stats.json", dict(saved=len(saved), target=args.count,
                attempts=len(state["attempts"]), max_attempts=args.max_attempts,
                saved_counts=counts, layout_targets=quotas,
                remaining_per_layout={str(layout): quotas[layout]-counts["layout"][str(layout)] for layout in args.layouts},
                exhausted_cases=[dict(layout=case[0], style=case[1], angle_deg=case[2], failures=number)
                                 for case, number in sorted(failures.items()) if number >= args.max_failures_per_case],
                failures=dict(Counter(row["failure_reason"] for row in state["attempts"] if not row["success"]))))


def choose_case(cases, rows, quotas, rng, single_pass, max_failures):
    """Reserve layout quotas, explore new cases, then spread saved styles/angles."""
    saved = [row for row in rows if row["success"]]
    successes = Counter(row["layout"] for row in saved)
    styles = Counter(row["style"] for row in saved)
    angles = Counter(row["angle_deg"] for row in saved)
    tried_layouts = Counter(row["layout"] for row in rows)
    tried = Counter((row["layout"], row["style"], row["angle_deg"]) for row in rows)
    failed = Counter((row["layout"], row["style"], row["angle_deg"]) for row in rows
                     if not row["success"] and row["failure_reason"] != "interrupted")
    available = [case for case in cases if successes[case[0]] < quotas[case[0]]
                 and failed[case] < max_failures and (not single_pass or not tried[case])]
    if not available:
        return None
    # Equal effort initially prevents a failing layout from monopolizing the run.
    layout = min({case[0] for case in available}, key=lambda value: (tried_layouts[value], value))
    available = [case for case in available if case[0] == layout]
    rng.shuffle(available)
    return min(available, key=lambda case: (tried[case], styles[case[1]] + angles[case[2]]))


def run_attempt(command, log_path, timeout, lock_fd):
    with log_path.open("w") as log:
        log.write("Command: " + shlex.join(command) + "\n")
        log.flush()
        child = subprocess.Popen(command, cwd=Path(__file__).resolve().parents[2], stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(lock_fd,))
        reason = None
        try:
            child.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
            reason = "interrupted" if isinstance(error, KeyboardInterrupt) else "timeout"
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return child.returncode, reason


def finish_attempt(output, pending, returncode, override, bases):
    case = Path(pending["directory"])
    if case.parent.resolve() != (output / "demos").resolve():
        raise ValueError("Attempt directory is outside this collection's demos directory")
    try:
        result = json.loads((case / "result.json").read_text())
    except (OSError, ValueError):
        result = dict(success=False, failure_reason="runtime_error", detail="No valid result.json; see console log")
    reason = override or result.get("failure_reason")
    if not reason and (returncode not in (0, None) or not result.get("success")):
        reason = "runtime_error"
    base = result.get("base_position", []) + result.get("base_orientation", [])
    key = pending["layout"], pending["style"], pending["seed"]
    if len(base) == 6:
        if key in bases and any(abs(a-b) > 1e-6 for a, b in zip(base, bases[key])):
            reason = "base_pose_changed"
        else:
            bases[key] = base
    raw = Path(result["raw_demo"]).resolve() if result.get("raw_demo") else None
    if not reason and (raw is None or not raw.is_relative_to(case) or not (raw / "model.xml").is_file()
                       or not (raw / "ep_meta.json").is_file() or not any(raw.glob("state_*.npz"))):
        reason = "missing_recording"
    success = bool(result.get("success") and not reason)
    row = dict(demo=pending["demo"] if success else "", attempt=pending["attempt"],
               layout=pending["layout"], style=pending["style"], angle_deg=pending["angle_deg"],
               seed=pending["seed"], spawn_offset_y=pending["spawn_offset_y"], success=success,
               failure_reason=reason or "", phase=result.get("phase", "setup"), detail=result.get("detail", ""),
               exit_code=returncode, actual_initial_angle_deg=result.get("actual_initial_angle_deg"),
               final_angle_deg=result.get("final_angle_deg"), base_pose=base,
               directory=str(case) if success else "", raw_demo=str(raw) if success else "",
               log=pending["log"], command=shlex.join(pending["command"]))
    if not success and case.exists():
        shutil.rmtree(case)
    return row


def source_hashes():
    folder = Path(__file__).resolve().parent
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in
            ("collect_close_door_demos.py", "auto_close_microwave_door.py", "auto_close_door_general.py", "door_motion.py")}


def request_stop(signum, frame):
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True, help="New collection directory, or existing with --resume")
    parser.add_argument("--count", type=int, help="Target successful demos; omit for one attempt per grid case")
    parser.add_argument("--layouts", nargs="+", type=int, default=[1])
    parser.add_argument("--layout", type=int, help="Alias for a single layout")
    parser.add_argument("--styles", nargs="+", type=int, default=[1])
    parser.add_argument("--angles", nargs="+", type=float, default=[20, 40, 60, 80])
    parser.add_argument("--seed", "--seed-start", type=int, default=None,
                        help="Fixed scene seed for grid validation only; default: a fresh seed per attempt")
    parser.add_argument("--sampler-seed", type=int, help="Reproduce batch scheduling while still varying scene seeds")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--base-local", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"))
    parser.add_argument("--spawn-offset-x", type=float, default=0.0)
    parser.add_argument("--spawn-offset-y", type=float, default=None, help="Override the automatic per-layout setback")
    parser.add_argument("--timeout-minutes", type=float, default=10)
    parser.add_argument("--max-attempts", type=int, help="Total attempt budget; default: 10 times --count, or one grid pass")
    parser.add_argument("--max-failures-per-case", type=int, default=3,
                        help="Stop retrying a layout/style/angle after this many failed scene seeds")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print settings without creating files or running simulations")
    args, extra = parser.parse_known_args(argv)
    args.layouts = list(dict.fromkeys([args.layout] if args.layout is not None else args.layouts))
    args.styles, args.angles = list(dict.fromkeys(args.styles)), list(dict.fromkeys(args.angles))
    if args.timeout_minutes <= 0 or any(not 0 < angle <= 90 for angle in args.angles):
        parser.error("Use positive timeout and angles between 0 and 90 degrees")
    if any(layout not in range(10) for layout in args.layouts):
        parser.error("Layout IDs must be between 0 and 9")
    if any(style not in range(12) for style in args.styles):
        parser.error("Style IDs must be between 0 and 11")
    if args.count is not None and args.count < len(args.layouts):
        parser.error("--count must allow at least one demo per selected layout")
    if args.count is not None and args.seed is not None:
        parser.error("Omit --seed for collection; use --sampler-seed for reproducible random scene seeds")
    if args.seed is not None and not 0 <= args.seed < 2**32:
        parser.error("Fixed scene seeds must be between 0 and 2**32-1")
    if args.max_failures_per_case <= 0:
        parser.error("--max-failures-per-case must be positive")
    if args.spawn_offset_y is None and any(layout not in LAYOUT_Y_OFFSETS for layout in args.layouts):
        parser.error("No placement rule for that layout; provide --spawn-offset-y explicitly")
    forbidden = {"--style", "--initial-door-angle-deg", "--record-dir", "--output-dir", "--robot-x", "--robot-y",
                 "--robot-yaw", "--spawn-mode", "--probe-only"}
    if any(arg.split("=")[0] in forbidden for arg in extra):
        parser.error("Scene identity, base placement and outputs are managed by this collector")
    cases = list(itertools.product(args.layouts, args.styles, args.angles))
    args.max_attempts = args.max_attempts if args.max_attempts is not None else (10*args.count if args.count else len(cases))
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    target = args.count if args.count else len(cases)
    quotas = {layout: target // len(args.layouts) + (index < target % len(args.layouts))
              for index, layout in enumerate(args.layouts)}
    offsets = {layout: args.spawn_offset_y if args.spawn_offset_y is not None else LAYOUT_Y_OFFSETS[layout]
               for layout in args.layouts}
    output = args.output.expanduser().resolve()
    if args.resume:
        try:
            previous = json.loads((output / "collection_config.json").read_text())
        except (OSError, ValueError):
            parser.error("No resumable collection_config.json; use a new output directory for older batches")
        if args.sampler_seed is None:
            args.sampler_seed = previous["sampler_seed"]
    if args.sampler_seed is None:
        args.sampler_seed = secrets.randbits(32)
    config = dict(version=2, layouts=args.layouts, styles=args.styles, angles=args.angles, count=args.count,
                  fixed_scene_seed=args.seed, sampler_seed=args.sampler_seed, offsets={str(k): v for k, v in offsets.items()},
                  spawn_offset_x=args.spawn_offset_x, base_local=args.base_local, headless=args.headless,
                  timeout_minutes=args.timeout_minutes, max_failures_per_case=args.max_failures_per_case,
                  motion_args=extra, python=sys.executable, source_sha256=source_hashes())
    if args.resume and config != previous:
        parser.error("Collection settings or source code changed; use the original settings or start a new output directory")
    print(json.dumps(dict(**config, layout_targets=quotas, max_attempts=args.max_attempts), indent=2), flush=True)
    if args.dry_run:
        return 0
    output.mkdir(parents=True, exist_ok=args.resume)
    with (output / "collection.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("This collection still has a running collector or child process")
        (output / "demos").mkdir(exist_ok=True)
        (output / "logs").mkdir(exist_ok=True)
        atomic_json(output / "collection_config.json", config)
        rng = random.Random(args.sampler_seed)
        state_path = output / "progress.json"
        state = json.loads(state_path.read_text()) if args.resume and state_path.exists() else dict(attempts=[], pending=None)
        if "rng_state" in state:
            version, internal, gaussian = state["rng_state"]
            rng.setstate((version, tuple(internal), gaussian))
        bases = {(row["layout"], row["style"], row["seed"]): row["base_pose"]
                 for row in state["attempts"] if len(row.get("base_pose", [])) == 6}
        if state["pending"] is not None:
            pending = state["pending"]
            # The inherited file lock prevents recovery while an orphan child is still writing.
            override = None if (Path(pending["directory"]) / "result.json").exists() else "interrupted"
            state["attempts"].append(finish_attempt(output, pending, None, override, bases))
            state["pending"] = None
        used_seeds = {row["seed"] for row in state["attempts"]}
        stopped = False
        handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while len(state["attempts"]) < args.max_attempts:
                saved = sum(row["success"] for row in state["attempts"])
                if args.count is not None and saved >= args.count:
                    break
                case_spec = choose_case(cases, state["attempts"], quotas, rng, args.count is None,
                                        args.max_failures_per_case)
                if case_spec is None:
                    break
                layout, style, angle = case_spec
                seed = args.seed
                if seed is None:
                    seed = rng.randrange(2**31)
                    while seed in used_seeds:
                        seed = rng.randrange(2**31)
                used_seeds.add(seed)
                attempt = len(state["attempts"]) + 1
                tag = f"demo_{saved+1:04d}_layout{layout}_style{style}_angle{angle:g}_seed{seed}"
                case, log_path = output / "demos" / tag, output / "logs" / f"attempt_{attempt:05d}.log"
                command = [sys.executable, "-u", "-m", "mobipi.collect_demos.auto_close_microwave_door",
                           "--layout", str(layout), "--style", str(style), "--seed", str(seed),
                           "--initial-door-angle-deg", str(angle), "--output-dir", str(case),
                           "--record-dir", str(case / "raw"), "--spawn-offset-x", str(args.spawn_offset_x),
                           "--spawn-offset-y", str(offsets[layout]), *extra]
                if args.headless:
                    command.append("--headless")
                if args.base_local is not None:
                    command += ["--base-local", *map(str, args.base_local)]
                state["pending"] = dict(demo=saved+1, attempt=attempt, layout=layout, style=style, angle_deg=angle,
                                        seed=seed, spawn_offset_y=offsets[layout], directory=str(case),
                                        log=str(log_path), command=command)
                state["rng_state"] = rng.getstate()
                write_progress(output, state, args, quotas)
                case.mkdir()
                print(f"Attempt {attempt}: layout={layout} style={style} angle={angle:g} seed={seed} "
                      f"offset_y={offsets[layout]} saved={saved}/{target}", flush=True)
                returncode, reason = run_attempt(command, log_path, args.timeout_minutes*60, lock.fileno())
                row = finish_attempt(output, state["pending"], returncode, reason, bases)
                state["attempts"].append(row)
                state["pending"] = None
                print(f"  success={row['success']} phase={row['phase']} reason={row['failure_reason']} "
                      f"detail={row['detail']}", flush=True)
                write_progress(output, state, args, quotas)
                if reason == "interrupted":
                    stopped = True
                    break
                if row["failure_reason"] in ("runtime_error", "missing_recording"):
                    print("Stopping after a runtime/recording error; inspect the attempt log.", flush=True)
                    break
        except KeyboardInterrupt:
            stopped = True
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            state["rng_state"] = rng.getstate()
            write_progress(output, state, args, quotas)
        saved = sum(row["success"] for row in state["attempts"])
        print(f"Saved {saved}/{target} complete demos after {len(state['attempts'])} attempts; {output}", flush=True)
        print("Saved counts and unfilled conditions: stats.json; all outcomes: attempts.csv", flush=True)
        if stopped:
            return 130
        if args.count is not None:
            return 0 if saved == args.count else 2
        return 0 if saved == len(cases) else 2


if __name__ == "__main__":
    raise SystemExit(main())
