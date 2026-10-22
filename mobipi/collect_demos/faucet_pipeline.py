"""Prepare, validate, replay and expand faucet seeds through installed MimicGen."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

import h5py
import imageio.v2 as imageio
import numpy as np

# Register the task, gripper and interface before MimicGen / RoboMimic create envs.
from mobipi.collect_demos.faucet_runtime import MG_RBY1TurnOnSinkFaucet, UnsafeTrajectory
from mobipi.collect_demos.collect_faucet_demos import Recorder, make_env
from mobipi.collect_demos.faucet_dataset import validate, write_episode


def prepare(source, output):
    from mimicgen.scripts.prepare_src_dataset import prepare_src_dataset
    validate(source)
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError("Prepared output must be new and separate from source")
    output.parent.mkdir(parents=True, exist_ok=True)
    # The upstream utility copies then edits; use a temporary file and validate first.
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        candidate = Path(temporary)/"prepared.hdf5"
        prepare_src_dataset(str(source), "MG_RBY1TurnOnSinkFaucet", "robosuite",
                            output_path=str(candidate))
        report = validate(candidate, prepared=True)
        # Exclusive-create publication prevents accidental replacement.
        with output.open("xb") as dest, candidate.open("rb") as src:
            import shutil
            shutil.copyfileobj(src, dest)
    print(json.dumps(report, indent=2))


@contextmanager
def rby1_waypoint_executor(settings, video, log):
    """Adapt only execution; upstream DataGenerator keeps segmentation and transforms.

    The installed fork appends hardcoded mobile actions and clips absolute commands
    even at zero noise. Scope this replacement to the generation call, restoring it
    in finally, and never modify the dependency checkout.
    """
    from mimicgen.datagen.waypoint import WaypointTrajectory
    original = WaypointTrajectory.execute

    def execute(trajectory, env, env_interface, **kwargs):
        if not hasattr(env_interface, "recorder"):
            env_interface.refresh()
            env_interface.recorder = Recorder(env, env_interface, settings, video, log)
            env_interface.segment = 0
        recorder = env_interface.recorder
        env_interface.segment += 1
        rows_start = len(recorder.rows)
        infos = []
        for sequence in trajectory.waypoint_sequences:
            for waypoint in sequence:
                if waypoint.noise not in (None, 0.):
                    raise ValueError("RB-Y1 executor currently requires action_noise=0")
                pose = waypoint.pose
                action = env_interface.full_action(pose, waypoint.gripper_action)
                infos.append(env_interface.get_datagen_info(action=action))
                recorder.step(pose, waypoint.gripper_action,
                              f"mimicgen_subtask_{env_interface.segment}")
        rows = recorder.rows[rows_start:]
        return dict(states=[r["states"] for r in rows], observations=[r["obs"] for r in rows],
                    datagen_infos=infos, actions=np.array([r["actions"] for r in rows]),
                    success=bool(env.is_success()["task"]))

    WaypointTrajectory.execute = execute
    try:
        yield
    finally:
        WaypointTrajectory.execute = original


def generate(source, output, config, target, max_attempts):
    from mimicgen.configs.task_spec import MG_TaskSpec
    from mimicgen.datagen.data_generator import DataGenerator
    validate(source, prepared=True)
    if not 1 <= target <= max_attempts:
        raise ValueError("Need 1 <= target successes <= max attempts")
    spec = MG_TaskSpec.from_json(json_dict=config["mimicgen"]["task_spec"])
    if any(s["action_noise"] != 0 for s in spec):
        raise ValueError("Absolute IK executor requires zero action noise")
    with h5py.File(source, "r") as f:
        keys = sorted(f["data"], key=lambda x: int(x.split("_")[-1]))
        meta = json.loads(f["data"].attrs["env_args"])
    # Retain source model / controller / cameras, permitting explicit reset overrides.
    effective = json.loads(json.dumps(config))
    effective["environment"] = meta["env_kwargs"]
    for key in ("has_renderer", "has_offscreen_renderer", "ignore_done", "use_object_obs",
                "use_camera_obs", "camera_depths", "controller_configs", "seed"):
        effective["environment"].pop(key, None)
    effective["environment"].update(config.get("generation_environment", {}))
    output.mkdir(parents=True, exist_ok=False)
    effective["source"] = str(source.resolve())
    effective["target_successes"] = target
    effective["max_attempts"] = max_attempts
    (output/"config.json").write_text(json.dumps(effective, indent=2))
    generator = DataGenerator(spec, str(source), demo_keys=keys)
    summary = dict(attempted=0, successful=0, success_rate=0., failures=[],
                   engine="mimicgen.datagen.data_generator.DataGenerator", source_demo_indices=[])
    np.random.seed(config["mimicgen"]["seed"])
    with h5py.File(output/"generated.hdf5", "x") as f:
        data = f.create_group("data")
        for attempt in range(max_attempts):
            if summary["successful"] >= target:
                break
            summary["attempted"] += 1
            env, video = None, None
            try:
                env = make_env(effective, config["mimicgen"]["seed"]+attempt)
                env.reset()
                interface = MG_RBY1TurnOnSinkFaucet(env.base_env)
                if attempt < 2:
                    video = imageio.get_writer(str(output/f"attempt_{attempt:03d}.mp4"), fps=20/3)
                with (output/f"attempt_{attempt:03d}.jsonl").open("x") as log:
                    with rby1_waypoint_executor(config["script"], video, log):
                        result = generator.generate(
                            env, interface,
                            select_src_per_subtask=config["mimicgen"]["select_src_per_subtask"],
                            interpolate_from_last_target_pose=config["mimicgen"]["interpolate_from_last_target_pose"])
                # Upstream success is accumulated across segments. Require final success too.
                if not result["success"] or not env.is_success()["task"]:
                    raise UnsafeTrajectory("MimicGen failed final faucet-on criterion")
                interface.geom.check()
                rows = interface.recorder.rows
                if not rows or rows[0]["diagnostics"]["water_on"]:
                    raise UnsafeTrajectory("Generated episode did not begin with faucet off")
                if not any(r["diagnostics"]["contact"] for r in rows):
                    raise UnsafeTrajectory("No physical handle contact")
                write_episode(data.create_group(f"demo_{summary['successful']}"), env,
                              result["initial_state"], rows, "mimicgen", result["src_demo_inds"], result["datagen_infos"])
                summary["source_demo_indices"].append([int(i) for i in result["src_demo_inds"]])
                summary["successful"] += 1
                f.flush()
            except UnsafeTrajectory as error:
                summary["failures"].append(dict(attempt=attempt, reason=str(error)))
            except Exception as error:
                summary["failures"].append(dict(attempt=attempt, reason=str(error), error_type=type(error).__name__))
                raise
            finally:
                if video is not None:
                    video.close()
                if env is not None:
                    env.env.close()
                summary["success_rate"] = summary["successful"]/summary["attempted"]
                (output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if summary["successful"]:
        validate(output/"generated.hdf5")
    return summary["successful"] >= target


def replay(source, video_path, config, demo):
    validate(source)
    if video_path.exists():
        raise FileExistsError(video_path)
    with h5py.File(source, "r") as f:
        ep = f["data"][demo]
        meta = json.loads(f["data"].attrs["env_args"])
        effective = dict(config)
        effective["environment"] = dict(meta["env_kwargs"])
        for key in ("has_renderer", "has_offscreen_renderer", "ignore_done", "use_object_obs",
                    "use_camera_obs", "camera_depths", "controller_configs", "seed"):
            effective["environment"].pop(key, None)
        env = make_env(effective, 0)
        try:
            env.reset_to(dict(model=ep.attrs["model_file"], ep_meta=ep.attrs["ep_meta"], states=ep["states"][0]))
            interface = MG_RBY1TurnOnSinkFaucet(env.base_env)
            max_error = 0.
            with imageio.get_writer(str(video_path), fps=20/3) as video:
                for t, action in enumerate(ep["actions"]):
                    current = env.get_state()["states"]
                    error = float(np.max(np.abs(current-ep["states"][t])))
                    max_error = max(max_error, error)
                    if error > 1e-3:
                        raise UnsafeTrajectory(f"Action replay diverged at frame {t}: {error}")
                    interface.preflight(action)
                    env.step(action)
                    interface.geom.check()
                    if t % 3 == 0:
                        video.append_data(env.render(mode="rgb_array", height=512, width=512,
                                                      camera_name=config["script"]["video_camera"]))
            if not env.is_success()["task"]:
                raise UnsafeTrajectory("Replayed commands did not solve task")
            print(json.dumps(dict(demo=demo, success=True, max_state_error=max_error)))
        finally:
            env.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "validate", "generate", "replay"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("faucet_config.json"))
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--target-successes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--demo", default="demo_0")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage == "validate":
        print(json.dumps(validate(args.source, args.prepared), indent=2))
        return
    if args.output is None:
        parser.error("--output is required for this stage")
    if args.stage == "prepare":
        prepare(args.source, args.output)
    elif args.stage == "replay":
        replay(args.source, args.output, config, args.demo)
    else:
        raise SystemExit(0 if generate(args.source, args.output, config, args.target_successes, args.max_attempts) else 1)


if __name__ == "__main__":
    main()
