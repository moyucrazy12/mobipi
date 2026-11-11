"""Prepare, validate, replay and expand faucet seeds through installed MimicGen."""
import argparse
from contextlib import contextmanager
import json
import io
from pathlib import Path

from rby1.paths import new_output
import tempfile

import h5py
import imageio.v2 as imageio
import numpy as np

# Register the task, gripper and interface before MimicGen / RoboMimic create envs.
from rby1.robot.faucet_runtime import MG_RBY1TurnOnSinkFaucet, UnsafeTrajectory
from rby1.demos.collect_faucet_demos import JointSpaceRecorder, make_env
from rby1.robot.faucet_runtime import raw_data
from rby1.demos.faucet_dataset import validate, write_episode


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
def rby1_waypoint_executor(settings, video, log, tracking=None):
    """Adapt only execution; upstream DataGenerator keeps segmentation and transforms.

    The installed fork appends hardcoded mobile actions and clips absolute commands
    even at zero noise. Scope this replacement to the generation call, restoring it
    in finally, and never modify the dependency checkout.
    """
    from mimicgen.datagen.waypoint import WaypointTrajectory
    original = WaypointTrajectory.execute
    if tracking is None:
        tracking = io.StringIO()

    def execute(trajectory, env, env_interface, **kwargs):
        if not hasattr(env_interface, "recorder"):
            env_interface.refresh()
            env_interface.recorder = JointSpaceRecorder(env, env_interface, settings, video, log, tracking)
            env_interface.recorder.datagen_infos = []
            env_interface.segment = 0
        recorder = env_interface.recorder
        env_interface.segment += 1
        rows_start = len(recorder.rows)
        for sequence in trajectory.waypoint_sequences:
            for waypoint in sequence:
                if waypoint.noise not in (None, 0.):
                    raise ValueError("RB-Y1 executor currently requires action_noise=0")
                pose = waypoint.pose
                phase = f"mimicgen_subtask_{env_interface.segment}"
                solver = recorder.bounded
                q = solver.optimize(pose, seed=solver.command)
                recorder.tracking.write(dict(event="bounded_ik", phase=phase,
                    target_tcp=pose, **solver.last))
                if settings.get("require_cartesian_tolerance", True) and (
                        solver.last["position_error"] > settings["contact_position_tolerance"]
                        or solver.last["orientation_error"] > .08):
                    raise UnsafeTrajectory(f"Bounded IK infeasible for {phase}: {solver.last}")
                previous = solver.command
                if previous is None:
                    previous = raw_data(env_interface.env.sim).qpos[solver.addresses]
                # Source targets already sample a smooth joint trajectory at the
                # control rate. Keep their timing (including gripper holds).
                # Interpolate larger transformed/interpolation jumps using the
                # same checked joint transition as the scripted collector.
                if np.max(np.abs(q-previous)) > settings["joint_velocity"]*env_interface.env.control_timestep:
                    recorder.transition(q, waypoint.gripper_action, phase)
                else:
                    recorder.step(solver.command_joints(q), waypoint.gripper_action, phase)
        rows = recorder.rows[rows_start:]
        return dict(states=[r["states"] for r in rows], observations=[r["obs"] for r in rows],
                    datagen_infos=recorder.datagen_infos[rows_start:], actions=np.array([r["actions"] for r in rows]),
                    success=bool(env.is_success()["task"]))

    WaypointTrajectory.execute = execute
    try:
        yield
    finally:
        WaypointTrajectory.execute = original


def generate(source, output, config, target, max_attempts, replay_source=None):
    from mimicgen.configs.task_spec import MG_TaskSpec
    from mimicgen.datagen.data_generator import DataGenerator
    validate(source, prepared=True)
    if not 1 <= target <= max_attempts:
        raise ValueError("Need 1 <= target successes <= max attempts")
    spec = MG_TaskSpec.from_json(json_dict=config["mimicgen"]["task_spec"])
    if any(s["action_noise"] != 0 for s in spec):
        raise ValueError("Absolute IK executor requires zero action noise")
    fixture_offsets = config["mimicgen"].get("fixture_offsets")
    if fixture_offsets is not None:
        offsets = np.asarray(fixture_offsets, dtype=float)
        if offsets.ndim != 2 or offsets.shape[1] != 4 or len(offsets) == 0 or not np.isfinite(offsets).all():
            raise ValueError("fixture_offsets must be a nonempty list of finite [dx, dy, dz, yaw] offsets")
    with h5py.File(source, "r") as f:
        keys = sorted(f["data"], key=lambda x: int(x.split("_")[-1]))
        selected_keys = config["mimicgen"].get("source_demo_keys")
        if selected_keys is not None:
            if not selected_keys or not set(selected_keys).issubset(keys):
                raise ValueError("source_demo_keys must select existing prepared demonstrations")
            keys = list(selected_keys)
        if replay_source is not None:
            if replay_source not in keys or target != 1 or max_attempts != 1:
                raise ValueError("Source execution requires one existing demo and one attempt")
            keys = [replay_source]
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
    effective["source_demo_keys"] = keys
    (output/"config.json").write_text(json.dumps(effective, indent=2))
    generator = DataGenerator(spec, str(source), demo_keys=keys)
    summary = dict(attempted=0, successful=0, success_rate=0., failures=[],
                   engine="mimicgen.datagen.data_generator.DataGenerator", source_demo_indices=[],
                   transforms_enabled=replay_source is None)
    np.random.seed(config["mimicgen"]["seed"])
    with h5py.File(output/"generated.hdf5", "x") as f:
        data = f.create_group("data")
        data.attrs["source_demo_keys"] = json.dumps(generator.demo_keys)
        for attempt in range(max_attempts):
            if summary["successful"] >= target:
                break
            summary["attempted"] += 1
            env, video = None, None
            interface = None
            attempt_config = json.loads(json.dumps(effective))
            if fixture_offsets is not None:
                attempt_config["environment"]["faucet_fixture_offset"] = fixture_offsets[attempt % len(fixture_offsets)]
            attempt_metadata = dict(attempt=attempt, seed=config["mimicgen"]["seed"]+attempt,
                requested_fixture_offset=attempt_config["environment"].get("faucet_fixture_offset", [0., 0., 0., 0.]))
            try:
                env = make_env(attempt_config, config["mimicgen"]["seed"]+attempt)
                # Construction already initializes the environment; native
                # DataGenerator.generate performs the generation hard reset.
                if replay_source is not None:
                    with h5py.File(source, "r") as src:
                        ep = src["data"][replay_source]
                        env.reset_to(dict(model=ep.attrs["model_file"], ep_meta=ep.attrs["ep_meta"], states=ep["states"][0]))
                interface = MG_RBY1TurnOnSinkFaucet(env.base_env)
                if attempt < config["mimicgen"].get("video_attempts", 2):
                    video = imageio.get_writer(str(output/f"attempt_{attempt:03d}.mp4"), fps=20/3)
                with (output/f"attempt_{attempt:03d}.jsonl").open("x") as log, \
                     (output/f"attempt_{attempt:03d}_tracking.jsonl").open("x") as tracking:
                    with rby1_waypoint_executor(config["script"], video, log, tracking):
                        if replay_source is None:
                            result = generator.generate(
                                env, interface,
                                select_src_per_subtask=config["mimicgen"]["select_src_per_subtask"],
                                interpolate_from_last_target_pose=config["mimicgen"]["interpolate_from_last_target_pose"])
                        else:
                            # Execution-only gate: original source targets, exact
                            # saved reset, native segmentation, no pose transform.
                            from mimicgen.datagen.waypoint import WaypointTrajectory, WaypointSequence
                            initial = env.get_state()
                            info = generator.src_dataset_infos[0]
                            for start, end in generator.src_subtask_indices[0]:
                                trajectory = WaypointTrajectory()
                                trajectory.add_waypoint_sequence(WaypointSequence.from_poses(
                                    info.target_pose[start:end], info.gripper_action[start:end], 0.))
                                trajectory.execute(env, interface)
                            result = dict(initial_state=initial, success=bool(env.is_success()["task"]),
                                src_demo_inds=[0]*len(spec), datagen_infos=interface.recorder.datagen_infos)
                        interface.recorder.tracking.terminate("complete")
                # Upstream success is accumulated across segments. Require final success too.
                if not result["success"] or not env.is_success()["task"]:
                    raise UnsafeTrajectory("MimicGen failed final faucet-on criterion")
                interface.geom.check()
                rows = interface.recorder.rows
                if not rows or rows[0]["diagnostics"]["water_on"]:
                    raise UnsafeTrajectory("Generated episode did not begin with faucet off")
                if not any(r["diagnostics"]["contact"] for r in rows):
                    raise UnsafeTrajectory("No physical handle contact")
                if (interface.segment != len(spec) or interface.geom.contact()
                        or interface.action_to_gripper_action(rows[-1]["actions"])[0] >= 0):
                    raise UnsafeTrajectory("Episode did not complete safe release and retreat")
                release_poses = [info.eef_pose for row, info in zip(rows, interface.recorder.datagen_infos)
                    if row["diagnostics"]["phase"] == f"mimicgen_subtask_{len(spec)}" and info.gripper_action[0] < 0]
                final_tcp = interface.get_robot_eef_pose()
                retreat = float(np.linalg.norm(final_tcp[:3, 3]-release_poses[0][:3, 3])) if release_poses else 0.
                if retreat < .08:
                    raise UnsafeTrajectory(f"Incomplete physical retreat: {retreat:.3f} m")
                attempt_metadata.update(retreat_distance_m=retreat, final_tcp=final_tcp.tolist(), final_contact=False)
                write_episode(data.create_group(f"demo_{summary['successful']}"), env,
                              result["initial_state"], rows, "mimicgen" if replay_source is None else "source_executor_replay",
                              result["src_demo_inds"], result["datagen_infos"])
                summary["source_demo_indices"].append([int(i) for i in result["src_demo_inds"]])
                summary.setdefault("source_demo_names", []).append([generator.demo_keys[int(i)] for i in result["src_demo_inds"]])
                summary["successful"] += 1
                attempt_metadata["successful"] = True
                f.flush()
            except UnsafeTrajectory as error:
                summary["failures"].append(dict(attempt=attempt, reason=str(error)))
                attempt_metadata.update(successful=False, reason=str(error))
                if env is not None and getattr(env.base_env, "sim", None) is not None:
                    state = env.get_state()
                    np.savez_compressed(output/f"attempt_{attempt:03d}_failed_state.npz", **state)
                if video is None and interface is not None and hasattr(interface, "recorder"):
                    rows = interface.recorder.rows
                    camera = config["script"]["video_camera"]+"_image"
                    if rows and camera in rows[0]["obs"]:
                        try:
                            with imageio.get_writer(str(output/f"attempt_{attempt:03d}_failed.mp4"), fps=20/3) as failed_video:
                                for row in rows[::3]:
                                    failed_video.append_data(row["obs"][camera])
                        except Exception as video_error:
                            attempt_metadata["failure_video_error"] = str(video_error)
            except Exception as error:
                summary["failures"].append(dict(attempt=attempt, reason=str(error), error_type=type(error).__name__))
                attempt_metadata.update(successful=False, reason=str(error), error_type=type(error).__name__)
                raise
            finally:
                if video is not None:
                    video.close()
                if env is not None:
                    if getattr(env.base_env, "sim", None) is not None:
                        summary["last_task_state"] = env.base_env.sink.get_handle_state(env.base_env)
                        summary["last_episode_success"] = bool(env.is_success()["task"])
                        if interface is not None:
                            summary["last_fixture_pose"] = interface.get_object_poses()["faucet"].tolist()
                            summary["last_base_pose"] = np.asarray(env.base_env.sim.data.get_body_xpos(env.base_env.robots[0].robot_model.root_body)).tolist()
                            attempt_metadata["fixture_pose"] = summary["last_fixture_pose"]
                            attempt_metadata["base_pose"] = summary["last_base_pose"]
                            attempt_metadata["final_task_state"] = summary["last_task_state"]
                    env.env.close()
                (output/f"attempt_{attempt:03d}_metadata.json").write_text(json.dumps(attempt_metadata, indent=2))
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
    parser.add_argument("stage", choices=["prepare", "validate", "generate", "replay", "execute-source"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--config", type=Path, help="External faucet task JSON; required for generation and replay")
    parser.add_argument("--prepared", action="store_true")
    parser.add_argument("--target-successes", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--demo", default="demo_0")
    args = parser.parse_args()
    if args.stage == "validate":
        print(json.dumps(validate(args.source, args.prepared), indent=2))
        return
    config = None
    if args.stage != "prepare":
        if args.config is None:
            parser.error("--config is required for generate, replay and execute-source")
        config = json.loads(args.config.expanduser().read_text())
    if args.output is None:
        if args.stage == "prepare":
            args.output = new_output("demos", "faucet_source", ".hdf5")
        elif args.stage == "replay":
            args.output = new_output("rollouts", "faucet_replay") / "replay.mp4"
        elif args.stage == "execute-source":
            args.output = new_output("diagnostics", "faucet_source_execution")
        else:
            args.output = new_output("demos", "faucet_mimicgen")
    if args.stage == "prepare":
        prepare(args.source, args.output)
    elif args.stage == "replay":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        replay(args.source, args.output, config, args.demo)
    elif args.stage == "execute-source":
        raise SystemExit(0 if generate(args.source, args.output, config, 1, 1, replay_source=args.demo) else 1)
    else:
        raise SystemExit(0 if generate(args.source, args.output, config, args.target_successes, args.max_attempts) else 1)


if __name__ == "__main__":
    main()
