"""Autonomous faucet evaluation using native relative DP and the proven IK executor."""
from rby1.paths import CONFIG_ROOT
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from rby1.paths import new_output, historical_path
import time

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

from robomimic.utils import file_utils as FileUtils, obs_utils as O
from robomimic.utils.rby1_evaluation import atomic_json, fingerprint
from robomimic.scripts.eval_rby1_cartesian_dp import seed_all
from robomimic.utils.rby1_relative_cartesian import tcp_pose, pose9_to_transform

RGB = ["faucet_overview_image", "robot0_right_eye_in_hand_image"]
LOWDIM = ["robot0_base_to_right_eef_pos", "robot0_base_to_right_eef_quat_site"]
DEFAULT_RUN = historical_path("train", "rby1_faucet_relative_cartesian_gripper_100d_15epochs", "20261027135336")
VALIDATION_DEMOS = ["demo_5", "demo_6", "demo_12", "demo_56", "demo_65"]
# Exact order from CloseSingleDoor/rby1_relative_final_125_eval/metadata.json.
GRID_SEEDS = [0, 1, 42, 123, 12345, *range(2, 22)]
RENDERING_METADATA = dict(
    historical_collection_renderer_configuration="unknown",
    image_comparison_policy="diagnostic only after strict physical and geometric restoration checks pass",
    policy_observations="actual rendered observations; preprocessing and inference unchanged")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def generation_seed(group):
    """Read the original scene-construction seed, not a new evaluation seed."""
    parts = Path(group.attrs["generation_directory"]).parts
    if not parts or parts[0] != "data" or ".." in parts:
        raise ValueError("Expected a recorded data/ generation directory")
    directory = historical_path("demos", *parts[1:])
    attempt = int(group.attrs["generation_attempt"])
    record = json.loads((directory / f"attempt_{attempt:03d}_metadata.json").read_text())
    config = json.loads((directory / "config.json").read_text())
    seed = int(record["seed"])
    if not record["successful"] or record["attempt"] != attempt or seed != config["mimicgen"]["seed"] + attempt:
        raise ValueError("Inconsistent demonstration generation provenance")
    np.testing.assert_array_equal(record["requested_fixture_offset"], group.attrs["fixture_offset"])
    return seed


def restore_validation_state(env, group):
    """Keep the original full-precision compiled scene; restore only its saved state.

    make_env followed by env.reset reproduces DataGenerator.generate's two model
    constructions, using the recorded generation seed. Recompiling model_file
    would instead load mj_saveLastXML's rounded numbers. Require an identical
    serialized model (including joint order, cameras, assets and controllers'
    actuators) before accepting the reconstruction. No physics steps follow the
    saved state assignment.
    """
    def elements(xml):
        return [(node.tag, dict(node.attrib)) for node in ET.fromstring(xml).iter()]

    expected_xml = group.attrs["model_file"]
    actual_xml = env.get_state()["model"]
    expected = elements(expected_xml)
    actual = elements(actual_xml)
    if actual_xml != expected_xml:
        first = next((i for i, pair in enumerate(zip(actual, expected)) if pair[0] != pair[1]),
                     min(len(actual), len(expected)))
        raise ValueError(f"Rebuilt demonstration model differs from saved XML at element {first}: "
                         f"actual={actual[first:first+1]}, saved={expected[first:first+1]}")
    sim = env.base_env.sim
    sim.reset()
    sim.set_state_from_flattened(group["states"][0])
    sim.forward()
    env.base_env.update_state()


def validation_configurations(dataset, demos):
    """Inspect saved starts without constructing a simulator or changing the HDF5."""
    configurations = []
    if not demos or len(set(demos)) != len(demos):
        raise ValueError("Demonstration keys must be nonempty and unique")
    with h5py.File(dataset, "r") as f:
        valid = set(f["mask/valid"].asstr()[:])
        metadata = json.loads(f["data"].attrs["env_args"])
        if metadata["env_name"] != "TurnOnSinkFaucetRBY1":
            raise ValueError("Expected the faucet dataset")
        for index, name in enumerate(demos, 1):
            if name not in valid:
                raise ValueError(f"{name} is not a held-out validation demonstration")
            group = f[f"data/{name}"]
            xml = group.attrs["model_file"]
            root = ET.fromstring(xml)
            state = group["states"][0]
            meta = json.loads(group.attrs["ep_meta"])
            sink = root.find(".//body[@name='sink_island_group_main']")
            if sink is None or not np.isfinite(state).all():
                raise ValueError(f"Invalid saved start: {name}")
            # These saved demos use the fixed mount and a floating robot root,
            # not today's default virtual mobile base. Preserve their model.
            if root.find(".//body[@name='fixed_mount0_base']") is None:
                raise ValueError(f"Unsupported saved robot base in {name}")
            for asset in root.findall("./asset/*"):
                filename = asset.get("file")
                if filename and not Path(filename).is_file():
                    raise FileNotFoundError(f"{name}: saved XML asset missing: {filename}")
            if float(group["diagnostics/handle_angle"][0]) != 0.:
                raise ValueError(f"{name} does not begin with the faucet off")
            configurations.append(dict(
                id=f"config_{index:02d}_{name}", demo_key=name, split="valid",
                fixture_offset_xyz_yaw=np.asarray(group.attrs["fixture_offset"]).tolist(),
                fixture_position=list(map(float, sink.get("pos").split())),
                fixture_quaternion_wxyz=list(map(float, sink.get("quat").split())),
                state_sha256=digest(state.tobytes()), model_sha256=digest(xml.encode()),
                ep_meta_sha256=digest(group.attrs["ep_meta"].encode()),
                state_length=len(state), source_demo_names=json.loads(group.attrs["source_demo_names"]),
                generation_seed=generation_seed(group),
                objects=[dict(name=o["name"], category=o["info"]["cat"]) for o in meta["object_cfgs"]]))
    signatures = {(tuple(c["fixture_position"]), tuple(c["fixture_quaternion_wxyz"])) for c in configurations}
    if len(signatures) != len(configurations):
        raise ValueError("Selected configurations must have distinct faucet transforms")
    if len({c["state_sha256"] for c in configurations}) != len(configurations):
        raise ValueError("Selected configurations repeat a saved state")
    return configurations


def verify_physical_start(env, interface, initial, group, base_pose):
    """Every model, state and geometry failure remains fatal, before image diagnostics."""
    np.testing.assert_array_equal(env.base_env.sim.get_state().flatten(), group["states"][0])
    if env.get_state()["model"] != group.attrs["model_file"]:
        raise ValueError("Restored XML differs from the saved demonstration (including camera geometry)")
    root = ET.fromstring(group.attrs["model_file"])
    for key in RGB:
        if root.find(f".//camera[@name='{key.removesuffix('_image')}']") is None:
            raise ValueError(f"Saved camera geometry missing: {key}")
    offset = 1  # Flattened simulator state starts with time, followed by qpos.
    saved_joints = {}
    for joint in root.find("worldbody").iter("joint"):
        width = {"free": 7, "ball": 4}.get(joint.get("type", "hinge"), 1)
        name = joint.get("name")
        expected = group["states"][0, offset:offset+width]
        saved_joints[name] = expected
        np.testing.assert_array_equal(
            np.atleast_1d(env.base_env.sim.data.get_joint_qpos(name)), expected,
            err_msg=f"Restored joint mismatch: {name}")
        offset += width
    expected_base = tcp_pose(group["obs/robot0_base_pos"][0], group["obs/robot0_base_quat"][0])
    np.testing.assert_allclose(base_pose, expected_base, atol=2e-6, rtol=0,
                               err_msg="Restored base transform mismatch")
    for key in LOWDIM:
        np.testing.assert_allclose(initial[key], group[f"obs/{key}"][0], atol=2e-6, rtol=0,
                                   err_msg=f"Restored initial observation mismatch: {key}")
    np.testing.assert_allclose(interface.get_robot_eef_pose(), group["diagnostics/actual_tcp"][0],
                               atol=2e-6, rtol=0, err_msg="Restored world TCP mismatch")
    fixtures = interface.get_object_poses()
    if set(fixtures) != {"faucet", "handle"}:
        raise ValueError("Restored faucet/handle references differ")
    for name, pose in fixtures.items():
        np.testing.assert_allclose(pose, group[f"datagen_info/object_poses/{name}"][0],
                                   atol=2e-6, rtol=0, err_msg=f"Restored fixture mismatch: {name}")
    meta = json.loads(group.attrs["ep_meta"])
    expected_objects = {o["name"] for o in meta["object_cfgs"]}
    if set(env.base_env.objects) != expected_objects:
        raise ValueError("Restored distractor object identities differ")
    for name, obj in env.base_env.objects.items():
        q = saved_joints[name + "_joint0"]
        expected_pose = tcp_pose(q[:3], np.roll(q[3:], -1))  # MuJoCo free-joint quaternion is wxyz.
        np.testing.assert_allclose(interface.get_object_pose(obj.root_body, "body"), expected_pose,
                                   atol=2e-6, rtol=0, err_msg=f"Restored object mismatch: {name}")


def image_diagnostics(initial, group):
    """Compare unchanged policy inputs; metrics use original 8-bit RGB differences."""
    cameras = {}
    for key in RGB:
        expected = O.process_obs(group[f"obs/{key}"][0], obs_key=key)
        actual = np.asarray(initial[key])
        if (actual.shape != expected.shape or not np.isfinite(actual).all()
                or actual.min() < 0 or actual.max() > 1):
            raise ValueError(f"Invalid rendered image observation: {key}")
        # Both images originate as uint8 RGB. Recover levels only in local metric
        # arrays, avoiding float32 roundoff at the one-level threshold. Neither
        # the input arrays nor the policy observation history is modified.
        error_255 = np.abs(np.rint(actual.astype(np.float64) * 255.)
                           - np.rint(expected.astype(np.float64) * 255.))
        cameras[key] = dict(
            mae=float(error_255.mean() / 255.), maximum_absolute_error=float(error_255.max() / 255.),
            mae_255=float(error_255.mean()), maximum_absolute_error_255=float(error_255.max()),
            fraction_exceeding_1_over_255=float(np.mean(error_255 > 1)),
            channels_exceeding_1_over_255=int(np.count_nonzero(error_255 > 1)),
            total_channels=int(error_255.size),
            original_comparison_passed=bool(np.allclose(actual, expected, atol=1./255 + 1e-7, rtol=0)))
    return dict(**RENDERING_METADATA, physical_and_geometric_restoration_verified=True,
                original_comparison_atol=1./255 + 1e-7, original_comparison_rtol=0.,
                metric_units="normalized RGB; derived from original 8-bit channel differences",
                cameras=cameras)


def verify_restored_start(env, interface, initial, group, base_pose, output):
    verify_physical_start(env, interface, initial, group, base_pose)
    diagnostics = image_diagnostics(initial, group)
    atomic_json(output / "image_diagnostics.json", diagnostics)
    differences = [f"{key}: MAE={d['mae_255']:.3f}/255, max={d['maximum_absolute_error_255']:.0f}/255, "
                   f">1/255={d['fraction_exceeding_1_over_255']:.1%}"
                   for key, d in diagnostics["cameras"].items() if d["maximum_absolute_error_255"] > 0]
    if differences:
        print(f"WARNING: {group.name} image differences after physical/geometry checks passed; "
              + "; ".join(differences) + ". Historical collection renderer unknown.", flush=True)
    return diagnostics


def save_initial_state(env, interface, output):
    """Persist full replay inputs and measured poses before the first policy action."""
    state = env.get_state()
    with (output / "initial_state.npy").open("xb") as stream:
        np.save(stream, state["states"], allow_pickle=False)
    with (output / "initial_model.xml").open("x") as stream:
        stream.write(state["model"])
    model, data = env.base_env.sim.model, env.base_env.sim.data
    atomic_json(output / "initial.json", dict(
        ep_meta=state.get("ep_meta"), simulation_time=float(data.time),
        state_sha256=digest(np.asarray(state["states"]).tobytes()),
        joint_positions={model.joint_id2name(j): np.asarray(data.get_joint_qpos(model.joint_id2name(j))).tolist()
                         for j in range(model.njnt)},
        tcp_world=interface.get_robot_eef_pose().tolist(),
        faucet_state=env.base_env.sink.get_handle_state(env.base_env),
        object_poses={k: interface.get_object_pose(v.root_body, "body").tolist()
                      for k, v in env.base_env.objects.items()},
        fixture_poses={k: v.tolist() for k, v in interface.get_object_poses().items()}))


def load_policy(path, device):
    policy, checkpoint = FileUtils.policy_from_checkpoint(ckpt_path=str(path), device=torch.device(device))
    config = policy.policy.global_config
    assert config.algo_name == "diffusion_policy" and checkpoint["shape_metadata"]["ac_dim"] == 10
    assert config.algo.relative_actions.enabled and config.algo.relative_actions.arm == "right"
    assert list(config.train.action_keys) == ["actions_eef_cmd", "actions_gripper"]
    assert list(config.observation.modalities.obs.rgb) == RGB
    assert list(config.observation.modalities.obs.low_dim) == LOWDIM
    assert not config.train.hdf5_normalize_obs
    assert dict(config.algo.horizon) == dict(observation_horizon=2, prediction_horizon=16, action_horizon=8)
    assert config.train.frame_stack == 2 and config.algo.ddim.enabled
    assert config.algo.ddim.num_inference_timesteps == 10
    assert checkpoint["env_metadata"]["env_name"] == "TurnOnSinkFaucetRBY1"
    assert set(policy.action_normalization_stats) == {"actions_eef_cmd", "actions_gripper"}
    for key in config.train.action_keys:
        assert config.train.action_config[key]["normalization"] == "min_max"
    expected = {**{key: [3, 128, 128] for key in RGB}, LOWDIM[0]: [3], LOWDIM[1]: [4]}
    assert {key: list(shape) for key, shape in checkpoint["shape_metadata"]["all_shapes"].items()} == expected
    assert config.algo.ema.enabled and policy.policy.ema is not None
    assert checkpoint["model"].get("ema") is not None
    return policy, checkpoint


def check_shapes(policy, dataset, seed):
    """One inference on two training observations, without creating an environment."""
    with h5py.File(dataset, "r") as f:
        name = f["mask/train"].asstr()[0]
        ep = f[f"data/{name}/obs"]
        inputs = {key: np.stack([O.process_obs(ep[key][t], obs_key=key) for t in (0, 1)]) for key in RGB}
        inputs.update({key: ep[key][:2] for key in LOWDIM})
    seed_all(seed)
    policy.start_episode()
    with torch.no_grad():
        action = policy(inputs)
    assert action.shape == (10,) and np.isfinite(action).all()
    pose9_to_transform(action[:9])
    np.testing.assert_allclose(policy.policy.relative_anchor, tcp_pose(inputs[LOWDIM[0]][-1], inputs[LOWDIM[1]][-1]), atol=1e-7)
    return dict(observation_shapes={k: list(v.shape) for k, v in inputs.items()},
                action_shape=list(action.shape), action=action.tolist(), queued_actions=len(policy.policy.action_queue))


def read_observation(env, interface):
    """Same image processing and right-TCP reconstruction as training conversion."""
    raw = env.get_observation()  # Native wrapper applies the dataset's image flip.
    base = tcp_pose(raw["robot0_base_pos"], raw["robot0_base_quat"])
    measured = np.linalg.inv(base) @ interface.get_robot_eef_pose()
    from scipy.spatial.transform import Rotation
    return {**{key: O.process_obs(raw[key], obs_key=key) for key in RGB},
            LOWDIM[0]: measured[:3, 3].copy(), LOWDIM[1]: Rotation.from_matrix(measured[:3, :3]).as_quat()}, base


class HorizonReached(Exception):
    pass


def rollout(policy, checkpoint, args, seed, output, configuration=None):
    # Task registration and simulator dependencies are needed only for real rollouts.
    from rby1.demos.collect_faucet_demos import make_env, JointSpaceRecorder
    from rby1.robot.faucet_runtime import MG_RBY1TurnOnSinkFaucet, UnsafeTrajectory
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads(Path(args.task_config).read_text())
    settings["environment"] = json.loads(json.dumps(checkpoint["env_metadata"]["env_kwargs"]))
    for key in ("controller_configs", "seed", "has_renderer", "has_offscreen_renderer", "ignore_done",
                "use_object_obs", "use_camera_obs", "camera_depths"):
        settings["environment"].pop(key, None)
    settings["environment"]["faucet_fixture_offset"] = [args.fixture_offset[0], args.fixture_offset[1], 0., np.deg2rad(args.fixture_offset[2])]
    if configuration is not None:
        settings["seed"] = configuration["generation_seed"]
        settings["environment"]["base_types"] = "RBY1Mount"
        settings["environment"]["faucet_fixture_offset"] = configuration["fixture_offset_xyz_yaw"]
        settings["initial_configuration"] = configuration
        settings["rendering_validation"] = dict(RENDERING_METADATA)
    # A fixed 2500-step evaluation lasts 125 s, independently of collector timeout.
    settings["script"]["max_rollout_seconds"] = args.max_steps / 20 + 1.
    atomic_json(output / "config.json", settings)
    result = dict(diffusion_seed=seed, environment_seed=args.env_seed, fixture_offset_xyz_yaw=settings["environment"]["faucet_fixture_offset"],
                  rollout_length=0, activation=False, physical_contact=False, contact_steps=0, release=False,
                  retreat=False, full_success=False, final_water_on=False, final_faucet_angle=None,
                  maximum_faucet_angle=None, activation_step=None, release_step=None,
                  final_retreat_m=0., maximum_retreat_m=0., policy_queries=0, action_chunks=0,
                  termination="initialization_failure", error=None, video=str(output / "rollout.mp4"))
    env, recorder = None, None
    started = time.monotonic()
    try:
        construction_seed = args.env_seed if configuration is None else configuration["generation_seed"]
        result["environment_seed"] = construction_seed
        seed_all(construction_seed)
        env = make_env(settings, construction_seed)
        if configuration is None:
            env.reset()
        else:
            # Same second hard reset used when this demonstration was generated.
            env.reset()
            with h5py.File(args.dataset, "r") as f:
                group = f["data/" + configuration["demo_key"]]
                restore_validation_state(env, group)
            # Reset controller goals to the restored joints, without stepping physics.
            for controller in env.base_env.robots[0].part_controllers.values():
                controller.update(force=True)
                controller.reset_goal()
        interface = MG_RBY1TurnOnSinkFaucet(env.base_env)
        O.initialize_obs_utils_with_config(policy.policy.global_config)
        initial, base = read_observation(env, interface)
        result["initial_fixture_pose"] = interface.get_object_poses()["faucet"].tolist()
        result["initial_base_pose"] = base.tolist()
        if configuration is not None:
            # Preserve diagnostics even when a restoration check fails.
            save_initial_state(env, interface, output)
            np.savez_compressed(output / "initial_images.npz", **{key: initial[key] for key in RGB})
            with h5py.File(args.dataset, "r") as f:
                diagnostics = verify_restored_start(
                    env, interface, initial, f["data/" + configuration["demo_key"]], base, output)
            result.update(initialization_verified=True, configuration_id=configuration["id"],
                          demo_key=configuration["demo_key"],
                          initial_state_sha256=configuration["state_sha256"], image_diagnostics=diagnostics)
        history = deque([initial, initial], maxlen=2)
        release_pose = None
        last_gripper = None

        def measure(gripper, count_contact=True):
            nonlocal release_pose, last_gripper
            state = env.base_env.sink.get_handle_state(env.base_env)
            angle, water = float(state["handle_joint"]), bool(state["water_on"])
            contact = bool(interface.geom.contact())
            tcp = interface.get_robot_eef_pose()[:3, 3]
            last_gripper = gripper
            result["final_faucet_angle"], result["final_water_on"] = angle, water
            previous = result["maximum_faucet_angle"]
            result["maximum_faucet_angle"] = angle if previous is None else max(previous, angle)
            result["physical_contact"] |= contact
            result["contact_steps"] += int(contact and count_contact)
            if water and result["activation_step"] is None:
                result["activation_step"] = result["rollout_length"]
            result["activation"] |= water
            if result["activation"] and result["physical_contact"] and gripper < 0 and not contact and release_pose is None:
                release_pose = tcp.copy()
                result["release_step"] = result["rollout_length"]
                result["release"] = True
            if release_pose is not None:
                distance = float(np.linalg.norm(tcp - release_pose))
                result["final_retreat_m"] = distance
                result["maximum_retreat_m"] = max(result["maximum_retreat_m"], distance)
            result["final_contact"] = contact
            result["final_gripper_command"] = float(gripper)

        class EvaluationRecorder(JointSpaceRecorder):
            def step(self, pose, gripper=1., phase=None):
                if result["rollout_length"] >= args.max_steps:
                    raise HorizonReached()
                before = float(self.interface.env.sim.data.time)
                try:
                    return super().step(pose, gripper, phase)
                finally:
                    if float(self.interface.env.sim.data.time) > before:
                        result["rollout_length"] += 1
                        measure(float(np.asarray(gripper).item()))
                        current, _ = read_observation(env, interface)
                        history.append(current)  # Last TWO control frames, including interpolation.

        with imageio.get_writer(str(output / "rollout.mp4"), fps=20/3) as video, \
             (output / "control.jsonl").open("x") as log, \
             (output / "tracking.jsonl").open("x") as tracking, \
             (output / "policy_actions.jsonl").open("x") as actions:
            recorder = EvaluationRecorder(env, interface, settings["script"], video, log, tracking)
            solver = recorder.bounded
            if env.is_success()["task"]:
                raise ValueError("Fresh evaluation starts with water already on")
            seed_all(seed)
            policy.start_episode()
            result["termination"] = "horizon"
            try:
                while result["rollout_length"] < args.max_steps:
                    inputs = {key: np.stack([item[key] for item in history]) for key in RGB + LOWDIM}
                    result["action_chunks"] += int(not policy.policy.action_queue)
                    with torch.no_grad():
                        action = np.asarray(policy(inputs), dtype=float)
                    result["policy_queries"] += 1
                    if action.shape != (10,) or not np.isfinite(action).all() or abs(action[9]) > 1.0001:
                        raise UnsafeTrajectory("Nonfinite or invalid decoded policy action/gripper")
                    _, base = read_observation(env, interface)
                    target = base @ pose9_to_transform(action[:9])
                    gripper = float(np.clip(action[9], -1., 1.))  # Only numerical boundary roundoff.
                    actions.write(json.dumps(dict(control_step=result["rollout_length"], action=action.tolist(),
                        target_world=target.tolist(), anchor=policy.policy.relative_anchor.tolist(),
                        queue_remaining=len(policy.policy.action_queue))) + "\n")
                    actions.flush()
                    q = solver.optimize(target, seed=solver.command)
                    recorder.tracking.write(dict(event="bounded_ik", phase="policy", target_tcp=target, **solver.last))
                    # Numerical termination is diagnostic, not proof of physical
                    # infeasibility: an iteration-limited solve can already have
                    # micron-scale residuals. Use the existing faucet pose gates,
                    # including for numerically converged but inaccurate solves.
                    # command_joints / transition still check joint limits,
                    # collisions, discontinuities and the executed joint path.
                    if (not np.isfinite(q).all()
                            or not np.isfinite([solver.last["position_error"], solver.last["orientation_error"]]).all()
                            or solver.last["position_error"] > settings["script"]["contact_position_tolerance"]
                            or solver.last["orientation_error"] > .08):
                        raise UnsafeTrajectory(f"Bounded IK target infeasible: {solver.last}")
                    previous = solver.command
                    if previous is None:
                        previous = np.asarray(interface.env.sim.data.qpos[solver.addresses])
                    # Same retiming decision and joint interpolation used by MimicGen.
                    if np.max(np.abs(q - previous)) > settings["script"]["joint_velocity"] * interface.env.control_timestep:
                        recorder.transition(q, gripper, "policy")
                    else:
                        recorder.step(solver.command_joints(q), gripper, "policy")
            except HorizonReached:
                pass
            video.append_data(env.render(mode="rgb_array", height=512, width=512, camera_name=settings["script"]["video_camera"]))
            measure(last_gripper if last_gripper is not None else -1., count_contact=False)
            result["retreat"] = bool(result["release"] and result["final_retreat_m"] >= .08
                                      and not result["final_contact"] and result["final_gripper_command"] < 0)
            result["full_success"] = bool(result["activation"] and result["physical_contact"] and result["retreat"] and result["final_water_on"])
    except Exception as error:
        result.update(termination="controller_or_runtime_failure", error=str(error), error_type=type(error).__name__, full_success=False)
        if env is not None and getattr(env.base_env, "sim", None) is not None:
            state = env.base_env.sink.get_handle_state(env.base_env)
            result.update(final_faucet_angle=float(state["handle_joint"]), final_water_on=bool(state["water_on"]))
    finally:
        if env is not None:
            try:
                env.env.close()
            except Exception as error:
                result["cleanup_error"] = str(error)
        result["timeout"] = result["termination"] == "horizon" and result["rollout_length"] == args.max_steps
        result["wall_seconds"] = time.monotonic() - started
        atomic_json(output / "result.json", result)
    return result


def grid_summary(protocol, results):
    def metrics(rows):
        return dict(completed=len(rows),
                    **{key: sum(bool(r[key]) for r in rows)
                       for key in ("full_success", "activation", "physical_contact")},
                    **{key + "_rate": sum(bool(r[key]) for r in rows) / len(rows) if rows else None
                       for key in ("full_success", "activation", "physical_contact")})
    return dict(planned_rollouts=len(protocol["configurations"]) * len(protocol["seeds"]),
                **metrics(results),
                configurations={c["id"]: metrics([r for r in results if r["configuration_id"] == c["id"]])
                                for c in protocol["configurations"]}, results=results)


def read_grid_result(directory, protocol_sha256):
    path = directory / "result.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if result.get("completed") is not True:
        return None
    if result.get("protocol_sha256") != protocol_sha256:
        raise ValueError(f"Completed rollout has a different protocol: {path}")
    if not result.get("initialization_verified") or any(
            type(result.get(key)) is not bool for key in ("full_success", "activation", "physical_contact")):
        raise ValueError(f"Invalid completed rollout: {path}")
    return result


def archive_partial(directory):
    """Keep interrupted artifacts; restart the trial in a new, exclusive directory."""
    if directory.exists():
        destination = directory.with_name(f"{directory.name}_interrupted_{time.time_ns()}")
        directory.rename(destination)


def validation_grid(args, checkpoints):
    if len(checkpoints) != 1:
        raise ValueError("Validation grid requires one explicit checkpoint or epoch")
    if args.check_only or any(args.fixture_offset):
        raise ValueError("Validation restores saved fixture transforms; do not combine with --fixture-offset or --check-only")
    args.dataset = args.dataset.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    # Results must live outside working repositories, including nested worktrees.
    if any((p / ".git").is_file() or (p / ".git/HEAD").is_file()
           for p in (args.output, *args.output.parents)):
        raise ValueError("Validation grid output must be outside Git repositories")
    configurations = validation_configurations(args.dataset, args.demo_keys or VALIDATION_DEMOS)
    protocol = dict(schema_version=1, initialization="validation", checkpoint=fingerprint(checkpoints[0]),
                    dataset=fingerprint(args.dataset), configurations=configurations, seeds=args.seeds,
                    rendering_validation=dict(RENDERING_METADATA),
                    environment_seed_source="configurations[].generation_seed",
                    fallback_environment_seed=args.env_seed, max_steps=args.max_steps, device=args.device,
                    task_config_sha256=digest(args.task_config.read_bytes()),
                    evaluator_sha256=digest(Path(__file__).read_bytes()))
    protocol_sha256 = digest(json.dumps(protocol, sort_keys=True).encode())
    if args.plan_only:
        print(json.dumps(dict(protocol=protocol, **grid_summary(protocol, [])), indent=2))
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"Use --resume for this existing evaluation: {args.output}")
    if args.resume and not (args.output / "manifest.json").is_file():
        raise ValueError("--resume requires an existing grid manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    # A second terminal must not write the same grid concurrently.
    import fcntl
    with (args.output / ".evaluation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.output / "manifest.json"
        if args.resume:
            if json.loads(manifest.read_text()) != protocol:
                raise ValueError("Resume protocol mismatch (inputs, code, seeds, horizon or device changed)")
        else:
            atomic_json(manifest, protocol)
        trials = [(index, c, seed, args.output / c["id"] / f"seed_{seed}")
                  for index, c in enumerate(configurations, 1) for seed in args.seeds]
        results, pending = [], []
        for index, config, seed, directory in trials:
            result = read_grid_result(directory, protocol_sha256)
            if result is None:
                pending.append((index, config, seed, directory))
            else:
                if result["configuration_id"] != config["id"] or result["diffusion_seed"] != seed:
                    raise ValueError(f"Rollout identity mismatch: {directory}")
                results.append(result)

        def progress(index, config, seed):
            summary = grid_summary(protocol, results)
            print(f"Configuration {index}/{len(configurations)} ({config['demo_key']}); diffusion seed {seed}; "
                  f"completed {len(results)}/{summary['planned_rollouts']}; "
                  f"success={summary['full_success']} activation={summary['activation']} "
                  f"contact={summary['physical_contact']}; results={args.output}", flush=True)
            atomic_json(args.output / "summary.json", summary)

        if not pending:
            progress(len(configurations), configurations[-1], args.seeds[-1])
            return
        policy, checkpoint = load_policy(checkpoints[0], args.device)
        for index, configuration, seed, directory in pending:
            progress(index, configuration, seed)
            archive_partial(directory)
            result = rollout(policy, checkpoint, args, seed, directory, configuration)
            if not result.get("initialization_verified"):
                raise RuntimeError(f"Saved initialization failed; grid stopped: {directory / 'result.json'}")
            result.update(checkpoint=checkpoints[0].stem, checkpoint_path=str(checkpoints[0]),
                          protocol_sha256=protocol_sha256, completed=True)
            atomic_json(directory / "result.json", result)
            results.append(result)
            progress(index, configuration, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint; overrides --epochs")
    parser.add_argument("--epochs", type=int, nargs="+", default=[9, 13, 15])
    parser.add_argument("--seeds", type=int, nargs="+", help="Fresh default: 0 1 42; validation default: CloseSingleDoor's 25 seeds")
    parser.add_argument("--initialization", choices=("fresh", "validation"), default="fresh")
    parser.add_argument("--demo-keys", nargs="+", help="Held-out saved starts; validation default: demo_5 demo_6 demo_12 demo_56 demo_65")
    parser.add_argument("--resume", action="store_true", help="Resume an identical validation grid; completed trials are preserved")
    parser.add_argument("--plan-only", action="store_true", help="Inspect validation grid inputs only; no policy, simulator, or output writes")
    parser.add_argument("--env-seed", type=int, default=0, help="Same fresh start for every checkpoint and diffusion seed")
    parser.add_argument("--fixture-offset", type=float, nargs=3, default=[0., 0., 0.], metavar=("DX", "DY", "YAW_DEG"))
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true", help="One offline checkpoint/input/action check; no environment")
    parser.add_argument("--dataset", type=Path, default=historical_path("demos", "faucet_relative_cartesian_100", "train.hdf5"))
    parser.add_argument("--task-config", type=Path, default=(CONFIG_ROOT / "faucet_config.json"))
    args = parser.parse_args()
    args.seeds = args.seeds if args.seeds is not None else (GRID_SEEDS.copy() if args.initialization == "validation" else [0, 1, 42])
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or max(args.seeds) >= 2**32:
        parser.error("Diffusion seeds must be unique integers in [0, 2**32)")
    if args.initialization == "fresh" and (args.demo_keys or args.resume or args.plan_only):
        parser.error("--demo-keys, --resume and --plan-only require --initialization validation")
    args.output = args.output or new_output("diagnostics" if args.check_only else "rollouts", "faucet_dp")
    if args.max_steps < 1 or args.max_steps > 2500 or not np.isfinite(args.fixture_offset).all():
        parser.error("Horizon must be 1..2500 and offsets finite")
    if max(abs(x) for x in args.fixture_offset[:2]) > .02 or abs(args.fixture_offset[2]) > 2:
        parser.error("Evaluation offsets are limited to ±2 cm XY and ±2 degrees yaw")
    if args.checkpoint:
        checkpoints = [args.checkpoint.resolve()]
    else:
        run = args.run_dir
        if not (run / "models").exists():
            runs = sorted(p for p in run.iterdir() if (p / "models").is_dir())
            if len(runs) != 1:
                parser.error("Specify the timestamped --run-dir when multiple runs exist")
            run = runs[0]
        checkpoints = [(run / "models" / f"model_epoch_{epoch}.pth").resolve() for epoch in args.epochs]
    if not all(p.is_file() for p in checkpoints):
        parser.error("Requested checkpoint is missing")
    if args.initialization == "validation":
        validation_grid(args, checkpoints)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    summary = dict(settings={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, checkpoints=[], results=[])
    for path in checkpoints[:1] if args.check_only else checkpoints:
        policy, checkpoint = load_policy(path, args.device)
        identity = dict(checkpoint=path.stem, **fingerprint(path))
        summary["checkpoints"].append(identity)
        if args.check_only:
            summary["shape_check"] = check_shapes(policy, args.dataset, args.seeds[0])
        else:
            for seed in args.seeds:
                result = rollout(policy, checkpoint, args, seed, args.output / path.stem / f"seed_{seed}")
                result.update(checkpoint=path.stem, checkpoint_path=str(path))
                atomic_json(args.output / path.stem / f"seed_{seed}" / "result.json", result)
                summary["results"].append(result)
                rows = [r for r in summary["results"] if r["checkpoint"] == path.stem]
                identity.update(rollouts=len(rows), activation_successes=sum(r["activation"] for r in rows),
                    full_successes=sum(r["full_success"] for r in rows),
                    activation_rate=sum(r["activation"] for r in rows)/len(rows),
                    full_success_rate=sum(r["full_success"] for r in rows)/len(rows))
                atomic_json(args.output / "summary.json", summary)
        del policy, checkpoint
    atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
