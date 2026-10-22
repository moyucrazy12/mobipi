"""Scripted right-arm faucet seeds, using the existing collector's IK/action helpers."""
import argparse
import json
import math
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.controllers import load_composite_controller_config
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.obs_utils as ObsUtils

from mobipi.collect_demos.faucet_runtime import (
    CONTROLLER, MG_RBY1TurnOnSinkFaucet, UnsafeTrajectory, make_T,
    raw_data,
)
from mobipi.collect_demos.auto_close_microwave_door import interpolate_rotation, lerp_points
from mobipi.collect_demos.faucet_dataset import write_episode


def make_env(config, seed):
    kwargs = dict(config["environment"])
    kwargs["controller_configs"] = load_composite_controller_config(
        controller=str(CONTROLLER), robot="RBY1")
    kwargs["seed"] = seed
    cameras = kwargs.get("camera_names", ["freeview"])
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": [], "rgb": [c+"_image" for c in cameras]}})
    return EnvRobosuite("TurnOnSinkFaucetRBY1", render=False,
                        render_offscreen=True, use_image_obs=True,
                        postprocess_visual_obs=False, **kwargs)


class Recorder:
    def __init__(self, env, interface, settings, video, diagnostic_file):
        self.env, self.interface, self.settings = env, interface, settings
        self.video, self.log = video, diagnostic_file
        self.rows = []
        self.phase = "initialize"
        self.bad_tracking = 0

    def step(self, pose, gripper=1., phase=None):
        if phase is not None:
            self.phase = phase
        interface = self.interface
        action = interface.full_action(pose, gripper)
        interface.preflight(action)
        actual = interface.get_robot_eef_pose()
        faucet = interface.env.sink.get_handle_state(interface.env)
        error = float(np.linalg.norm(actual[:3, 3]-pose[:3, 3]))
        diag = dict(phase=self.phase, actual_tcp=actual, commanded_tcp=pose,
                    tracking_error=error, handle_angle=float(faucet["handle_joint"]),
                    water_on=bool(faucet["water_on"]), contact=interface.geom.contact(),
                    ik_failure=False, collision=False, task_success=bool(interface.env._check_success()))
        row = dict(states=self.env.get_state()["states"].copy(), actions=action.copy(),
                   obs=self.env.get_observation(), diagnostics=diag)
        self.rows.append(row)
        serial = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in diag.items()}
        self.log.write(json.dumps(serial)+"\n")
        self.log.flush()
        if self.video is not None and len(self.rows) % 3 == 1:
            self.video.append_data(self.env.render(mode="rgb_array", height=512, width=512,
                                                   camera_name=self.settings["video_camera"]))
        _, reward, done, _ = self.env.step(action)
        row["rewards"] = float(reward)
        row["dones"] = bool(done)
        interface.geom.check()
        achieved = interface.get_robot_eef_pose()
        tracking = np.linalg.norm(achieved[:3, 3]-pose[:3, 3])
        rotation_error = np.linalg.norm(Rotation.from_matrix(pose[:3,:3] @ achieved[:3,:3].T).as_rotvec())
        self.bad_tracking = self.bad_tracking+1 if (tracking > self.settings["max_tracking_error"] or rotation_error > .5) else 0
        if self.bad_tracking >= self.settings["tracking_patience"]:
            raise UnsafeTrajectory(f"Persistent IK tracking error: {tracking:.3f} m / {rotation_error:.3f} rad")
        return tracking

    def move(self, position, rotation, phase, gripper=1.):
        start = self.interface.get_robot_eef_pose()
        distance = np.linalg.norm(position-start[:3, 3])
        angle = np.linalg.norm(Rotation.from_matrix(rotation@start[:3, :3].T).as_rotvec())
        n = max(2, math.ceil(distance/self.settings["cartesian_increment"]), math.ceil(angle/.02))
        for i, p in enumerate(lerp_points(start[:3, 3], position, n)):
            r = interpolate_rotation(start[:3, :3], rotation, i/(n-1))
            self.step(make_T(p, r), gripper, phase)
        for _ in range(self.settings["settle_steps"]):
            error = self.step(make_T(position, rotation), gripper, phase)
            if error < self.settings["waypoint_tolerance"]:
                return
        raise UnsafeTrajectory(f"Unreachable {phase} waypoint; residual {error:.3f} m")


def scripted_rollout(recorder):
    interface = recorder.interface
    geom, env = interface.geom, interface.env
    if env._check_success() or env.sink.get_handle_state(env)["water_on"]:
        raise UnsafeTrajectory("Faucet did not begin off")
    start = interface.get_robot_eef_pose()
    point = geom.landmark()
    axis, anchor = geom.d.xaxis[geom.joint].copy(), geom.d.xanchor[geom.joint].copy()
    # Follow the real articulation's positive arc. Grasp across the hinge axis,
    # with wrist displaced horizontally outward from the handle / basin.
    radial = point-anchor
    radial -= axis*np.dot(axis, radial)
    if np.linalg.norm(radial) < .015:
        raise UnsafeTrajectory("Handle lever arm too short")
    approach = recorder.settings.get("approach_world")
    if approach is None:
        outward = start[:3, 3]-point
        outward[2] = 0
        outward /= np.linalg.norm(outward)
    else:
        outward = np.asarray(approach, dtype=float)
        outward /= np.linalg.norm(outward)
    # TCP local +z points back toward wrist for the repository adapter.
    z = outward
    x = axis-z*np.dot(axis, z)
    if np.linalg.norm(x) < .1:
        x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rotation = np.column_stack([x, y, z])
    pre = point+.12*outward
    elevated = pre.copy()
    elevated[2] = max(start[2, 3], point[2]+.18)
    recorder.move(elevated, rotation, "safe_preapproach", -1.)
    recorder.move(pre, rotation, "align", -1.)
    recorder.move(point, rotation, "approach", -1.)
    for _ in range(15):
        recorder.step(make_T(point, rotation), 1., "grasp")
    if not geom.contact():
        raise UnsafeTrajectory("No physical finger-to-handle contact")
    initial_angle = float(env.sim.data.get_joint_qpos(env.sink.naming_prefix+"handle_joint"))
    for delta in np.linspace(0., recorder.settings["target_angle"]-initial_angle, 90):
        position = anchor+Rotation.from_rotvec(axis*delta).apply(point-anchor)
        r = Rotation.from_rotvec(axis*delta).as_matrix() @ rotation
        recorder.step(make_T(position, r), 1., "activate")
    if not env._check_success():
        raise UnsafeTrajectory("Handle arc did not activate faucet")
    current = interface.get_robot_eef_pose()
    for _ in range(12):
        recorder.step(current, -1., "release")
    recorder.move(current[:3, 3]+.12*outward, current[:3, :3], "retreat", -1.)
    if not env._check_success():
        raise UnsafeTrajectory("Faucet switched off during retreat")


def collect(config, output, count, max_attempts):
    if count < 1 or max_attempts < count:
        raise ValueError("Need 1 <= count <= max_attempts")
    output.mkdir(parents=True, exist_ok=False)
    (output/"config.json").write_text(json.dumps(config, indent=2))
    summary = dict(attempted=0, successful=0, failures=[])
    with h5py.File(output/"seeds.hdf5", "x") as dataset:
        data = dataset.create_group("data")
        for attempt in range(max_attempts):
            if summary["successful"] >= count:
                break
            summary["attempted"] += 1
            env, video = None, None
            try:
                env = make_env(config, config["seed"]+attempt)
                env.reset()
                initial = env.get_state()
                interface = MG_RBY1TurnOnSinkFaucet(env.base_env)
                video = imageio.get_writer(str(output/f"attempt_{attempt:03d}.mp4"), fps=20/3)
                video.append_data(env.render(mode="rgb_array", height=512, width=512, camera_name=config["script"]["video_camera"]))
                with (output/f"attempt_{attempt:03d}.jsonl").open("x") as log:
                    recorder = Recorder(env, interface, config["script"], video, log)
                    scripted_rollout(recorder)
                name = f"demo_{summary['successful']}"
                write_episode(data.create_group(name), env, initial, recorder.rows, "scripted")
                summary["successful"] += 1
                dataset.flush()
            except UnsafeTrajectory as error:
                summary["failures"].append(dict(attempt=attempt, reason=str(error)))
                print(f"Attempt {attempt}: rejected: {error}", flush=True)
            except Exception as error:
                summary["failures"].append(dict(attempt=attempt, reason=str(error), error_type=type(error).__name__))
                raise
            finally:
                if video is not None:
                    video.close()
                if env is not None:
                    env.env.close()
                (output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary["successful"] == count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("faucet_config.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(0 if collect(json.loads(args.config.read_text()), args.output, args.count, args.max_attempts) else 1)


if __name__ == "__main__":
    main()
