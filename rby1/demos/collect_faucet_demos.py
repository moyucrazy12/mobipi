"""Scripted right-arm faucet seeds, using the existing collector's IK/action helpers."""
import argparse
import json
import math
import traceback
from pathlib import Path

from rby1.paths import new_output

import h5py
import imageio.v2 as imageio
import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.controllers import load_composite_controller_config
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.obs_utils as ObsUtils

from rby1.robot.faucet_runtime import (
    CONTROLLER, MG_RBY1TurnOnSinkFaucet, UnsafeTrajectory, make_T,
    raw_data,
)
from rby1.demos.auto_close_microwave_door import interpolate_rotation, lerp_points
from rby1.demos.faucet_dataset import write_episode


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
        info = interface.get_datagen_info(action=action) if hasattr(self, "datagen_infos") else None
        actual = interface.get_robot_eef_pose()
        faucet = interface.env.sink.get_handle_state(interface.env)
        error = float(np.linalg.norm(actual[:3, 3]-pose[:3, 3]))
        diag = dict(phase=self.phase, actual_tcp=actual, commanded_tcp=pose,
                    tracking_error=error, handle_angle=float(faucet["handle_joint"]),
                    water_on=bool(faucet["water_on"]), contact=interface.geom.contact(),
                    ik_failure=False, collision=False, task_success=bool(interface.env._check_success()))
        # Model XML / episode metadata are captured once in the initial state.
        # Per-step recording needs only the identical flattened simulator state.
        row = dict(states=np.asarray(interface.env.sim.get_state().flatten()).copy(), actions=action.copy(),
                   obs=self.env.get_observation(), diagnostics=diag)
        self.rows.append(row)
        if info is not None:
            self.datagen_infos.append(info)
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
        self.check_tracking(tracking, rotation_error)
        return tracking

    def check_tracking(self, tracking, rotation_error):
        # Keep the shared MimicGen recorder's existing behavior unchanged.
        self.bad_tracking = self.bad_tracking+1 if (tracking > self.settings["max_tracking_error"] or rotation_error > .5) else 0
        if self.bad_tracking >= self.settings["tracking_patience"]:
            raise UnsafeTrajectory(f"Persistent IK tracking error: {tracking:.3f} m / {rotation_error:.3f} rad")

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


class ScriptedRecorder(Recorder):
    """Hold scripted waypoints until convergence, progress failure or finite timeout."""
    def __init__(self, env, interface, settings, video, diagnostic_file, tracking_file):
        super().__init__(env, interface, settings, video, diagnostic_file)
        from rby1.demos.faucet_progress import TrackingDiagnostics
        self.tracking = TrackingDiagnostics(self, tracking_file)
        self.waypoint_index = -1
        self.gate = None
        self.initial_sim_time = float(raw_data(interface.env.sim).time)
        self.last_condition = "running"

    def check_tracking(self, tracking, rotation_error):
        # Scripted progression is governed by a fixed-target progress gate instead
        # of the shared recorder's moving-target consecutive-error counter.
        pass

    def step(self, pose, gripper=1., phase=None):
        if phase is not None:
            self.phase = phase
        try:
            if float(raw_data(self.interface.env.sim).time)-self.initial_sim_time >= self.settings["max_rollout_seconds"]:
                self.last_condition = "rollout_timeout"
                raise UnsafeTrajectory(self.last_condition)
            before = self.tracking.before_step(pose, gripper)
            try:
                value = super().step(pose, gripper, phase)
            except UnsafeTrajectory as error:
                self.last_condition = "ik_failure" if "IK" in str(error) or "finite" in str(error) else "unsafe_configuration"
                self.tracking.record(pose, before, self.last_condition)
                raise
            position, orientation = self.tracking.record(pose, before)
            if self.gate is not None:
                condition = self.gate.update(position, orientation)
                self.last_condition = condition
                # Add the gate result to a separate event; do not change dataset rows.
                self.tracking.write(dict(self.tracking.latest, event="progress",
                    termination_condition=condition, waypoint_steps=self.gate.steps,
                    window_error_reduction=self.gate.progress))
                if condition in ("stalled_convergence", "persistent_tracking_error", "waypoint_timeout", "ik_failure"):
                    raise UnsafeTrajectory(f"{condition}: waypoint {self.waypoint_index}, "
                        f"{position:.4f} m / {orientation:.4f} rad, {self.gate.steps} held steps")
            return value
        except UnsafeTrajectory as error:
            failures = ("rollout_timeout", "stalled_convergence", "persistent_tracking_error",
                        "waypoint_timeout", "ik_failure", "unsafe_configuration")
            if self.last_condition not in failures:
                self.last_condition = "ik_failure" if "IK" in str(error) or "finite" in str(error) else "unsafe_configuration"
            self.tracking.terminate(self.last_condition)
            raise

    def waypoint(self, pose, gripper=1., phase=None):
        from rby1.demos.faucet_progress import ProgressGate
        self.waypoint_index += 1
        self.gate = ProgressGate(self.settings["waypoint"], self.interface.env.control_timestep)
        try:
            while True:
                self.step(pose, gripper, phase)
                if self.last_condition == "converged":
                    return
        finally:
            self.gate = None

    def move(self, position, rotation, phase, gripper=1.):
        start = self.interface.get_robot_eef_pose()
        distance = np.linalg.norm(position-start[:3, 3])
        angle = np.linalg.norm(Rotation.from_matrix(rotation@start[:3, :3].T).as_rotvec())
        # Smoothstep's peak derivative is 1.5; bound each translation increment.
        n = max(2, math.ceil(1.5*distance/self.settings["cartesian_increment"])+1,
                math.ceil(angle/self.settings["rotation_increment"])+1)
        self.tracking.write(dict(event="motion_plan", phase=phase,
            desired_phase_goal=make_T(position, rotation), interpolation_points=n,
            initial_tcp=start, control_timestep=self.interface.env.control_timestep))
        for i, point in enumerate(lerp_points(start[:3, 3], position, n)):
            rotation_i = interpolate_rotation(start[:3, :3], rotation, i/(n-1))
            self.waypoint(make_T(point, rotation_i), gripper, phase)


class JointSpaceRecorder(ScriptedRecorder):
    """Sparse bounded Cartesian IK followed by checked smooth joint interpolation."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from rby1.robot.faucet_bounded_ik import bind_bounded_ik
        self.bounded = getattr(self.interface, "bounded", None)
        if self.bounded is None:
            self.bounded = bind_bounded_ik(self.interface)
        # Preserve the intentionally selected reset torso/head posture.
        for part in ("torso", "head"):
            ctrl = self.interface.env.robots[0].part_controllers.get(part)
            if ctrl is not None:
                self.interface.holds[part] = self.interface.env.sim.data.qpos[ctrl.qpos_index].copy()

    def step(self, pose, gripper=1., phase=None):
        value = super().step(pose, gripper, phase)
        self.bounded.check_scene(raw_data(self.interface.env.sim))
        return value

    def transition(self, q, gripper, phase):
        current = raw_data(self.interface.env.sim).qpos[self.bounded.addresses].copy()
        delta = float(np.max(np.abs(q-current)))
        if delta > self.settings.get("max_joint_anchor_jump", 1.2):
            raise UnsafeTrajectory(f"Discontinuous IK branch: {delta:.3f} rad")
        dt = self.interface.env.control_timestep
        duration = max(1.5*delta/self.settings["joint_velocity"],
                       math.sqrt(6*delta/self.settings.get("joint_acceleration", 1.)))
        count = max(2, math.ceil(duration/dt)+1)
        # Validate the complete interpolated joint path before moving.
        path = lerp_points(current, q, count)
        for point in path:
            self.bounded.check(point)
        bad = 0
        for point in path:
            pose = self.bounded.command_joints(point)
            self.step(pose, gripper, phase)
            actual = raw_data(self.interface.env.sim).qpos[self.bounded.addresses]
            error = float(np.max(np.abs(actual-point)))
            bad = bad+1 if error > .12 else 0
            if bad >= 20:
                raise UnsafeTrajectory(f"Joint trajectory tracking stalled: {error:.3f} rad")

    def waypoint(self, pose, gripper=1., phase=None):
        self.waypoint_index += 1
        q = self.bounded.optimize(pose)
        info = dict(self.bounded.last)
        self.tracking.write(dict(event="bounded_ik", phase=phase, waypoint_index=self.waypoint_index,
                                 target_tcp=pose, **info))
        tolerance = self.settings["contact_position_tolerance"] if phase in ("approach", "grasp", "activate") else self.settings["approach_position_tolerance"]
        if info["position_error"] > tolerance or info["orientation_error"] > .08:
            raise UnsafeTrajectory(f"Bounded IK infeasible for {phase}: "
                f"{info['position_error']:.4f} m / {info['orientation_error']:.4f} rad")
        self.transition(q, gripper, phase)
        # Settling only at meaningful anchors, not every small control increment.
        for _ in range(40):
            command = self.bounded.command_joints(q)
            self.step(command, gripper, phase)
            achieved = self.interface.get_robot_eef_pose()
            pe = np.linalg.norm(achieved[:3, 3]-pose[:3, 3])
            re = np.linalg.norm(Rotation.from_matrix(pose[:3, :3]@achieved[:3, :3].T).as_rotvec())
            if pe <= tolerance and re <= .08:
                return
        raise UnsafeTrajectory(f"Dynamic tracking failed for {phase}: {pe:.4f} m / {re:.4f} rad")

    def move(self, position, rotation, phase, gripper=1.):
        start = self.interface.get_robot_eef_pose()
        distance = np.linalg.norm(position-start[:3, 3])
        angle = np.linalg.norm(Rotation.from_matrix(rotation@start[:3, :3].T).as_rotvec())
        count = max(1, math.ceil(distance/self.settings["cartesian_anchor_spacing"]), math.ceil(angle/.35))
        self.tracking.write(dict(event="motion_plan", phase=phase, desired_phase_goal=make_T(position, rotation),
                                 interpolation_points=count, initial_tcp=start))
        for i in range(1, count+1):
            ratio = i/count
            target = start[:3, 3]+ratio*(position-start[:3, 3])
            r = interpolate_rotation(start[:3, :3], rotation, ratio)
            self.waypoint(make_T(target, r), gripper, phase)


def scripted_rollout(recorder):
    interface = recorder.interface
    geom, env = interface.geom, interface.env
    if env._check_success() or env.sink.get_handle_state(env)["water_on"]:
        raise UnsafeTrajectory("Faucet did not begin off")
    start = interface.get_robot_eef_pose()
    point = geom.landmark()+np.array([0., 0., recorder.settings.get("handle_height_offset", 0.)])
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
    # Embedded fingers open along wrist x, which is TCP y. Keep that
    # closing axis horizontal across the vertical lever, not along its length.
    y = np.cross([0., 0., 1.], z)
    if np.linalg.norm(y) < .1:
        raise UnsafeTrajectory("Faucet approach must provide horizontal finger clearance")
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    rotation = np.column_stack([x, y, z])
    pre = point+.12*outward
    point = point-recorder.settings.get("grasp_depth", 0.)*outward
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
    if recorder.settings.get("feedback_articulation", False):
        # Keep a small positional lead on the measured articulation; never run
        # through a pre-timed arc while the lever remains behind or slips away.
        grasp_point = interface.get_robot_eef_pose()[:3, 3].copy()
        command_angle = initial_angle
        history = []
        for iteration in range(40):
            angle = float(env.sim.data.get_joint_qpos(env.sink.naming_prefix+"handle_joint"))
            history.append(angle)
            if angle >= recorder.settings["target_angle"] and env._check_success():
                break
            if len(history) > 10 and angle-max(history[:-10]) < .002:
                raise UnsafeTrajectory("Physical handle articulation stalled")
            command_angle = min(max(command_angle, angle+recorder.settings["handle_angle_lead"]),
                                recorder.settings["max_commanded_arc_angle"])
            delta = command_angle-initial_angle
            position = anchor+Rotation.from_rotvec(axis*delta).apply(grasp_point-anchor)
            recorder.waypoint(make_T(position, rotation), 1., "activate")
            for _ in range(recorder.settings.get("arc_hold_steps", 8)):
                recorder.step(make_T(position, rotation), 1., "activate")
        else:
            raise UnsafeTrajectory("Handle articulation iteration timeout")
    else:
        for delta in np.linspace(0., recorder.settings["target_angle"]-initial_angle, recorder.settings.get("arc_waypoints", 90)):
            position = anchor+Rotation.from_rotvec(axis*delta).apply(point-anchor)
            # Translate along the handle hinge arc with the wrist held outside the
            # basin. Rotating the entire wrist about this side lever dives the wrist.
            r = rotation if recorder.settings.get("freeze_articulation_orientation", False) else Rotation.from_rotvec(axis*delta).as_matrix() @ rotation
            recorder.waypoint(make_T(position, r), 1., "activate")
            for _ in range(recorder.settings.get("arc_hold_steps", 0)):
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
            reset_complete = False
            stage = "initialize"
            try:
                env = make_env(config, config["seed"]+attempt)
                stage = "reset"
                env.reset()
                reset_complete = True
                stage = "rollout"
                initial = env.get_state()
                interface = MG_RBY1TurnOnSinkFaucet(env.base_env)
                video = imageio.get_writer(str(output/f"attempt_{attempt:03d}.mp4"), fps=env.base_env.control_freq/3)
                video.append_data(env.render(mode="rgb_array", height=512, width=512, camera_name=config["script"]["video_camera"]))
                with (output/f"attempt_{attempt:03d}.jsonl").open("x") as log, \
                     (output/f"attempt_{attempt:03d}_tracking.jsonl").open("x") as tracking:
                    recorder_class = JointSpaceRecorder if config["script"].get("controller_mode") == "bounded_joint" else ScriptedRecorder
                    recorder = recorder_class(env, interface, config["script"], video, log, tracking)
                    policy = interface._policy
                    recorder.tracking.write(dict(event="runtime", seed=config["seed"]+attempt,
                        model_timestep=interface.env.model_timestep,
                        physics_timestep=interface.geom.m.opt.timestep,
                        control_timestep=interface.env.control_timestep,
                        horizon=interface.env.horizon, ignore_done=interface.env.ignore_done,
                        tcp_to_wrist=interface.tcp_ref, joint_names=policy.joint_names,
                        qpos_addresses=policy.qpos_addresses, velocity_addresses=policy.velocity_addresses,
                        base_pose=interface.get_controller_base_pose(),
                        ik_damping=policy.damping, ik_integration_dt=policy.integration_dt,
                        ik_max_dq=policy.max_dq))
                    scripted_rollout(recorder)
                    recorder.tracking.terminate("success")
                name = f"demo_{summary['successful']}"
                write_episode(data.create_group(name), env, initial, recorder.rows, "scripted")
                summary["successful"] += 1
                dataset.flush()
            except UnsafeTrajectory as error:
                summary["failures"].append(dict(attempt=attempt, reason=str(error)))
                print(f"Attempt {attempt}: rejected: {error}", flush=True)
            except Exception as error:
                summary["failures"].append(dict(attempt=attempt, seed=config["seed"]+attempt,
                    stage=stage, reason=str(error), error_type=type(error).__name__,
                    traceback=traceback.format_exc()))
                raise
            finally:
                def cleanup_error(operation, error):
                    summary.setdefault("cleanup_errors", []).append(dict(
                        attempt=attempt, operation=operation, reason=str(error)))
                    print(f"Attempt {attempt}: {operation}: {error}", flush=True)

                if video is not None:
                    try:
                        video.close()
                    except Exception as error:
                        cleanup_error("close video", error)
                if env is not None:
                    base = env.base_env
                    if reset_complete and getattr(base, "sim", None) is not None:
                        try:
                            summary["last_task_state"] = base.sink.get_handle_state(base)
                            summary["last_episode_success"] = bool(base._check_success())
                        except Exception as error:
                            cleanup_error("read final task state", error)
                    try:
                        env.env.close()
                    except Exception as error:
                        cleanup_error("close environment", error)
                (output/"summary.json").write_text(json.dumps(summary, indent=2))
                tracking_path = output/f"attempt_{attempt:03d}_tracking.jsonl"
                if tracking_path.exists() and tracking_path.stat().st_size:
                    from rby1.demos.faucet_progress import plot_tracking
                    try:
                        plot_tracking(tracking_path, output/f"attempt_{attempt:03d}_tracking.png")
                    except Exception as error:
                        print(f"No tracking plot: {error}", flush=True)
    print(json.dumps(summary, indent=2))
    return summary["successful"] == count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="External faucet task JSON from the team's private configuration folder")
    parser.add_argument("--output", type=Path, default=new_output("demos", "faucet_seeds"))
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=1)
    args = parser.parse_args()
    raise SystemExit(0 if collect(json.loads(args.config.expanduser().read_text()), args.output, args.count, args.max_attempts) else 1)


if __name__ == "__main__":
    main()
