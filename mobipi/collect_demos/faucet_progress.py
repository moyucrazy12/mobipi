"""Progress gates and sidecar diagnostics for scripted faucet rollouts only."""
import argparse
from collections import deque
import json
from pathlib import Path

from mobipi.collect_demos.experiment_paths import new_output

import numpy as np
from scipy.spatial.transform import Rotation


class ProgressGate:
    """Monitor one fixed target; errors of different moving targets are never compared."""
    def __init__(self, settings, control_dt):
        self.settings = settings
        self.dt = control_dt
        self.history = deque(maxlen=settings["stall_window_steps"] + 1)
        self.steps = 0
        self.best = float("inf")
        self.progress = 0.
        for key in ("position_tolerance", "orientation_tolerance", "max_seconds",
                    "min_position_progress", "min_orientation_progress"):
            if settings[key] <= 0:
                raise ValueError(f"waypoint.{key} must be positive")
        if settings["max_steps"] < 1 or settings["stall_window_steps"] < 2:
            raise ValueError("Waypoint step limits must be positive; stall window >= 2")

    def update(self, position_error, orientation_error):
        self.steps += 1
        s = self.settings
        if not np.isfinite([position_error, orientation_error]).all():
            return "ik_failure"
        if position_error <= s["position_tolerance"] and orientation_error <= s["orientation_tolerance"]:
            return "converged"
        # A normalized sum accounts for both translation and rotation progress.
        score = position_error/s["min_position_progress"] + orientation_error/s["min_orientation_progress"]
        self.history.append(score)
        self.best = min(self.best, score)
        self.progress = self.history[0] - min(self.history)
        if self.steps >= s["max_steps"] or self.steps*self.dt >= s["max_seconds"]-1e-9:
            return "waypoint_timeout"
        if len(self.history) == self.history.maxlen and self.progress < 1.:
            return "persistent_tracking_error" if position_error > s["transient_position_error"] else "stalled_convergence"
        return "converging" if self.progress >= 1. else "transient_tracking_error"


def pose_errors(actual, target):
    return (float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
            float(np.linalg.norm(Rotation.from_matrix(target[:3, :3]@actual[:3, :3].T).as_rotvec())))


class TrackingDiagnostics:
    """Readable post-control samples in a sidecar; the HDF5 schema stays unchanged."""
    def __init__(self, recorder, file):
        self.recorder, self.file = recorder, file
        self.previous = None
        self.latest = None

    def before_step(self, pose, gripper):
        import mujoco
        from mobipi.collect_demos.faucet_runtime import raw_model, raw_data
        interface = self.recorder.interface
        env, policy = interface.env, interface._policy
        m, d = raw_model(env.sim), raw_data(env.sim)
        action = interface.full_action(pose, gripper)
        q = policy.solve(action[:policy.control_dim].copy())
        if not np.isfinite(q).all():
            from mobipi.collect_demos.faucet_runtime import UnsafeTrajectory
            raise UnsafeTrajectory("IK returned non-finite joints during diagnostics")
        right = [policy.joint_names.index(f"robot0_right_arm_{j}") for j in range(7)]
        qadr = policy.qpos_addresses[right]
        jids = policy.joint_ids[right]
        trial = mujoco.MjData(m)
        trial.qpos[:] = d.qpos
        trial.qpos[policy.qpos_addresses] = q
        mujoco.mj_forward(m, trial)
        reference = m.site("gripper0_right_ft_frame").id
        desired_ref = pose @ interface.tcp_ref
        predicted_ref = np.eye(4)
        predicted_ref[:3, 3] = trial.site_xpos[reference]
        predicted_ref[:3, :3] = trial.site_xmat[reference].reshape(3, 3)
        ctrl = env.robots[0].part_controllers["right"]
        return dict(qadr=qadr, jids=jids, q_target=q[right].copy(),
                    action=action.copy(), q_before=d.qpos[qadr].copy(),
                    ik_predicted_errors=pose_errors(predicted_ref, desired_ref),
                    controller_input_type=ctrl.input_type,
                    model_timestep=float(env.model_timestep), physics_timestep=float(m.opt.timestep),
                    control_timestep=float(env.control_timestep))

    def record(self, pose, before, condition="running", event="step"):
        from mobipi.collect_demos.faucet_runtime import raw_model, raw_data
        rec, interface = self.recorder, self.recorder.interface
        env = interface.env
        m, d = raw_model(env.sim), raw_data(env.sim)
        actual = interface.get_robot_eef_pose()
        position, orientation = pose_errors(actual, pose)
        previous = self.previous
        same_target = previous is not None and previous[0] == rec.waypoint_index and np.allclose(previous[1], pose, atol=1e-12)
        reduction = previous[2]-position if same_target else 0.
        rotation_reduction = previous[3]-orientation if same_target else 0.
        self.previous = (rec.waypoint_index, pose.copy(), position, orientation)
        q = d.qpos[before["qadr"]]
        ctrl = env.robots[0].part_controllers["right"]
        actuator_ids = np.array([np.flatnonzero((m.actuator_trnid[:, 0] == j) & (m.actuator_trntype == 0))[0] for j in before["jids"]])
        control = d.ctrl[actuator_ids]
        ranges = m.actuator_ctrlrange[actuator_ids]
        limits = m.jnt_range[before["jids"]]
        result = dict(event=event, control_step=len(rec.rows), simulation_time=float(d.time),
                      physics_step=int(round(d.time/m.opt.timestep)), phase=rec.phase,
                      waypoint_index=rec.waypoint_index, desired_tcp=pose, actual_tcp=actual,
                      position_error=position, orientation_error=orientation,
                      position_error_reduction=reduction, orientation_error_reduction=rotation_reduction,
                      right_joint_positions=q.copy(), ik_joint_targets=before["q_target"],
                      joint_tracking_errors=before["q_target"]-q,
                      joint_error_before=before["q_target"]-before["q_before"],
                      controller_joint_goals=np.asarray(ctrl.goal_qpos).copy(),
                      joint_limit_margin=np.minimum(q-limits[:, 0], limits[:, 1]-q),
                      actuator_controls=control.copy(), actuator_ctrlrange=ranges.copy(),
                      actuator_saturated=(np.isclose(control,ranges[:, 0], atol=1e-3)|np.isclose(control,ranges[:, 1], atol=1e-3)),
                      ik_predicted_position_error=before["ik_predicted_errors"][0],
                      ik_predicted_orientation_error=before["ik_predicted_errors"][1],
                      controller_input_type=before["controller_input_type"],
                      model_timestep=before["model_timestep"], physics_timestep=before["physics_timestep"],
                      control_timestep=before["control_timestep"], termination_condition=condition)
        result["right_gripper_qpos"] = [float(env.sim.data.get_joint_qpos(f"robot0_gripper_finger_r{j}")) for j in (1, 2)]
        result["handle_state"] = env.sink.get_handle_state(env)
        self.latest = result
        self.write(result)
        return position, orientation

    def write(self, result):
        self.file.write(json.dumps(result, default=lambda a: a.tolist() if isinstance(a,np.ndarray) else a)+"\n")
        self.file.flush()

    def terminate(self, condition):
        if self.latest is not None:
            result = dict(self.latest, event="termination", termination_condition=condition)
            self.write(result)
        else:
            self.write(dict(event="termination", termination_condition=condition))


def plot_tracking(path, output):
    """Four diagnostic panels: tracking, coordinates, held targets, phase transitions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    samples = [r for r in rows if r.get("event") == "step"]
    if not samples:
        raise ValueError("No control samples to plot")
    t = np.array([r["simulation_time"] for r in samples])
    p = np.array([r["position_error"] for r in samples])
    r = np.array([r["orientation_error"] for r in samples])
    actual = np.array([r["actual_tcp"] for r in samples])[:, :3, 3]
    desired = np.array([r["desired_tcp"] for r in samples])[:, :3, 3]
    fig, axes = plt.subplots(4, 1, figsize=(11, 13), sharex=True, layout="constrained")
    axes[0].plot(t, p*1000, label="TCP position error (mm)")
    axes[0].set_ylabel("Position error (mm)")
    twin = axes[0].twinx()
    twin.plot(t, np.rad2deg(r), color="tab:orange", label="Rotation error")
    twin.set_ylabel("Rotation error (degrees)")
    axes[0].legend(loc="upper left")
    for j, coordinate in enumerate("xyz"):
        axes[1].plot(t, actual[:, j], label=f"actual {coordinate}")
        axes[1].plot(t, desired[:, j], "--", label=f"target {coordinate}")
    axes[1].set_ylabel("World TCP position (m)")
    axes[1].legend(ncol=3)
    axes[2].plot(t, p*1000, label="Distance to held waypoint")
    waypoint = np.array([row["waypoint_index"] for row in samples])
    for index in np.flatnonzero(np.diff(waypoint)):
        axes[2].axvline(t[index+1], color="gray", alpha=.25)
    axes[2].set_ylabel("Waypoint distance (mm)")
    axes[2].legend()
    phases = list(dict.fromkeys(row["phase"] for row in samples))
    axes[3].step(t, [phases.index(row["phase"]) for row in samples], where="post", label="Phase")
    axes[3].set_yticks(range(len(phases)), phases)
    axes[3].set_xlabel("Simulation time (s)")
    for ax in axes:
        ax.grid(alpha=.25)
    final = rows[-1].get("termination_condition", "unknown")
    fig.suptitle(f"Scripted faucet control: {final}")
    fig.savefig(output, dpi=160)
    plt.close(fig)
    output = Path(output)
    fig, coordinates = plt.subplots(3, 1, figsize=(10, 8), sharex=True, layout="constrained")
    for j, axis in enumerate(coordinates):
        axis.plot(t, actual[:, j], label="Actual TCP")
        axis.plot(t, desired[:, j], "--", label="Commanded TCP")
        axis.set_ylabel(f"World {'xyz'[j]} (m)")
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        axis.grid(alpha=.25)
        axis.legend()
    coordinates[-1].set_xlabel("Simulation time (s)")
    fig.savefig(output.with_name(output.stem+"_positions.png"), dpi=160)
    plt.close(fig)
    fig, joints = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout="constrained")
    q = np.array([row["right_joint_positions"] for row in samples])
    goal = np.array([row["ik_joint_targets"] for row in samples])
    torque = np.array([row["actuator_controls"] for row in samples])
    limits = np.array(samples[0]["actuator_ctrlrange"])
    joints[0].plot(t, q[:, 3], label="Actual elbow")
    joints[0].plot(t, goal[:, 3], "--", label="IK elbow target")
    joints[0].axhline(.01, color="red", linestyle=":", label="RB-Y1 elbow upper limit")
    joints[0].set_ylabel("Elbow angle (rad)")
    for j in range(q.shape[1]):
        joints[1].plot(t, goal[:, j]-q[:, j], label=f"Joint {j}")
        joints[2].plot(t, torque[:, j]/np.max(np.abs(limits[j])), label=f"Joint {j}")
    joints[1].set_ylabel("Joint command error (rad)")
    joints[2].set_ylabel("Actuator control / limit")
    joints[2].set_xlabel("Simulation time (s)")
    for axis in joints:
        axis.grid(alpha=.25)
        axis.legend(ncol=4)
    fig.savefig(output.with_name(output.stem+"_joints.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=new_output("diagnostics", "faucet_tracking") / "tracking.png")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_tracking(args.source, args.output)
