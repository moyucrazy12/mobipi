"""
RB-Y1 SDK backend for the Mobi-pi pipeline (docs/rby1_sdk_integration.md).

The robot is driven only through rby1_sdk calls, so the same code runs
against rby1-sim and the real robot. The SDK has no cameras and no task
state, so a RoboCasa env acts as a follower: every control tick it copies the
SDK base pose (teleport) and joint positions (joint-position servos, so the
arm still pushes the door), steps its physics, and provides the camera
images, task progress and environment contacts. The coupling is one-way:
contacts in the follower do not push back on the SDK robot.

Navigation reuses rby1_utils' closed-loop driver with SDK odometry and SE(2)
velocity commands (SDKBaseDriver). The policy's TCP targets go through the
training-time WholeBody IK and servo response to a reference arm that the
SDK tracks with streamed joint targets (SDKPolicyEnv).
"""
import time

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T

from mobipi.utils import rby1_utils
from mobipi.utils.nav_utils import angle_wrap


# Frames, fitted by comparing SDK FK (base -> ee_left) with the robosuite
# model (mobilebase0_center -> left grip_site) over random joint vectors
# (residual < 0.001 mm, 0.0001 deg): the robosuite base origin sits 3 mm above
# the SDK base link (MOBILE_BASE_CLEARANCE), and the policy TCP is 0.124 m
# along ee_left's z axis, rotated -90 deg about it. Odometry tracks the SDK
# base link, the same point as get_base_vec.
T_RSBASE_SDKBASE = np.eye(4)
T_RSBASE_SDKBASE[2, 3] = 0.003
T_EE_LEFT_TCP = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, -0.124],
    [0.0, 0.0, 0.0, 1.0],
])

CONTROL_FREQ = 20
# Command hold times: the robot stops if no new command arrives in time. The
# arm one covers policy inference at chunk boundaries.
BASE_HOLD_TIME = 0.3
ARM_HOLD_TIME = 0.5
# Cartesian tracking: velocity limits [m/s, rad/s], acceleration scaling and
# minimum time to reach each target [s]
CARTESIAN_LIN_VEL = 1.0
CARTESIAN_ANG_VEL = 3.0
CARTESIAN_ACC_SCALING = 1.0
CARTESIAN_MIN_TIME = 0.1
# Fraction of the WholeBody IK step that robosuite's joint servos complete in
# one control step: q_next = q + 0.22 (q_ik - q), fitted on close_microwave
# rollouts (R^2 = 1.00, same for all left arm joints). The "joint" arm control
# integrates this response (SDKPolicyEnv), so the SDK arm moves at the speed the
# policy saw in training; streaming q_ik itself moves the arm ~4.5x faster.
TRAINING_SERVO_RESPONSE = 0.22
# Measured-to-reference arm error [rad] at which SDKPolicyEnv restarts its
# reference arm from the measured joints
ARM_REF_RESYNC = 0.1
# Streamed joint targets: acceleration limit as a multiple of the URDF limit,
# and distance kept from the SDK joint limits [rad]
JOINT_STREAM_ACC_SCALE = 30.0
JOINT_LIMIT_MARGIN = 1e-3
# Joint moves (posture, arm return): cubic-like profile at this peak speed
JOINT_MOVE_SPEED = 0.5
JOINT_MOVE_MIN_TIME = 2.0

# Differential drive geometry (URDF: wheel base 0.53 m, wheel radius 0.1 m)
WHEEL_TRACK = 0.53
WHEEL_RADIUS = 0.1
# Wheel speed deadband [rad/s] that SDKBaseDriver adds to every nonzero wheel
# speed. In Rainbow's unpatched rby1-sim model the base drags on the floor, so
# the wheels barely turn below ~0.24 rad/s and the closed-loop driver crawls
# near its goals and reports stalls ("sim"). The patched model
# (assets/rby1_sim/base_geoms_frictionless.xml) and the robot need "none".
BASE_DEADBANDS = {
    "none": 0.0,
    "sim": 0.24,
}

# Policy arm control through the SDK (see SDKPolicyEnv)
ARM_CONTROLS = ("joint", "cartesian")

# SDK body parts and the robosuite joints they mirror
SDK_PARTS = ("torso", "right_arm", "left_arm", "head")
RS_PREFIX = "robot0_"


def pose_to_transform(pose):
    """Planar pose [x, y, yaw] as a 4x4 transform at z = 0."""
    x, y, yaw = pose
    result = np.eye(4)
    result[:2, :2] = [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]
    result[:2, 3] = x, y
    return result


class _Rate:
    """Fixed-rate loop timing that does not try to catch up after an overrun."""

    def __init__(self, freq):
        self.period = 1.0 / freq
        self.deadline = None
        self.overruns = 0
        self.max_lateness = 0.0

    def sleep(self):
        now = time.perf_counter()
        self.deadline = now + self.period if self.deadline is None else self.deadline + self.period
        delay = self.deadline - now
        if delay > 0:
            time.sleep(delay)
        else:
            self.overruns += 1
            self.max_lateness = max(self.max_lateness, -delay)
            self.deadline = now


class SDKRobot:
    """The rby1_sdk robot with the calls the pipeline needs; poses are in the pipeline's frames."""

    def __init__(self, address="localhost:50051", model="a", power=".*", servo=".*"):
        import rby1_sdk as rby

        self.rby = rby
        robot = rby.create_robot(address, model)
        if not robot.connect():
            raise RuntimeError(f"Cannot connect to the RB-Y1 at {address}")
        if not robot.is_power_on(power) and not robot.power_on(power):
            raise RuntimeError(f"Cannot power on {power}")
        if not robot.is_servo_on(servo) and not robot.servo_on(servo):
            raise RuntimeError(f"Cannot turn on servo {servo}")
        faults = (rby.ControlManagerState.State.MajorFault, rby.ControlManagerState.State.MinorFault)
        if robot.get_control_manager_state().state in faults and not robot.reset_fault_control_manager():
            raise RuntimeError("Cannot reset the control manager fault")
        if not robot.enable_control_manager():
            raise RuntimeError("Cannot enable the control manager")
        self.robot = robot
        self.address = address
        robot_model = robot.model()
        self.joint_names = list(robot_model.robot_joint_names)
        self.part_indices = {
            "torso": list(robot_model.torso_idx),
            "right_arm": list(robot_model.right_arm_idx),
            "left_arm": list(robot_model.left_arm_idx),
            "head": list(robot_model.head_idx),
        }
        self.dyn = robot.get_dynamics()
        self.dyn_state = self.dyn.make_state(["base", "ee_left"], self.joint_names)
        self.qdot_limit = np.asarray(self.dyn.get_limit_qdot_upper(self.dyn_state), dtype=float)
        self.qddot_limit = np.asarray(self.dyn.get_limit_qddot_upper(self.dyn_state), dtype=float)
        self.q_lower = np.asarray(self.dyn.get_limit_q_lower(self.dyn_state), dtype=float) + JOINT_LIMIT_MARGIN
        self.q_upper = np.asarray(self.dyn.get_limit_q_upper(self.dyn_state), dtype=float) - JOINT_LIMIT_MARGIN
        self.clipped_commands = 0
        self.clipped_joints = {}
        self.max_clip = 0.0
        self.stream = None
        self.state = robot.get_state()

    def update(self, attempts=3):
        """
        Reads a new state snapshot; the getters below use the latest one.
        rby1-sim occasionally fails to serialize the self-collision list
        ("invalid UTF-8 data" in its log, about 1 in 1000 calls) and the call
        fails, so failed reads are retried.
        """
        for attempt in range(attempts):
            try:
                self.state = self.robot.get_state()
                return self.state
            except RuntimeError as error:
                if attempt == attempts - 1:
                    raise
                print(f"[rby1_sdk_utils.py] State read failed ({error}); retrying", flush=True)

    @property
    def positions(self):
        return np.asarray(self.state.position, dtype=float).copy()

    def base_pose(self):
        """Base link pose [x, y, yaw] from odometry, which reset_base_pose aligns with the world."""
        odom = np.asarray(self.state.odometry)
        return np.array([odom[0, 2], odom[1, 2], np.arctan2(odom[1, 0], odom[0, 0])])

    def reset_base_pose(self, pose, timeout=2.0):
        """Sets odometry so that base_pose() reads `pose` here."""
        pose = np.asarray(pose, dtype=float)
        self.robot.reset_odometry(float(pose[2]), pose[:2].copy())
        deadline = time.perf_counter() + timeout
        while True:
            self.update()
            current = self.base_pose()
            if np.linalg.norm(current[:2] - pose[:2]) < 1e-3 and abs(angle_wrap(current[2] - pose[2])) < 1e-3:
                return
            if time.perf_counter() > deadline:
                raise RuntimeError(f"Odometry did not reset to {pose.round(3)} (reads {current.round(3)})")
            time.sleep(0.02)

    def tcp_in_base(self, positions=None):
        """Policy TCP pose in the robosuite base frame (robot0_base_to_left_eef_*)."""
        self.dyn_state.set_q(self.positions if positions is None else np.asarray(positions, dtype=float))
        self.dyn.compute_forward_kinematics(self.dyn_state)
        return T_RSBASE_SDKBASE @ self.dyn.compute_transformation(self.dyn_state, 0, 1) @ T_EE_LEFT_TCP

    def check_control_manager(self):
        """Raises if the control manager is not enabled (e.g. after a controller fault)."""
        state = self.robot.get_control_manager_state().state
        if state != self.rby.ControlManagerState.State.Enabled:
            raise RuntimeError(f"SDK control manager is {state.name} (see the controller log); "
                               "commands are being rejected")

    def _send(self, component_command):
        rby = self.rby
        if self.stream is None or self.stream.is_done():
            if self.stream is not None:
                # the stream ended without being canceled: a hold time ran out or the control faulted
                self.check_control_manager()
            self.stream = self.robot.create_command_stream(1)
        self.stream.send_command(rby.RobotCommandBuilder().set_command(component_command))

    def command_base(self, v, w):
        """Body velocity [m/s, rad/s]; held for BASE_HOLD_TIME unless renewed."""
        rby = self.rby
        self._send(rby.ComponentBasedCommandBuilder().set_mobility_command(
            rby.SE2VelocityCommandBuilder()
            .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(BASE_HOLD_TIME))
            .set_minimum_time(1.0 / CONTROL_FREQ)
            .set_velocity([float(v), 0.0], float(w))
        ))

    def command_tcp(self, base_tcp):
        """Streams a policy TCP target (robosuite base frame) as an ee_left Cartesian target."""
        rby = self.rby
        target = np.linalg.inv(T_RSBASE_SDKBASE) @ base_tcp @ np.linalg.inv(T_EE_LEFT_TCP)
        self._send(rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder().set_left_arm_command(
                rby.CartesianCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(ARM_HOLD_TIME))
                .add_target("base", "ee_left", target, CARTESIAN_LIN_VEL, CARTESIAN_ANG_VEL, CARTESIAN_ACC_SCALING)
                .set_minimum_time(CARTESIAN_MIN_TIME)
            )
        ))

    def command_left_arm(self, q):
        """
        Streams a left arm joint target, reached within one control period.
        The default acceleration limits make every streamed target start
        slowly (~10 % of the commanded motion per period), so the limits are
        raised as in Rainbow's leader-arm teleop example (URDF velocity
        limits, 30x the URDF acceleration limits).
        """
        rby = self.rby
        left = self.part_indices["left_arm"]
        self._send(rby.ComponentBasedCommandBuilder().set_body_command(
            rby.BodyComponentBasedCommandBuilder().set_left_arm_command(
                rby.JointPositionCommandBuilder()
                .set_command_header(rby.CommandHeaderBuilder().set_control_hold_time(ARM_HOLD_TIME))
                .set_velocity_limit(self.qdot_limit[left])
                .set_acceleration_limit(self.qddot_limit[left] * JOINT_STREAM_ACC_SCALE)
                .set_minimum_time(1.01 / CONTROL_FREQ)
                .set_position(self.clip("left_arm", q))
            )
        ))

    def clip(self, part, q):
        """
        Clips joint targets to the SDK limits, which the controller enforces
        by silently dropping the whole command. The robosuite model's ranges
        differ (e.g. left_arm_1 >= -0.05 there, >= -0.017 here).
        """
        indices = self.part_indices[part]
        q = np.asarray(q, dtype=float)
        clipped = np.clip(q, self.q_lower[indices], self.q_upper[indices])
        excess = np.abs(clipped - q)
        if excess.max() > 0:
            self.clipped_commands += 1
            self.max_clip = max(self.max_clip, float(excess.max()))
            for i in np.flatnonzero(excess):
                name = self.joint_names[indices[i]]
                self.clipped_joints[name] = self.clipped_joints.get(name, 0) + 1
        return clipped

    def move_joints(self, targets, duration=None):
        """
        Starts a joint-space move of the given parts ({"torso": q, "left_arm": q, ...})
        and returns its handler (is_done(), get()). The duration defaults to the
        largest joint distance at JOINT_MOVE_SPEED.
        """
        rby = self.rby
        positions = self.positions
        if duration is None:
            distance = max(np.max(np.abs(np.asarray(q) - positions[self.part_indices[part]]))
                           for part, q in targets.items())
            duration = max(JOINT_MOVE_MIN_TIME, 1.5 * float(distance) / JOINT_MOVE_SPEED)

        def joint_command(part, q):
            return rby.JointPositionCommandBuilder().set_minimum_time(duration).set_position(self.clip(part, q))

        body = rby.BodyComponentBasedCommandBuilder()
        setters = dict(torso=body.set_torso_command, right_arm=body.set_right_arm_command,
                       left_arm=body.set_left_arm_command)
        for part, q in targets.items():
            if part in setters:
                setters[part](joint_command(part, q))
        command = rby.ComponentBasedCommandBuilder()
        if set(targets) & set(setters):
            command.set_body_command(body)
        if "head" in targets:
            command.set_head_command(joint_command("head", targets["head"]))
        return self.robot.send_command(rby.RobotCommandBuilder().set_command(command), 1)

    def stop(self):
        """Stops streaming and every running command; safe to call more than once."""
        try:
            if self.stream is not None and not self.stream.is_done():
                self.command_base(0.0, 0.0)
                self.stream.cancel()
        finally:
            self.stream = None
            self.robot.cancel_control()

    def close(self):
        try:
            self.stop()
        finally:
            self.robot.disconnect()


class Follower:
    """
    Mirrors the SDK robot in a RoboCasa env, one env.step per control tick.

    The base is teleported to the SDK pose (it pushes nothing). The joint
    position parts (torso, head, right arm) get the SDK joints as their
    targets, and the WholeBody IK, which drives the left arm, returns the SDK
    left arm joints instead of solving (the same override as
    return_arm_to_start), so the follower arm tracks the SDK arm with the
    servos the policy was trained with.
    """

    def __init__(self, env, robot, gripper_action=None):
        self.env = env
        self.robot = robot
        self.gripper_action = gripper_action
        sim = env.sim
        controller = env.robots[0].composite_controller
        sdk_index = {RS_PREFIX + name: i for i, name in enumerate(robot.joint_names)}
        self.ik = controller.joint_action_policy
        self.ik_indices = [sdk_index[name] for name in self.ik.joint_names]
        ik_parts = controller.composite_controller_specific_config["actuation_part_names"]
        self.part_slices = []
        for part, (start, end) in controller._whole_body_controller_action_split_indexes.items():
            part_controller = controller.part_controllers.get(part)
            if part in ik_parts or part_controller is None or part_controller.name != "JOINT_POSITION":
                continue
            if part_controller.input_type != "absolute":
                raise ValueError(f"Follower expects absolute joint position control, got {part}")
            self.part_slices.append((slice(start, end), [sdk_index[name] for name in part_controller.joint_names]))
        mirrored = [name for name in sdk_index if name in sim.model.joint_names and "wheel" not in name]
        self.qpos_addrs = [sim.model.get_joint_qpos_addr(name) for name in mirrored]
        self.qvel_addrs = [sim.model.get_joint_qvel_addr(name) for name in mirrored]
        self.mirrored_indices = [sdk_index[name] for name in mirrored]
        self.positions = robot.positions
        self.original_solve = self.ik.solve
        self.ik.solve = lambda unused: self.positions[self.ik_indices].copy()
        self.ik_data = mujoco.MjData(self.ik.full_model)
        self.contacts = []
        self.joint_error = 0.0
        self.max_joint_error = 0.0

    def sync(self, physics=True):
        """
        Reads the SDK state and steps the follower to it. Without physics the
        joints are set directly (~1 ms instead of one ~40 ms env.step); use it
        while the robot cannot touch anything that moves.
        """
        sim = self.env.sim
        self.robot.update()
        self.positions = self.robot.positions
        rby1_utils.set_base_vec(sim, self.robot.base_pose())
        if physics:
            action = rby1_utils.hold_action(self.env, self.gripper_action)
            for action_slice, indices in self.part_slices:
                action[action_slice] = self.positions[indices]
            self.env.step(action)
        else:
            sim.data.qpos[self.qpos_addrs] = self.positions[self.mirrored_indices]
            sim.data.qvel[self.qvel_addrs] = 0.0
            sim.forward()
            # env.step refreshes the --vis viewer; without it the robot would only jump there later
            if self.env.viewer is not None and self.env.renderer != "mujoco":
                self.env.viewer.update()
        self.contacts = rby1_utils.robot_env_contacts(sim)
        self.joint_error = float(np.max(np.abs(self.env.sim.data.qpos[self.qpos_addrs]
                                               - self.positions[self.mirrored_indices])))
        self.max_joint_error = max(self.max_joint_error, self.joint_error)

    def solve_ik(self, ik_input, positions):
        """
        The WholeBody IK step the follower's controller would take for
        `ik_input` with the robot at the SDK joint vector `positions`, solved
        on a scratch copy of the follower's state (its own arm trails the SDK).
        """
        model, sim_data = self.ik.full_model, self.ik.full_model_data
        data = self.ik_data
        data.qpos[:] = sim_data.qpos
        data.qpos[self.qpos_addrs] = np.asarray(positions)[self.mirrored_indices]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        self.ik.full_model_data = data
        try:
            return np.asarray(self.original_solve(ik_input), dtype=float)
        finally:
            self.ik.full_model_data = sim_data

    def joint_targets(self):
        """The follower's current joints per SDK part, e.g. to put the SDK robot in the task posture."""
        qpos = self.env.sim.data
        return {part: np.array([qpos.get_joint_qpos(RS_PREFIX + self.robot.joint_names[i])
                                for i in self.robot.part_indices[part]])
                for part in SDK_PARTS}

    def close(self):
        self.ik.solve = self.original_solve


class SDKBaseDriver(rby1_utils._BaseDriver):
    """
    _BaseDriver that reads odometry and sends SE(2) velocity commands through
    the SDK, adding the session's wheel deadband (BASE_DEADBANDS) per wheel.
    rby1-sim's wheels settle at ~58 % of the commanded speed; the closed loop
    absorbs that.
    """

    def __init__(self, session, step_callback):
        self.session = session
        self.env = session.env
        self.sim = session.env.sim
        self.step_callback = step_callback
        self.stuck = False
        model = self.sim.model
        # the URDF wheel axle is at the same x (0.228 m) as the virtual base's yaw anchor
        self.axle_offset = float(model.jnt_pos[model.joint_name2id(rby1_utils.YAW_JOINT)][0])
        self.wheel_deadband = BASE_DEADBANDS[session.base_deadband]

    def pose(self):
        base_vec = self.session.robot.base_pose()
        return np.r_[self.axle_xy(base_vec), base_vec[2]]

    def _compensate(self, v, w):
        """Body velocity command whose wheel speeds include the deadband."""
        wheels = (v + np.array([-1.0, 1.0]) * w * WHEEL_TRACK / 2) / WHEEL_RADIUS  # left, right
        wheels = np.where(np.abs(wheels) < 1e-6, 0.0, wheels + np.sign(wheels) * self.wheel_deadband)
        return WHEEL_RADIUS * wheels.mean(), WHEEL_RADIUS * (wheels[1] - wheels[0]) / WHEEL_TRACK

    def step(self, v, w):
        self.session.robot.command_base(*self._compensate(v, w))
        self.session.tick(physics=False)
        self.step_callback()

    def hold(self):
        self.step(0.0, 0.0)


class SDKCartesianAdapter:
    """
    Stands in for robomimic's RBY1CartesianAdapter in execute_loop. "World" is
    the odometry frame aligned to the scene, and the TCP is measured with SDK
    FK. The env action it packs is the validated 9-D policy action, which
    SDKPolicyEnv sends.
    """

    base_site = "base"
    tcp_site = "tcp"

    def __init__(self, robot):
        self.robot = robot

    def site_transform(self, site):
        world_base = pose_to_transform(self.robot.base_pose())
        if site == self.base_site:
            return world_base
        if site == self.tcp_site:
            return world_base @ self.robot.tcp_in_base()
        raise ValueError(f"Unknown site {site}")

    @staticmethod
    def base_target(action):
        """The policy's TCP target in the robosuite base frame."""
        from robomimic.utils.rby1_cartesian import make_transform, rotation6d_to_matrix

        action = np.asarray(action, dtype=float)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise ValueError("Expected one finite 9D Cartesian action")
        return make_transform(action[:3], rotation6d_to_matrix(action[3:]))

    def target_world(self, action):
        return self.site_transform(self.base_site) @ self.base_target(action)

    def __call__(self, action):
        self.base_target(action)
        return np.asarray(action, dtype=float).copy()


class SDKPolicyEnv:
    """
    The env execute_loop steps: sends the TCP target, waits one period, syncs
    the follower.

    arm_control "joint" reproduces the training controller as a reference
    arm: q_ref moves by TRAINING_SERVO_RESPONSE of the WholeBody IK step
    solved at q_ref, exactly as robosuite's servos moved the arm, and the SDK
    streams q_ref. The IK uses the reference rather than the measured joints
    because the SDK arm trails each streamed target a little, and targets
    built from the trailing measurement compound that lag (the arm then
    crawls). If the measured arm falls more than ARM_REF_RESYNC behind (e.g. a
    joint limit, or contact on the real robot), q_ref restarts from it, as the
    training IK saw a blocked arm. This is also how the robot would run
    without a follower.

    arm_control "cartesian" streams the target to the SDK Cartesian
    controller. It faults ("Singularity") at the training start posture with
    the arm hanging straight down, and tracks differently from training.
    """

    def __init__(self, session, arm_control):
        from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter

        self.session = session
        self.arm_control = arm_control
        self.resyncs = 0
        self.max_ref_error = 0.0
        if arm_control == "joint":
            follower = session.follower
            # sets the IK nullspace reference (q0) to the current posture, as at the start of a sim rollout
            self.ik_adapter = RBY1CartesianAdapter(session.env)
            self.ik_dim = follower.ik.control_dim
            self.left = session.robot.part_indices["left_arm"]
            self.ik_slots = [follower.ik_indices.index(i) for i in self.left]
            session.robot.update()
            self.q_ref = session.robot.positions[self.left]

    @property
    def sim(self):
        return self.session.env.sim

    def step(self, action):
        robot = self.session.robot
        if self.arm_control == "cartesian":
            robot.command_tcp(SDKCartesianAdapter.base_target(action))
        else:
            positions = robot.positions
            positions[self.left] = self.q_ref
            q_ik = self.session.follower.solve_ik(self.ik_adapter(action)[:self.ik_dim], positions)[self.ik_slots]
            self.q_ref = robot.clip("left_arm", self.q_ref + TRAINING_SERVO_RESPONSE * (q_ik - self.q_ref))
            robot.command_left_arm(self.q_ref)
        self.session.tick()
        if self.arm_control == "joint":
            measured = robot.positions[self.left]
            error = float(np.max(np.abs(measured - self.q_ref)))
            self.max_ref_error = max(self.max_ref_error, error)
            if error > ARM_REF_RESYNC:
                self.q_ref = measured
                self.resyncs += 1


class SDKSession:
    """SDK robot + RoboCasa follower + control rate, with the pipeline stages that move the robot."""

    def __init__(self, env, address="localhost:50051", gripper_action=None, base_deadband="none",
                 arm_control="joint"):
        if base_deadband not in BASE_DEADBANDS:
            raise ValueError(f"Unknown base deadband {base_deadband}; known: {sorted(BASE_DEADBANDS)}")
        if arm_control not in ARM_CONTROLS:
            raise ValueError(f"Unknown arm control {arm_control}; known: {ARM_CONTROLS}")
        self.env = env
        self.base_deadband = base_deadband
        self.arm_control = arm_control
        self.robot = SDKRobot(address)
        try:
            self.follower = Follower(env, self.robot, gripper_action)
        except Exception:
            self.robot.close()
            raise
        self.rate = _Rate(CONTROL_FREQ)
        self.ticks = 0
        self.policy_env = None

    def tick(self, physics=True):
        """Waits for the end of the control period, then syncs the follower (see Follower.sync)."""
        self.rate.sleep()
        self.follower.sync(physics)
        self.ticks += 1

    def wait(self, handler, timeout, physics=True):
        """Ticks until a joint move finishes; returns its finish code."""
        deadline = time.perf_counter() + timeout
        while not handler.is_done():
            if time.perf_counter() > deadline:
                handler.cancel()
                raise RuntimeError(f"SDK joint move did not finish within {timeout:.1f} s")
            self.tick(physics)
        return handler.get().finish_code

    def attach(self, pose):
        """
        Aligns the SDK robot with the follower: odometry reads `pose` (the
        follower's base pose) and the joints move to the follower's posture
        (the task's torso, head and arm start pose).
        """
        self.robot.reset_base_pose(pose)
        self.robot.update()
        targets = self.follower.joint_targets()
        handler = self.robot.move_joints(targets)
        # the follower's own posture is the target, so it is only teleported along
        code = self.wait(handler, timeout=30.0, physics=False)
        if code != self.robot.rby.RobotCommandFeedback.FinishCode.Ok:
            raise RuntimeError(f"SDK posture move failed: {code}")
        for _ in range(CONTROL_FREQ // 2):  # let the follower's servos settle onto the SDK joints
            self.tick()
        self.follower.max_joint_error = self.follower.joint_error

    def base_driver(self, step_callback):
        return SDKBaseDriver(self, step_callback)

    def policy_interfaces(self):
        """(env, adapter, read_observation) for robomimic's execute_loop."""
        import robosuite.macros as macros
        from robosuite.utils.mjcf_utils import IMAGE_CONVENTION_MAPPING
        from robomimic.utils import obs_utils as O
        from robomimic.utils.rby1_evaluation import RGB_KEYS

        convention = IMAGE_CONVENTION_MAPPING[macros.IMAGE_CONVENTION]
        sim = self.env.sim

        def read():
            # low-dim from SDK FK, as on the real robot; RGB from the follower,
            # rendered as observation_reader does
            tcp = self.robot.tcp_in_base()
            result = {
                "robot0_base_to_left_eef_pos": tcp[:3, 3].copy(),
                "robot0_base_to_left_eef_quat_site": T.mat2quat(tcp[:3, :3]),
            }
            for key in RGB_KEYS:
                frame = sim.render(width=128, height=128, camera_name=key[:-6])[::convention][::-1].copy()
                result[key] = O.process_obs(frame, obs_key=key)
            return result

        self.policy_env = SDKPolicyEnv(self, self.arm_control)
        return self.policy_env, SDKCartesianAdapter(self.robot), read

    def return_arm(self, task, home_qpos, progress=None):
        """Moves the left arm back with one SDK joint move; same checks and result fields as return_arm_to_start."""
        task_progress = task.progress_fn(self.env)
        if task_progress() < task.success_threshold:
            raise ValueError("Arm return requires a completed task")
        left = self.robot.part_indices["left_arm"]
        home_qpos = np.asarray(home_qpos, dtype=float)
        base_start = self.robot.base_pose()
        if progress is None:
            progress = {}
        progress.update(completed=False, success=False, steps_executed=0, home_qpos=home_qpos.copy(),
                        task_stayed_done=True, num_contact_steps=0)
        if self.robot.stream is not None:  # end the policy's arm stream before the joint move
            self.robot.stream.cancel()
            self.robot.stream = None
        handler = self.robot.move_joints({"left_arm": home_qpos})
        settle_steps = CONTROL_FREQ
        deadline = time.perf_counter() + 30.0
        while settle_steps > 0:
            if handler.is_done():
                settle_steps -= 1
            elif time.perf_counter() > deadline:
                handler.cancel()
                progress["stop_reason"] = "SDK arm move did not finish"
                break
            self.tick()
            value = task_progress()
            progress.update(steps_executed=progress["steps_executed"] + 1, final_progress=value,
                            final_qpos=self.robot.positions[left])
            progress["num_contact_steps"] += int(bool(self.follower.contacts))
            if value < task.success_threshold:
                handler.cancel()
                progress["task_stayed_done"] = False
                progress["stop_reason"] = "Task undone during arm return"
                break
        error = float(np.max(np.abs(self.robot.positions[left] - home_qpos)))
        progress.update(completed=True, max_joint_error_rad=error,
                        base_drift_m=float(np.linalg.norm(self.robot.base_pose()[:2] - base_start[:2])),
                        final_contacts=self.follower.contacts)
        progress["success"] = bool(progress["task_stayed_done"] and error <= 0.02 and not progress["final_contacts"])
        print(f"Arm return (SDK): max joint error={error:.4f} rad, task still done={progress['task_stayed_done']}",
              flush=True)
        return progress

    def stats(self):
        return dict(
            robot_address=self.robot.address,
            control_ticks=self.ticks,
            tick_overruns=self.rate.overruns,
            max_tick_lateness_s=self.rate.max_lateness,
            follower_max_joint_error_rad=self.follower.max_joint_error,
            clipped_joint_commands=self.robot.clipped_commands,
            clipped_joints=dict(self.robot.clipped_joints),
            max_joint_clip_rad=self.robot.max_clip,
            arm_reference_resyncs=None if self.policy_env is None else self.policy_env.resyncs,
            max_arm_reference_error_rad=None if self.policy_env is None else self.policy_env.max_ref_error,
        )

    def close(self):
        try:
            self.follower.close()
        finally:
            self.robot.close()
