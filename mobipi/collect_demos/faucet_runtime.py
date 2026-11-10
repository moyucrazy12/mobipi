"""Task-local RB-Y1 faucet adapters. Import before creating or replaying environments."""
import importlib.util
import logging
from pathlib import Path
from types import MethodType
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import robosuite
import robocasa
from robosuite.models.grippers import GRIPPER_MAPPING
from robocasa.environments.kitchen.single_stage.kitchen_sink import TurnOnSinkFaucet
from mimicgen.env_interfaces.robosuite import RobosuiteInterface

from mobipi.collect_demos.auto_close_microwave_door import (
    build_controller_interface, create_robot_action, make_T, pose_target_for_ref,
    raw_data, raw_model, site_pose,
)

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets/robosuite"
CONTROLLER = ASSETS / "controllers/config/robots/default_rby1_whole_body_ik.json"
TCP = "gripper0_right_grip_site"
REF = "gripper0_right_ft_frame"

# Reuse the repository's adapter and XML without replacing installed packages.
_spec = importlib.util.spec_from_file_location(
    "_faucet_gripper", ASSETS / "models/grippers/rby1_gripper.py")
_gripper_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gripper_module)
_gripper_module.xml_path_completion = lambda name: str(ASSETS / "models/assets" / name)
GRIPPER_MAPPING["RBY1FaucetGripper"] = _gripper_module.RBY1Gripper


def handle_state(sink, env):
    """Same RoboCasa criterion, using qpos addresses rather than joint IDs."""
    handle = float(env.sim.data.get_joint_qpos(sink.naming_prefix + "handle_joint")) % (2 * np.pi)
    spout = float(env.sim.data.get_joint_qpos(sink.naming_prefix + "spout_joint")) % (2 * np.pi)
    orientation = "left" if np.pi <= spout <= 2*np.pi-np.pi/6 else (
        "right" if np.pi/6 <= spout <= np.pi else "center")
    return dict(handle_joint=handle, water_on=bool(0.40 < handle < np.pi),
                spout_joint=spout, spout_ori=orientation)


class TurnOnSinkFaucetRBY1(TurnOnSinkFaucet):
    """Original task with a scoped correction to the fixture state reader."""
    def __init__(self, faucet_initial_arm_qpos=None, faucet_hold_reset_pose=False, faucet_initial_torso_qpos=None, faucet_ik_mode="addressed", faucet_fixture_offset=None, **kwargs):
        if faucet_ik_mode not in ("addressed", "bounded"):
            raise ValueError("Unsupported faucet_ik_mode")
        self.faucet_ik_mode = faucet_ik_mode
        self.faucet_initial_arm_qpos = faucet_initial_arm_qpos
        self.faucet_hold_reset_pose = faucet_hold_reset_pose
        self.faucet_initial_torso_qpos = faucet_initial_torso_qpos
        offset = np.zeros(4) if faucet_fixture_offset is None else np.asarray(faucet_fixture_offset, dtype=float)
        if offset.shape != (4,) or not np.isfinite(offset).all():
            raise ValueError("faucet_fixture_offset requires finite world dx, dy, dz and yaw radians")
        self.faucet_fixture_offset = offset
        self._faucet_offset_model = None
        self._faucet_resetting = False
        self._reset_solver = None
        try:
            super().__init__(**kwargs)
        except Exception:
            # A constructor failure cannot return an environment to the caller
            # for cleanup. Compilation may have failed before sim was assigned.
            self.sim = getattr(self, "sim", None)
            self.viewer = getattr(self, "viewer", None)
            try:
                self.close()
            except Exception:
                logging.getLogger(__name__).exception("Failed to close partially initialized faucet environment")
            raise

    def _reset_internal(self):
        self._faucet_resetting = self.faucet_hold_reset_pose
        self._reset_solver = None
        try:
            super()._reset_internal()
        finally:
            self._faucet_resetting = False
            if self._reset_solver is not None:
                policy, solve = self._reset_solver
                policy.solve = solve
        if self.faucet_ik_mode == "bounded":
            # Persist solver choice in environment metadata for ordinary replay.
            self._faucet_control_interface = MG_RBY1TurnOnSinkFaucet(self)

    def _pre_action(self, action, policy_step=False):
        # RoboCasa settles objects using an all-zero action. For absolute WBIK
        # that commands both wrists to world origin, destroying the reset posture.
        if self._faucet_resetting:
            robot = self.robots[0]
            policy = robot.composite_controller.joint_action_policy
            if self._reset_solver is None:
                self._reset_solver = (policy, policy.solve)
                hold = np.array([float(self.sim.data.get_joint_qpos(n)) for n in policy.joint_names])
                policy.solve = lambda action: hold.copy()
            parts = {}
            for part in ("torso", "head"):
                ctrl = robot.part_controllers.get(part)
                if ctrl is not None:
                    parts[part] = self.sim.data.qpos[ctrl.qpos_index].copy()
            action = robot.create_action_vector(parts)
        return super()._pre_action(action, policy_step)

    def _load_model(self):
        super()._load_model()
        # Apply once to each newly assembled XML, including recursive placement
        # retries. Move the complete static sink fixture before compilation;
        # the robot base, hinge state and manipulation controller are unchanged.
        if np.any(self.faucet_fixture_offset) and self._faucet_offset_model is not self.model:
            self._faucet_offset_model = self.model
            self.sink.set_pos(np.asarray(self.sink.pos) + self.faucet_fixture_offset[:3])
            euler = self.sink.euler.copy()
            euler[2] += self.faucet_fixture_offset[3]
            self.sink.set_euler(euler)
        if self.faucet_initial_torso_qpos is not None:
            torso = np.asarray(self.faucet_initial_torso_qpos, dtype=float)
            if torso.shape != (6,) or not np.isfinite(torso).all():
                raise ValueError("faucet_initial_torso_qpos must contain six finite joint angles")
            self.robots[0].init_torso_qpos = torso
        if self.faucet_initial_arm_qpos is not None:
            values = np.asarray(self.faucet_initial_arm_qpos, dtype=float)
            if values.shape != (7,) or not np.isfinite(values).all():
                raise ValueError("faucet_initial_arm_qpos must contain seven finite joint angles")
            robot = self.robots[0]
            initial = np.asarray(robot.robot_model.init_qpos).copy()
            for index, value in enumerate(values):
                initial[robot.robot_model.joints.index(f"robot0_right_arm_{index}")] = value
            robot.init_qpos = initial
        # Fixture augmentation must not move the overview camera with the sink.
        target = np.asarray(self.sink.pos) - self.faucet_fixture_offset[:3] + [0., 0., .05]
        position = target + [-1.2, 1.4, 1.1]
        z = position-target
        z /= np.linalg.norm(z)
        x = np.cross([0., 0., 1.], z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        quat = np.roll(Rotation.from_matrix(np.column_stack([x, y, z])).as_quat(), 1)
        # Kitchen._load_model may recursively retry placement through this
        # override. Each unwinding call sees the same final model, so update its
        # camera instead of appending another one. No shared camera config/XML
        # is modified; the element belongs to this environment's task model.
        camera = self.model.worldbody.find("camera[@name='faucet_overview']")
        if camera is None:
            camera = ET.SubElement(self.model.worldbody, "camera", name="faucet_overview")
        camera.set("pos", " ".join(map(str, position)))
        camera.set("quat", " ".join(map(str, quat)))
        camera.set("fovy", "55")

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.sink.get_handle_state = MethodType(handle_state, self.sink)


class UnsafeTrajectory(RuntimeError):
    """A target or simulated configuration failed the collection safety gate."""


class FaucetGeometry:
    def __init__(self, env):
        self.env = env
        self.m, self.d = raw_model(env.sim), raw_data(env.sim)
        prefix = env.sink.naming_prefix
        self.joint = env.sim.model.joint_name2id(prefix + "handle_joint")
        self.handle = env.sim.model.geom_name2id(prefix + "handle_main")
        self.water = env.sim.model.site_name2id(prefix + "water")
        if self.m.jnt_type[self.joint] != mujoco.mjtJoint.mjJNT_HINGE:
            raise UnsafeTrajectory("Unsupported faucet: handle must be a hinge")
        if self.m.jnt_range[self.joint, 1] <= .42:
            raise UnsafeTrajectory("Unsupported faucet: hinge cannot reach water-on threshold")
        if not (self.m.geom_contype[self.handle] or self.m.geom_conaffinity[self.handle]):
            raise UnsafeTrajectory("Handle has no collision geometry")
        self.root_body = env.sim.model.body_name2id(env.sink.root_body)
        self.finger_geoms = []
        self.arm_geoms = []
        shoulder = int(self.m.jnt_bodyid[env.sim.model.joint_name2id("robot0_right_arm_0")])
        for g in range(self.m.ngeom):
            body_id = int(self.m.geom_bodyid[g])
            body = env.sim.model.body_id2name(body_id) or ""
            ancestor = body_id
            while ancestor not in (0, shoulder):
                ancestor = int(self.m.body_parentid[ancestor])
            if ancestor == shoulder:
                if self.m.geom_contype[g] or self.m.geom_conaffinity[g]:
                    self.arm_geoms.append(g)
                    if "finger_r" in body:
                        self.finger_geoms.append(g)
        if not self.finger_geoms or not self.arm_geoms:
            raise UnsafeTrajectory("Cannot identify RB-Y1 right-arm collision geometry")
        # Interior sites define an oriented basin volume, including additional bins.
        self.basins = []
        for name in env.sim.model.site_names:
            if name and name.startswith(prefix) and name.endswith("int_p0"):
                stem = name[:-2]
                points = [site_pose(env.sim, stem+s)[0] for s in ("p0", "px", "py", "pz")]
                origin = points[0]
                axes = np.column_stack([p-origin for p in points[1:]])
                if abs(np.linalg.det(axes)) > 1e-8:
                    self.basins.append((origin, axes))
        if not self.basins:
            raise UnsafeTrajectory("Missing valid sink basin bounds")

    def contact(self):
        return any((c.geom1 in self.finger_geoms and c.geom2 == self.handle) or
                   (c.geom2 in self.finger_geoms and c.geom1 == self.handle)
                   for c in self.d.contact[:self.d.ncon])

    def check(self, data=None):
        """Conservative world AABBs cover complete collision geoms, not just TCP."""
        d = self.d if data is None else data
        if not np.isfinite(d.qpos).all():
            raise UnsafeTrajectory("Non-finite simulator state")
        water_pos = d.site_xpos[self.water]
        water_rot = d.site_xmat[self.water].reshape(3, 3)
        # Water site cylinder local z axis, regardless of visibility / water state.
        water_half = np.array([self.m.site_size[self.water, 0]]*2 +
                              [self.m.site_size[self.water, 1]])
        for g in self.arm_geoms:
            rot = d.geom_xmat[g].reshape(3, 3)
            # MuJoCo geom_aabb stores center + half extents in geom coordinates.
            local_center, half = self.m.geom_aabb[g, :3], self.m.geom_aabb[g, 3:]
            center = d.geom_xpos[g] + rot @ local_center
            extent = np.abs(rot) @ half + .008
            for origin, axes in self.basins:
                inv = np.linalg.inv(axes)
                p, r = inv @ (center-origin), np.abs(inv) @ extent
                if np.all(p+r > 0) and np.all(p-r < 1):
                    raise UnsafeTrajectory(f"Right-arm geometry {g} overlaps sink basin")
            p = water_rot.T @ (center-water_pos)
            r = np.abs(water_rot.T) @ extent
            if np.all(np.abs(p) < water_half+r):
                raise UnsafeTrajectory(f"Right-arm geometry {g} overlaps water-flow volume")
        # Only finger-to-handle contact is allowed with the kitchen.
        for c in d.contact[:d.ncon]:
            pair = {int(c.geom1), int(c.geom2)}
            if not pair.intersection(self.arm_geoms):
                continue
            if pair <= set(self.arm_geoms):
                continue
            other = c.geom2 if c.geom1 in self.arm_geoms else c.geom1
            other_body = self.env.sim.model.body_id2name(int(self.m.geom_bodyid[other])) or ""
            if other_body.startswith("robot0_"):
                continue
            if self.handle in pair and pair.intersection(self.finger_geoms):
                continue
            if c.dist < -.001:
                raise UnsafeTrajectory(f"Disallowed arm contact {c.geom1}/{c.geom2}: "
                                       f"{self.env.sim.model.body_id2name(int(self.m.geom_bodyid[c.geom1]))} / "
                                       f"{self.env.sim.model.body_id2name(int(self.m.geom_bodyid[c.geom2]))}")

    def landmark(self):
        return self.d.geom_xpos[self.handle].copy()

    def arc(self, point, angle):
        anchor = self.d.xanchor[self.joint].copy()
        axis = self.d.xaxis[self.joint].copy()
        return anchor + Rotation.from_rotvec(axis*angle).apply(point-anchor)


class MG_RBY1TurnOnSinkFaucet(RobosuiteInterface):
    """TCP-space MimicGen interface with absolute world WholeBody IK commands."""
    def __init__(self, env):
        super().__init__(env)
        self.refresh()

    def refresh(self):
        robot = self.env.robots[0]
        config = robot.composite_controller_config
        (self.arm, _, ref, _, _, self.tcp_ref, self.holds) = build_controller_interface(
            self.env, robot, config, "right")
        if ref != REF or not np.isclose(np.linalg.norm(self.tcp_ref[:3, 3]), .2501, atol=.002):
            raise UnsafeTrajectory("RB-Y1 TCP / wrist definitions differ from repository model")
        policy = robot.composite_controller.joint_action_policy
        if policy.input_action_repr != "absolute" or policy.input_rotation_repr != "axis_angle":
            raise UnsafeTrajectory("Requires absolute world IK axis-angle targets")
        self.geom = FaucetGeometry(self.env)
        configure_addressed_ik(self.env)
        self._policy = policy
        # Hold inactive gripper at its current scalar joint position.
        left = robot.gripper["left"]
        actuator = self.env.sim.model.actuator_name2id(left.actuators[0])
        low, high = raw_model(self.env.sim).actuator_ctrlrange[actuator]
        q = float(self.env.sim.data.get_joint_qpos(left.joints[0]))
        self.left_gripper_hold = float(np.clip(2*(q-low)/(high-low)-1, -1, 1))
        if self.env.faucet_ik_mode == "bounded":
            from mobipi.collect_demos.faucet_bounded_ik import bind_bounded_ik
            self.bounded = bind_bounded_ik(self)

    def get_datagen_info(self, action=None):
        if self.env.robots[0].composite_controller.joint_action_policy is not self._policy:
            self.refresh()
        return super().get_datagen_info(action)

    def get_robot_eef_pose(self):
        return make_T(*site_pose(self.env.sim, TCP))

    def target_pose_to_action(self, target_pose, relative=True):
        # API's relative flag concerns OSC scaling; this interface always emits absolute IK.
        target_pose = np.asarray(target_pose)
        if target_pose.shape != (4, 4) or not np.isfinite(target_pose).all():
            raise UnsafeTrajectory("Non-finite target")
        target = pose_target_for_ref(target_pose[:3, 3], target_pose[:3, :3], self.tcp_ref)
        action = create_robot_action(self.env.robots[0], "right", target, self.holds, 1.)
        start, end = self.env.robots[0].composite_controller._whole_body_controller_action_split_indexes["left_gripper"]
        action[start:end] = self.left_gripper_hold
        return action

    def full_action(self, target_pose, gripper):
        action = self.target_pose_to_action(target_pose)
        start, end = self.env.robots[0].composite_controller._whole_body_controller_action_split_indexes["right_gripper"]
        action[start:end] = gripper
        return action

    def action_to_target_pose(self, action, relative=True):
        start, end = self.env.robots[0].composite_controller._whole_body_controller_action_split_indexes["right"]
        a = np.asarray(action)[start:end]
        return make_T(a[:3], Rotation.from_rotvec(a[3:6]).as_matrix()) @ np.linalg.inv(self.tcp_ref)

    def action_to_gripper_action(self, action):
        start, end = self.env.robots[0].composite_controller._whole_body_controller_action_split_indexes["right_gripper"]
        return np.asarray(action)[start:end]

    def get_object_poses(self):
        # Stable fixture frame throughout articulation. Moving handle is diagnostics only.
        return dict(faucet=self.get_object_pose(self.env.sink.root_body, "body"),
                    handle=self.get_object_pose(self.env.sink.naming_prefix+"handle_main", "geom"))

    def get_subtask_term_signals(self):
        return dict(contact=int(self.geom.contact()),
                    water_on=int(self.env.sink.get_handle_state(self.env)["water_on"]))

    def get_controller_base_pose(self):
        body = self.env.robots[0].robot_model.root_body
        pose = self.get_object_pose(body, "body")
        return pose[:3, 3], pose[:3, :3]

    def preflight(self, action):
        self.geom.check()
        policy = self.env.robots[0].composite_controller.joint_action_policy
        if np.asarray(action).shape != (self.env.action_dim,) or not np.isfinite(action).all():
            raise UnsafeTrajectory("Invalid full robot action")
        q = policy.solve(np.asarray(action[:policy.control_dim]).copy())
        if not np.isfinite(q).all():
            raise UnsafeTrajectory("IK returned non-finite joints")
        m = raw_model(self.env.sim)
        trial = mujoco.MjData(m)
        trial.qpos[:] = raw_data(self.env.sim).qpos
        trial.qpos[policy.qpos_addresses] = q
        limits = m.jnt_range[policy.joint_ids]
        if limits is not None and np.any((q < limits[:, 0]-.01) | (q > limits[:, 1]+.01)):
            raise UnsafeTrajectory("IK joint limits violated")
        mujoco.mj_forward(m, trial)
        self.geom.check(trial)


def addressed_solve(policy, target_action, Kpos=.95, Kori=.95):
    """Existing damped least-squares IK with distinct joint / qpos / dof indices.

    Bound only to this task's WholeBody IK instance. RB-Y1 has a free root.
    """
    m, d = policy.full_model, policy.full_model_data
    targets = np.asarray(target_action).reshape(len(policy.site_ids), 6)
    jacobians, twists = [], []
    for site, target in zip(policy.site_ids, targets):
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jp, jr, site)
        jacobians.append(np.vstack((jp, jr))[:, policy.velocity_addresses])
        current_rot = d.site_xmat[site].reshape(3, 3)
        error_rot = Rotation.from_rotvec(target[3:]).as_matrix() @ current_rot.T
        twists.append(np.r_[Kpos*(target[:3]-d.site_xpos[site]),
                            Kori*Rotation.from_matrix(error_rot).as_rotvec()] / policy.integration_dt)
    jac, twist = np.vstack(jacobians), np.concatenate(twists)
    dq = jac.T @ np.linalg.solve(jac@jac.T + policy.damping**2*np.eye(len(twist)), twist)
    q = d.qpos[policy.qpos_addresses]
    dq += (np.eye(len(q))-np.linalg.pinv(jac)@jac) @ (policy.Kn*(policy.q0-q))
    if policy.max_dq > 0:
        dq *= min(1., policy.max_dq/max(np.max(np.abs(dq)), 1e-12))
    result = q + dq*policy.integration_dt
    return np.clip(result, *m.jnt_range[policy.joint_ids].T)


def configure_addressed_ik(env):
    policy = env.robots[0].composite_controller.joint_action_policy
    m = raw_model(env.sim)
    policy.joint_ids = np.array([m.joint(n).id for n in policy.joint_names])
    if np.any(m.jnt_type[policy.joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
        raise UnsafeTrajectory("Expected scalar hinge arm joints")
    policy.qpos_addresses = m.jnt_qposadr[policy.joint_ids]
    policy.velocity_addresses = m.jnt_dofadr[policy.joint_ids]
    policy.q0 = raw_data(env.sim).qpos[policy.qpos_addresses].copy()
    policy.solve = MethodType(addressed_solve, policy)
