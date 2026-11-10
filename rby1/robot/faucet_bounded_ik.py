"""Joint-limit-aware numerical IK scoped to the faucet environment."""
from time import perf_counter
from types import MethodType

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from rby1.robot.faucet_runtime import raw_model, raw_data, TCP, UnsafeTrajectory


class BoundedFaucetIK:
    def __init__(self, interface):
        self.interface = interface
        self.policy = interface._policy
        self.model = raw_model(interface.env.sim)
        self.trial = mujoco.MjData(self.model)
        self.indices = np.array([self.policy.joint_names.index(f"robot0_right_arm_{j}") for j in range(7)])
        self.addresses = self.policy.qpos_addresses[self.indices]
        self.joints = self.policy.joint_ids[self.indices]
        self.low, self.high = self.model.jnt_range[self.joints].T
        self.tcp = self.model.site(TCP).id
        self.last = {}
        self.optimization_seconds = []
        self.command = None
        self.command_pose = None
        self.left_hold = raw_data(interface.env.sim).qpos[self.policy.qpos_addresses].copy()

    def forward(self, q):
        self.trial.qpos[:] = raw_data(self.interface.env.sim).qpos
        self.trial.qpos[self.addresses] = q
        mujoco.mj_kinematics(self.model, self.trial)
        pose = np.eye(4)
        pose[:3, 3] = self.trial.site_xpos[self.tcp]
        pose[:3, :3] = self.trial.site_xmat[self.tcp].reshape(3, 3)
        return pose

    def optimize(self, target, seed=None, max_evaluations=180):
        start = perf_counter()
        self.trial.qpos[:] = raw_data(self.interface.env.sim).qpos
        if seed is None:
            seed = self.trial.qpos[self.addresses].copy()
        def residual(q):
            pose = self.forward(q)
            return np.r_[pose[:3, 3]-target[:3, 3],
                .25*Rotation.from_matrix(target[:3, :3]@pose[:3, :3].T).as_rotvec()]
        result = least_squares(residual, np.clip(seed, self.low+1e-9, self.high-1e-9),
                               bounds=(self.low, self.high), max_nfev=max_evaluations,
                               ftol=1e-8, xtol=1e-8, gtol=1e-9)
        error = residual(result.x)
        self.last = dict(seconds=perf_counter()-start, evaluations=result.nfev,
                         position_error=float(np.linalg.norm(error[:3])),
                         orientation_error=float(np.linalg.norm(error[3:])/.25),
                         joint_target=result.x.copy(), converged=bool(result.success))
        self.optimization_seconds.append(self.last["seconds"])
        return result.x.copy()

    def check(self, q):
        if not np.isfinite(q).all() or np.any(q < self.low-1e-8) or np.any(q > self.high+1e-8):
            raise UnsafeTrajectory("Bounded IK violates scalar joint limits")
        self.trial.qpos[:] = raw_data(self.interface.env.sim).qpos
        self.trial.qpos[self.addresses] = q
        mujoco.mj_forward(self.model, self.trial)
        self.check_scene(self.trial)

    def check_scene(self, data):
        self.interface.geom.check(data)
        model = self.interface.env.sim.model
        for contact in data.contact[:data.ncon]:
            one, two = int(contact.geom1), int(contact.geom2)
            body_one = model.body_id2name(int(self.model.geom_bodyid[one])) or ""
            body_two = model.body_id2name(int(self.model.geom_bodyid[two])) or ""
            robot_one, robot_two = body_one.startswith("robot0_"), body_two.startswith("robot0_")
            if robot_one == robot_two or contact.dist >= -.001:
                continue
            other = two if robot_one else one
            other_name = model.geom_id2name(other) or ""
            if self.model.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE or "floor" in other_name:
                continue
            if other == self.interface.geom.handle and (one in self.interface.geom.finger_geoms or two in self.interface.geom.finger_geoms):
                continue
            raise UnsafeTrajectory(f"Robot/kitchen collision: {body_one} / {body_two}")

    def command_joints(self, q):
        self.check(q)
        self.command = np.asarray(q).copy()
        self.command_pose = self.forward(q)
        return self.command_pose.copy()

    def solve(self, action):
        # Controller input remains the existing absolute wrist pose action.
        ref = np.asarray(action[:6])
        wrist = np.eye(4)
        wrist[:3, 3] = ref[:3]
        wrist[:3, :3] = Rotation.from_rotvec(ref[3:]).as_matrix()
        target = wrist @ np.linalg.inv(self.interface.tcp_ref)
        if self.command is not None and np.allclose(target, self.command_pose, atol=1e-7):
            q = self.command
        else:
            q = self.optimize(target, max_evaluations=80)
        result = self.left_hold.copy()
        result[self.indices] = q
        return result


def bind_bounded_ik(interface):
    solver = BoundedFaucetIK(interface)
    interface._policy.solve = MethodType(lambda policy, action, **kwargs: solver.solve(action), interface._policy)
    return solver
