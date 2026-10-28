"""Closed-loop source-demo motion; all commands remain Cartesian TCP targets."""

import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares


class DemoFailure(RuntimeError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def descendants(model, root):
    result = set()
    for body in range(model.nbody):
        parent = body
        while parent:
            if parent == root:
                result.add(body)
                break
            parent = int(model.body_parentid[parent])
    return result | {root}


def hand_body_ids(model, arm="left"):
    """RB-Y1 fingers are embedded in the robot, separate from gripper sites."""
    suffix = "l" if arm == "left" else "r"
    roots = {f"robot0_{arm}_hand", f"robot0_EE_BODY_{suffix.upper()}",
             f"robot0_ee_finger_{suffix}1", f"robot0_ee_finger_{suffix}2"}
    bodies = set()
    for body in range(model.nbody):
        name = model.body(body).name
        if name in roots or name.startswith(f"gripper0_{arm}_"):
            bodies.update(descendants(model, body))
    if not bodies:
        raise ValueError(f"No {arm} hand collision bodies found")
    return bodies


def door_contacts(model, data, door_bodies, hand_bodies):
    contacts, unexpected = [], []
    for contact in data.contact[:data.ncon]:
        if contact.dist >= 0:
            continue
        g1, g2 = int(contact.geom1), int(contact.geom2)
        names = (model.geom(g1).name or "", model.geom(g2).name or "")
        robot = [name.startswith(("robot0_", "gripper0_")) for name in names]
        if robot[0] == robot[1]:
            continue
        rb, other = (g1, g2) if robot[0] else (g2, g1)
        rb_body, other_body = int(model.geom_bodyid[rb]), int(model.geom_bodyid[other])
        if (model.geom(other).name or "").startswith("floor"):
            continue
        if rb_body in hand_bodies and other_body in door_bodies:
            contacts.append(contact)
        elif contact.dist < -0.0005:
            unexpected.append(names)
    return contacts, unexpected


class DoorMotion:
    def __init__(self, env, command, tcp_pose, base_pose, hinge, panel_point, command_args):
        self.env, self.command, self.tcp_pose = env, command, tcp_pose
        self.model = getattr(env.sim.model, "_model", env.sim.model)
        self.data = getattr(env.sim.data, "_data", env.sim.data)
        self.args = command_args
        self.dt = env.control_timestep
        self.hinge = hinge
        self.qadr = int(self.model.jnt_qposadr[hinge])
        self.vadr = int(self.model.jnt_dofadr[hinge])
        self.anchor = self.data.xanchor[hinge].copy()
        self.axis = self.data.xaxis[hinge].copy()
        self.theta_start = self.angle
        self.panel_point = np.asarray(panel_point).copy()
        self.door_bodies = descendants(self.model, int(self.model.jnt_bodyid[hinge]))
        self.hand_bodies = hand_body_ids(self.model, command_args.arm)
        p, r = tcp_pose()
        self.target, self.rotation = p.copy(), r.copy()
        self.initial_base = base_pose()
        self.home = np.linalg.inv(self.initial_base) @ self.transform(p, r)
        self.arm_qpos = env.robots[0].composite_controller.part_controllers[command_args.arm].qpos_index
        self.home_qpos = self.data.qpos[self.arm_qpos].copy()
        self.tcp_id = self.model.site(f"gripper0_{command_args.arm}_grip_site").id
        self.base_pose = base_pose
        self.phase = "settle"
        self.trace = []
        self.events = {}
        self.contact_count = 0
        self.closed_count = 0

    @staticmethod
    def transform(p, r):
        result = np.eye(4)
        result[:3, :3], result[:3, 3] = r, p
        return result

    @property
    def angle(self):
        return float(self.data.qpos[self.qadr])

    @property
    def closed(self):
        return abs(self.angle) <= self.args.close_tolerance

    def geometry(self, theta=None):
        theta = self.angle if theta is None else theta
        rotation = Rotation.from_rotvec(self.axis * (theta - self.theta_start)).as_matrix()
        point = self.anchor + rotation @ (self.panel_point - self.anchor)
        radial = point - self.anchor
        radial -= np.dot(radial, self.axis) * self.axis
        radius = np.linalg.norm(radial)
        radial /= radius
        tangent = np.cross(self.axis, radial) * np.sign(-self.theta_start)
        return point, radial, tangent, radius

    def event(self, name, trace_index=None):
        if name not in self.events:
            index = len(self.trace) if trace_index is None else trace_index
            self.events[name] = index
            if index < len(self.trace):
                self.trace[index][name] = True
            print(f"EVENT {name}: step={index} hinge={self.angle:.4f}", flush=True)

    def step(self, goal, rotation, allow_door=False, monitor_closed=False, speed=None):
        started = time.perf_counter()
        speed = self.args.tcp_speed if speed is None else speed
        delta = np.asarray(goal) - self.target
        self.target += delta * min(1.0, speed * self.dt / max(np.linalg.norm(delta), 1e-9))
        # Limit target lead over measured TCP to avoid integrating unreachable targets.
        actual, _ = self.tcp_pose()
        delta = self.target - actual
        self.target = actual + delta * min(1.0, 0.06 / max(np.linalg.norm(delta), 1e-9))
        dr = Rotation.from_matrix(rotation @ self.rotation.T).as_rotvec()
        dr *= min(1.0, np.deg2rad(self.args.orientation_speed_deg) * self.dt / max(np.linalg.norm(dr), 1e-9))
        self.rotation = Rotation.from_rotvec(dr).as_matrix() @ self.rotation
        self.command(self.target, self.rotation)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise DemoFailure("controller_error", "Nonfinite simulator state")
        contacts, collisions = door_contacts(self.model, self.data, self.door_bodies, self.hand_bodies)
        actual, actual_r = self.tcp_pose()
        self.contact_count = self.contact_count + 1 if contacts else 0
        if self.contact_count >= 3:
            self.event("stage_contact_door")
        quiet_closed = self.closed and abs(self.data.qvel[self.vadr]) < 0.04
        self.closed_count = self.closed_count + 1 if quiet_closed else 0
        self.trace.append(dict(phase=self.phase, hinge=self.angle, hinge_velocity=float(self.data.qvel[self.vadr]),
                               tcp=actual.copy(), target=self.target.copy(), rotation=actual_r.copy(),
                               target_rotation=self.rotation.copy(), closed=bool(quiet_closed),
                               base=self.base_pose(),
                               contact=bool(contacts), contact_count=len(contacts),
                               tracking_error=float(np.linalg.norm(actual - self.target)),
                               arm_qpos=self.data.qpos[self.arm_qpos].copy(),
                               **{name: name in self.events for name in
                                  ("stage_contact_door", "stage_closed_door", "stage_released_door", "stowed")}))
        if collisions:
            raise DemoFailure("collision", str(collisions[:4]))
        if contacts and not allow_door:
            raise DemoFailure("collision", "Unexpected hand-door contact outside contact motion")
        if monitor_closed and abs(self.angle) > self.args.close_tolerance + 0.015:
            raise DemoFailure("rebound", f"Door reopened to {np.rad2deg(self.angle):.2f} degrees")
        if len(self.trace) % 100 == 0:
            print(f"{self.phase}: step={len(self.trace)} hinge={self.angle:.3f} "
                  f"contacts={len(contacts)} tracking={np.linalg.norm(actual-self.target):.3f}m", flush=True)
        if not self.args.headless:
            time.sleep(max(0, self.dt - (time.perf_counter() - started)))
        return actual, actual_r, bool(contacts)

    def move(self, goal, rotation, phase, allow_door=False, monitor_closed=False, tolerance=0.015):
        self.phase = phase
        start, start_r = self.tcp_pose()
        angular = Rotation.from_matrix(rotation @ start_r.T).magnitude()
        duration = max(np.linalg.norm(goal-start) / self.args.tcp_speed,
                       angular / np.deg2rad(self.args.orientation_speed_deg))
        for _ in range(int((duration * 2 + 5) / self.dt)):
            p, r, _ = self.step(goal, rotation, allow_door, monitor_closed)
            error = Rotation.from_matrix(rotation @ r.T).magnitude()
            if np.linalg.norm(p-goal) < tolerance and error < np.deg2rad(8):
                return
        reason = "failed_stow" if phase == "stow" else "unreachable_contact"
        raise DemoFailure(reason, f"{phase}: position error={np.linalg.norm(p-goal):.3f} m, rotation={np.rad2deg(error):.1f} deg")

    def posture_path(self, goal_qpos, phase, monitor_closed=False, straight_tcp=False):
        """Follow arm postures, or couple TCP translation and rotation along a straight reach."""
        fk = mujoco.MjData(self.model)
        fk.qpos[:] = self.data.qpos
        start = self.data.qpos[self.arm_qpos].copy()
        if straight_tcp:
            start_p, start_r = self.tcp_pose()
            fk.qpos[self.arm_qpos] = goal_qpos
            mujoco.mj_kinematics(self.model, fk)
            goal_p = fk.site_xpos[self.tcp_id].copy()
            turn = Rotation.from_matrix(fk.site_xmat[self.tcp_id].reshape(3,3) @ start_r.T).as_rotvec()
        for u in np.linspace(0, 1, 61)[1:]:
            blend = u*u*(3-2*u)
            if straight_tcp:
                point = start_p + blend * (goal_p-start_p)
                rotation = Rotation.from_rotvec(blend * turn).as_matrix() @ start_r
            else:
                fk.qpos[self.arm_qpos] = start + blend * (goal_qpos-start)
                mujoco.mj_kinematics(self.model, fk)
                point = fk.site_xpos[self.tcp_id].copy()
                rotation = fk.site_xmat[self.tcp_id].reshape(3,3).copy()
            self.move(point, rotation, phase, monitor_closed=monitor_closed, tolerance=0.03)

    def approach_posture(self, point, rotation):
        """Resolve the initial straight-arm singularity in scratch FK data."""
        fk = mujoco.MjData(self.model)
        fk.qpos[:] = self.data.qpos
        joints = self.env.robots[0].composite_controller.part_controllers[self.args.arm].joint_names
        limits = np.array([self.model.jnt_range[self.model.joint(name).id] for name in joints])
        lower, upper = limits[:, 0]+1e-4, limits[:, 1]-1e-4
        start = np.clip(self.data.qpos[self.arm_qpos], lower, upper)

        def residual(q):
            fk.qpos[self.arm_qpos] = q
            mujoco.mj_kinematics(self.model, fk)
            r = fk.site_xmat[self.tcp_id].reshape(3,3)
            return np.r_[5*(fk.site_xpos[self.tcp_id]-point),
                         0.5*Rotation.from_matrix(rotation @ r.T).as_rotvec(), 0.005*(q-start)]

        solutions = []
        for bend in (0, -1.2, 1.2):
            seed = start.copy()
            seed[3] = np.clip(bend, lower[3], upper[3])
            solution = least_squares(residual, seed, bounds=(lower,upper), max_nfev=400)
            solutions.append(solution)
        solution = min(solutions, key=lambda item: np.linalg.norm(item.fun))
        residual(solution.x)
        position_error = np.linalg.norm(fk.site_xpos[self.tcp_id]-point)
        rotation_error = Rotation.from_matrix(rotation @ fk.site_xmat[self.tcp_id].reshape(3,3).T).magnitude()
        print(f"Approach IK: position={position_error:.4f} m, orientation={np.rad2deg(rotation_error):.1f} deg", flush=True)
        if position_error > 0.015 or rotation_error > np.deg2rad(25):
            raise DemoFailure("unreachable_contact", "No reachable approach pose within position/orientation bounds")
        return solution.x

    def run(self, facing_rotation):
        point, radial, tangent, radius = self.geometry()
        approach_rotation = facing_rotation(tangent)
        pre = point - (self.args.standoff + self.args.contact_clearance) * tangent
        lift, _ = self.tcp_pose()
        # Keep the clearance posture outside the panel half-space.
        lift -= max(0.0, np.dot(lift-pre, tangent)) * tangent
        lift[2] = point[2]
        lift_degrees = self.args.pre_lift_deg
        if lift_degrees is None:
            lift_degrees = 90.0 if self.args.lift_before_turn else 0.0
        raised_qpos = None
        if lift_degrees > 0:
            # At closer base placements, turning while low can sweep the fingers
            # through the stove/counter. Raise sideways with the elbow/wrist
            # bent before reaching toward the panel. This is a scratch-FK
            # waypoint; posture_path still sends the usual Cartesian actions.
            raised_qpos = self.home_qpos.copy()
            raised_qpos[1] = np.deg2rad(lift_degrees) * (1 if self.args.arm == "left" else -1)
            print(f"Clearance lift: shoulder={lift_degrees:g} deg", flush=True)
            self.posture_path(raised_qpos, "lift")
            raised_qpos = self.data.qpos[self.arm_qpos].copy()
        safe_lift_qpos = self.approach_posture(lift, approach_rotation)
        if not self.args.direct_approach:
            self.posture_path(safe_lift_qpos, "lift")
            safe_lift_qpos = self.data.qpos[self.arm_qpos].copy()
        # Direct approach aligns during the reach. Keep the clearance posture
        # for the return to stow even when the approach does not visit it.
        if self.args.direct_approach:
            print("Direct approach: straight TCP reach", flush=True)
        self.posture_path(self.approach_posture(pre, approach_rotation), "approach",
                          straight_tcp=self.args.direct_approach)
        _, approach_rotation = self.tcp_pose()

        self.phase = "contact"
        clearance = self.args.contact_clearance + self.args.standoff
        for _ in range(self.args.contact_search_steps):
            point, _, tangent, _ = self.geometry()
            goal = point - clearance * tangent
            goal += self.args.contact_z_offset * self.axis
            p, r, touching = self.step(goal, approach_rotation, allow_door=True, speed=0.03)
            if "stage_contact_door" in self.events:
                break
            clearance = max(-self.args.max_contact_penetration, clearance - self.args.contact_search_step)
        else:
            raise DemoFailure("unreachable_contact", "No sustained active-hand contact during contact search")

        # Keep only normal clearance and height; radial position may slide on the panel.
        point, _, tangent, radius = self.geometry()
        normal_offset = float(np.dot(p - point, tangent))
        height_offset = float(np.dot(p - point, self.axis))
        contact_rotation, contact_theta = r.copy(), self.angle
        lost = 0
        history = []
        self.phase = "close"
        for _ in range(self.args.max_push_steps):
            point, radial, tangent, radius = self.geometry()
            p, _ = self.tcp_pose()
            error = -self.angle
            taper = np.clip(abs(error) / self.args.finish_angle, 0, 1)
            taper = taper * taper * (3 - 2 * taper)
            lead = np.sign(error) * min(abs(error), self.args.push_lead / radius * taper)
            ahead = Rotation.from_rotvec(self.axis * lead).as_matrix()
            slide = np.clip(np.dot(p-point, radial), -self.args.max_slide, self.args.max_slide)
            goal = self.anchor + ahead @ (point + slide * radial - self.anchor)
            # Relieve pressure during brief gaps instead of chasing a lost
            # contact with increasing penetration. Sustained loss aborts below.
            relief = min(self.args.max_normal_command, lost * 0.0005)
            goal += (normal_offset - relief) * (ahead @ tangent) + height_offset * self.axis
            turn = np.clip((self.angle-contact_theta) * self.args.orientation_follow,
                           -np.deg2rad(self.args.max_orientation_deg), np.deg2rad(self.args.max_orientation_deg))
            target_r = Rotation.from_rotvec(self.axis * turn).as_matrix() @ contact_rotation
            _, _, touching = self.step(goal, target_r, allow_door=True)
            lost = 0 if touching else lost + 1
            if lost > round(1.5 / self.dt) and not self.closed:
                raise DemoFailure("lost_contact", "Active hand lost door contact for 1.5 seconds")
            history.append(self.angle)
            window = round(3 / self.dt)
            if len(history) > window and abs(history[-1]-history[-window]) < 0.005 and not self.closed:
                raise DemoFailure("stalled_closing", "Less than 0.005 rad progress in 3 seconds")
            if self.closed_count >= round(self.args.closed_dwell / self.dt):
                self.event("stage_closed_door", len(self.trace)-1)
                break
        else:
            raise DemoFailure("stalled_closing", "Closing step limit exceeded")

        self.phase = "release"
        # Hold the measured pose briefly to remove residual target penetration.
        p, r = self.tcp_pose()
        for _ in range(round(0.25 / self.dt)):
            self.step(p, r, allow_door=True, monitor_closed=True)
        _, _, tangent, _ = self.geometry()
        self.move(p - self.args.retreat_distance*tangent, r, "retreat", allow_door=True, monitor_closed=True)
        contacts, _ = door_contacts(self.model, self.data, self.door_bodies, self.hand_bodies)
        if contacts:
            raise DemoFailure("failed_release", "Hand still touching door after retreat")
        self.event("stage_released_door", len(self.trace)-1)

        home = self.base_pose() @ self.home
        # A straight Cartesian return can cross an IK singularity. Build a smooth
        # reachable TCP path using FK of interpolated arm postures in scratch data.
        # The real simulator still receives only the usual Cartesian commands.
        # Return through the raised clearance posture before lowering; a direct
        # joint interpolation can swing the fingers back through the panel.
        self.posture_path(safe_lift_qpos, "stow", monitor_closed=True)
        if raised_qpos is not None:
            self.posture_path(raised_qpos, "stow", monitor_closed=True)
        self.posture_path(self.home_qpos, "stow", monitor_closed=True)
        self.move(home[:3, 3], home[:3, :3], "stow", monitor_closed=True)
        self.event("stowed", len(self.trace)-1)
        self.phase = "hold"
        for _ in range(round(self.args.hold_seconds / self.dt)):
            self.step(home[:3, 3], home[:3, :3], monitor_closed=True)
        p, r = self.tcp_pose()
        if np.linalg.norm(p-home[:3, 3]) > 0.02 or Rotation.from_matrix(home[:3, :3].T @ r).magnitude() > np.deg2rad(8):
            raise DemoFailure("failed_stow", "Stow did not remain within 2 cm / 8 degrees")
        if not self.closed or abs(self.data.qvel[self.vadr]) >= 0.04:
            raise DemoFailure("rebound", "Door did not remain stably closed through the final hold")
        return True

    def save_trace(self, path):
        if self.trace:
            np.savez_compressed(path, **{key: np.asarray([row[key] for row in self.trace]) for key in self.trace[0]})
