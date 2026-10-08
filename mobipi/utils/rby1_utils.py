"""
RB-Y1 specific helpers for the Mobi-pi mobilization pipeline.

Like Tiago and PandaOmron, the RB-Y1 moves on a virtual base
(`RBY1MobileBase`): `mobilebase0_*` forward/side slide joints and a yaw hinge
about the wheel axle midpoint, driven by the stock JOINT_VELOCITY base
controller. The original Mobi-pi utilities assume PandaOmron's 12-D action and
base frame conventions, while the RB-Y1 uses a WHOLE_BODY_IK composite
controller, so base pose access, camera extrinsics, initial nav placement, and
navigation are re-implemented here.
"""
import math
import numpy as np
import robosuite.utils.transform_utils as T

from robocasa.utils.env_utils import grid_based_pose_sampling

from mobipi.utils import nav_utils
from mobipi.utils.nav_utils import angle_wrap, check_path_collision, rrt_planner


ROOT_BODY = "robot0_base"  # stays at the env's robot placement; the base joints move the robot relative to it
BASE_SITE = "mobilebase0_center"  # base origin, moves with the base joints
FORWARD_JOINT = "mobilebase0_joint_mobile_forward"
SIDE_JOINT = "mobilebase0_joint_mobile_side"
YAW_JOINT = "mobilebase0_joint_mobile_yaw"
ROBOT_GEOM_PREFIXES = ("robot0_", "gripper0_")

# Same per-step limits as `nav_utils.move_to_pose` at 20 Hz control
MAX_DIST_PER_STEP = 0.5 / 20
MAX_ROT_PER_STEP = 1.0 / 20

# Closed-loop base navigation: speed limits (same as `nav_utils.move_to_pose`),
# P gains, and arrival tolerances
BASE_V_MAX = 0.5
BASE_W_MAX = 1.0
BASE_K_V = 1.5
BASE_K_W = 3.0
BASE_POS_TOL = 0.02
BASE_HEADING_TOL = 0.01


def get_base_vec(sim):
    """Returns the base pose as [x, y, yaw]."""
    pos = sim.data.get_site_xpos(BASE_SITE)
    yaw = T.mat2euler(sim.data.get_site_xmat(BASE_SITE))[2]
    return np.array([pos[0], pos[1], yaw])


def set_base_vec(sim, base_vec):
    """
    Teleports the base to [x, y, yaw] by setting the base joints.

    The slide joints translate the base in the root body frame, then the yaw
    hinge rotates it about its anchor `a`, so the base origin ends up at
    root + R_root (s + a - R_z(q_yaw) a).
    """
    model = sim.model
    root_pos = sim.data.get_body_xpos(ROOT_BODY)
    root_yaw = T.mat2euler(sim.data.get_body_xmat(ROOT_BODY))[2]
    q_yaw = angle_wrap(base_vec[2] - root_yaw)
    anchor = model.jnt_pos[model.joint_name2id(YAW_JOINT)][:2]
    rot_root = np.array([[np.cos(root_yaw), -np.sin(root_yaw)], [np.sin(root_yaw), np.cos(root_yaw)]])
    rot_yaw = np.array([[np.cos(q_yaw), -np.sin(q_yaw)], [np.sin(q_yaw), np.cos(q_yaw)]])
    slide = rot_root.T @ (np.asarray(base_vec[:2]) - root_pos[:2]) - anchor + rot_yaw @ anchor
    for joint, value in [
        (FORWARD_JOINT, slide @ model.jnt_axis[model.joint_name2id(FORWARD_JOINT)][:2]),
        (SIDE_JOINT, slide @ model.jnt_axis[model.joint_name2id(SIDE_JOINT)][:2]),
        (YAW_JOINT, q_yaw),
    ]:
        sim.data.set_joint_qpos(joint, value)
        sim.data.set_joint_qvel(joint, 0.0)
    sim.forward()


def get_robot_joint_qpos(env):
    robot = env.robots[0]
    return env.sim.data.qpos[robot._ref_joint_pos_indexes].copy()


def set_robot_joint_qpos(env, qpos):
    robot = env.robots[0]
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = qpos
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0


def compute_relative_cam_pose(sim, camera_name):
    """Camera pose relative to the chassis frame projected onto the floor (yaw only)."""
    x, y, yaw = get_base_vec(sim)
    base_rot = T.quat2mat(T.axisangle2quat(np.array([0, 0, yaw])))
    cam_pos = sim.data.get_camera_xpos(camera_name)
    cam_mat = sim.data.get_camera_xmat(camera_name).reshape(3, 3)
    rel_pos = base_rot.T @ (cam_pos - np.array([x, y, 0.0]))
    rel_mat = base_rot.T @ cam_mat
    return rel_pos, rel_mat


def render_image_with_robot_mask(sim, camera_name, image_height=224, image_width=224):
    """RB-Y1 version of `env_utils.render_image_with_robot_mask`."""
    rel_pos, rel_mat = compute_relative_cam_pose(sim, camera_name)
    image = sim.render(camera_name=camera_name, width=image_width, height=image_height)
    seg = sim.render(
        camera_name=camera_name,
        width=image_width,
        height=image_height,
        depth=False,
        segmentation=True,
    )
    robot_geom_ids = [
        i for i, name in enumerate(sim.model.geom_names)
        if name and name.startswith(ROBOT_GEOM_PREFIXES)
    ]
    mask = np.isin(seg[..., 1], robot_geom_ids).astype(np.uint8)
    return rel_pos, rel_mat, image[::-1], mask[::-1]


def render_cameras(sim, camera_names, image_size=224):
    """Renders upright RGB images, shape (V, H, W, 3), uint8."""
    return np.stack([
        sim.render(camera_name=c, width=image_size, height=image_size)[::-1]
        for c in camera_names
    ])


def robot_env_contacts(sim, ignore_prefixes=("floor",)):
    """Returns (robot_geom, other_geom) pairs that are in penetration."""
    contacts = []
    for i in range(sim.data.ncon):
        con = sim.data.contact[i]
        if con.dist >= 0:
            continue
        n1 = sim.model.geom_id2name(con.geom1) or ""
        n2 = sim.model.geom_id2name(con.geom2) or ""
        r1, r2 = n1.startswith(ROBOT_GEOM_PREFIXES), n2.startswith(ROBOT_GEOM_PREFIXES)
        if r1 == r2:
            continue
        other = n2 if r1 else n1
        if other.startswith(ignore_prefixes):
            continue
        contacts.append((n1, n2) if r1 else (n2, n1))
    return contacts


def sample_nav_init_pose(env, base_fixture_bounds_2d, floor_fixture_bounds_2d,
                         default_robot_pos, rng, robot_footprint=(0.6, 0.6),
                         k_col=None, max_tries=100):
    """
    Replacement for RoboCasa's `_place_robot_for_nav`, which relies on
    `mobilebase0_*` joints. Samples a pose from which the default pose is in
    view and teleports the RB-Y1 chassis there, resampling on collision.
    If `k_col` is given, poses it rejects are resampled too, since the
    navigation planner cannot start from them.
    """
    for _ in range(max_tries):
        pose = grid_based_pose_sampling(
            base_fixture_bounds_2d,
            floor_fixture_bounds_2d,
            np.asarray(default_robot_pos[:2]),
            rng,
            robot_size=np.asarray(robot_footprint),
            grid_resolution=0.2,
            orientation_resolution=np.pi / 6,
            fov=np.pi * 75 / 180,
        )
        if k_col is not None and not k_col.compute_score(None, None, np.array([pose]), numpy=True)[0]:
            continue
        set_base_vec(env.sim, pose)
        contacts = robot_env_contacts(env.sim)
        if len(contacts) == 0:
            return np.asarray(pose)
        print(f"[rby1_utils.py] Sampled init pose {np.round(pose, 2)} collides with {contacts[0]}; resampling")
    raise RuntimeError("Could not sample a collision-free initial nav pose.")


def hold_action(env, gripper_action=None):
    """
    Builds a WHOLE_BODY_IK action that holds the current arm/torso/head
    configuration with zero base velocity.
    """
    robot = env.robots[0]
    controller = robot.composite_controller
    sim = env.sim
    split = controller._whole_body_controller_action_split_indexes
    ik_parts = controller.composite_controller_specific_config["actuation_part_names"]
    ref_names = controller.composite_controller_specific_config["ref_name"]

    action = np.zeros(split[next(reversed(split))][1])
    for part, (start, end) in split.items():
        if part in ik_parts:
            ref = ref_names[ik_parts.index(part)] if len(ref_names) == len(ik_parts) \
                else f"gripper0_{part}_ft_frame"
            site_id = sim.model.site_name2id(ref)
            pos = sim.data.site_xpos[site_id]
            mat = sim.data.site_xmat[site_id].reshape(3, 3)
            action[start:end] = np.concatenate([pos, T.quat2axisangle(T.mat2quat(mat))])
        elif part in controller.part_controllers and part != "base" and not part.endswith("gripper"):
            joint_names = controller.part_controllers[part].joint_names
            action[start:end] = [sim.data.get_joint_qpos(n) for n in joint_names]
        elif part.endswith("gripper") and gripper_action is not None:
            action[start:end] = gripper_action[part]
    return action


def _diff_drive_poses(start, goal_xy, final_heading=None):
    """
    Discretizes a rotate -> drive straight -> (optionally) rotate maneuver.
    Drives backwards if that needs less rotation.
    """
    poses = []
    x, y, th = start
    delta = np.asarray(goal_xy) - np.array([x, y])
    dist = np.linalg.norm(delta)

    if dist > 1e-3:
        heading = math.atan2(delta[1], delta[0])
        direction = 1.0
        if abs(angle_wrap(heading - th)) > np.pi / 2:
            heading = angle_wrap(heading + np.pi)
            direction = -1.0
        # rotate in place
        dth = angle_wrap(heading - th)
        n_rot = int(np.ceil(abs(dth) / MAX_ROT_PER_STEP))
        for i in range(1, n_rot + 1):
            poses.append(np.array([x, y, angle_wrap(th + dth * i / n_rot)]))
        th = heading
        # drive straight
        n_lin = int(np.ceil(dist / MAX_DIST_PER_STEP))
        for i in range(1, n_lin + 1):
            poses.append(np.array([x + delta[0] * i / n_lin, y + delta[1] * i / n_lin, th]))
        x, y = goal_xy
        assert direction in (1.0, -1.0)

    if final_heading is not None:
        dth = angle_wrap(final_heading - th)
        n_rot = int(np.ceil(abs(dth) / MAX_ROT_PER_STEP))
        for i in range(1, n_rot + 1):
            poses.append(np.array([x, y, angle_wrap(th + dth * i / n_rot)]))
    return poses


class _BaseDriver:
    """
    Closed-loop unicycle control (no lateral motion, as for the real
    differential-drive robot) through the JOINT_VELOCITY base controller.

    The controller takes [forward, lateral, yaw] velocity in the robot frame,
    scaled to [-1, 1] by the actuator ranges, and rotates the translation into
    the root frame of the slide joints. Each joint's velocity servo loses
    frictionloss / kv to dry friction, so the root-frame velocity of every joint
    is compensated separately; compensating in the robot frame would bend the
    direction of motion whenever the heading differs from the root's.
    """

    def __init__(self, env, gripper_action, reset_joint_qpos, step_callback):
        self.env = env
        self.sim = env.sim
        self.gripper_action = gripper_action
        self.reset_joint_qpos = reset_joint_qpos
        self.step_callback = step_callback
        self.stuck = False

        model = self.sim.model
        robot = env.robots[0]
        controller = robot.composite_controller
        self.base_slice = slice(*controller._whole_body_controller_action_split_indexes["base"])
        self.base_controller = controller.part_controllers["base"]
        # Velocity limits and dry-friction velocity offsets of the slide joints
        # along the root x / y axes and of the yaw joint
        limits, friction = [], []
        for joint in (SIDE_JOINT, FORWARD_JOINT, YAW_JOINT):
            joint_id = model.joint_name2id(joint)
            act_id = next(a for a in range(model.nu) if model.actuator_trnid[a, 0] == joint_id)
            limits.append(model.actuator_ctrlrange[act_id, 1])
            friction.append(model.dof_frictionloss[model.jnt_dofadr[joint_id]] / model.actuator_gainprm[act_id, 0])
        assert np.allclose(model.jnt_axis[model.joint_name2id(SIDE_JOINT)], [1, 0, 0])
        assert np.allclose(model.jnt_axis[model.joint_name2id(FORWARD_JOINT)], [0, 1, 0])
        # the controller rotates the translation before scaling it per joint
        assert np.isclose(limits[0], limits[1]), limits
        self.v_limit, self.w_limit = limits[0], limits[2]
        self.xy_friction, self.w_friction = np.array(friction[:2]), friction[2]
        # the robot turns in place about the yaw joint anchor (the axle midpoint)
        self.axle_offset = float(model.jnt_pos[model.joint_name2id(YAW_JOINT)][0])

    def axle_xy(self, base_vec):
        """Axle midpoint for a base pose [x, y, yaw]."""
        x, y, th = base_vec
        return np.array([x + self.axle_offset * np.cos(th), y + self.axle_offset * np.sin(th)])

    def pose(self):
        """Current axle-midpoint pose [x, y, yaw]."""
        base_vec = get_base_vec(self.sim)
        return np.r_[self.axle_xy(base_vec), base_vec[2]]

    def go_to(self, goal_xy, final_heading=None):
        """Turns toward `goal_xy` (axle midpoint), drives there, then optionally turns in place."""
        x, y, th = self.pose()
        delta = np.asarray(goal_xy) - np.array([x, y])
        if np.linalg.norm(delta) > BASE_POS_TOL:
            heading = math.atan2(delta[1], delta[0])
            if abs(angle_wrap(heading - th)) > np.pi / 2:
                heading = angle_wrap(heading + np.pi)
            self.rotate_to(heading)
            self.drive_to(goal_xy)
        if final_heading is not None:
            self.rotate_to(final_heading)

    @staticmethod
    def _compensate(value, friction):
        """Servo velocity target that realizes `value` despite the joint's dry friction."""
        return np.where(np.abs(value) < 1e-6, 0.0, value + np.sign(value) * friction)

    def step(self, v, w):
        """Applies body velocity (v [m/s], w [rad/s]) for one control step."""
        # heading relative to the root frame, as the controller computes it
        theta = get_base_vec(self.sim)[2] - T.mat2euler(self.base_controller.init_ori)[2]
        c, s = np.cos(theta), np.sin(theta)
        u_x, u_y = self._compensate(v * np.array([c, s]), self.xy_friction)
        action = hold_action(self.env, self.gripper_action)
        action[self.base_slice] = np.clip([
            (u_x * c + u_y * s) / self.v_limit,
            (-u_x * s + u_y * c) / self.v_limit,
            self._compensate(w, self.w_friction) / self.w_limit,
        ], -1.0, 1.0)
        self.env.step(action)
        set_robot_joint_qpos(self.env, self.reset_joint_qpos)
        self.sim.forward()
        self.step_callback()

    def _timeout(self, amount, rate):
        """Max control steps for a maneuver: 3x the nominal duration plus slack."""
        return int(3 * abs(amount) / rate * self.env.control_freq) + 60

    def rotate_to(self, heading):
        err0 = angle_wrap(heading - self.pose()[2])
        for _ in range(self._timeout(err0, BASE_W_MAX)):
            err = angle_wrap(heading - self.pose()[2])
            if abs(err) < BASE_HEADING_TOL:
                break
            self.step(0.0, float(np.clip(BASE_K_W * err, -BASE_W_MAX, BASE_W_MAX)))
        else:
            self.stuck = True

    def drive_to(self, goal_xy):
        """Drives the axle midpoint straight to `goal_xy`, forwards or backwards, steering onto the line."""
        x, y, th = self.pose()
        delta = np.asarray(goal_xy) - np.array([x, y])
        direction = 1.0 if np.dot(delta, [np.cos(th), np.sin(th)]) >= 0 else -1.0
        best_dist, no_progress = np.inf, 0
        for _ in range(self._timeout(np.linalg.norm(delta), BASE_V_MAX)):
            x, y, th = self.pose()
            delta = np.asarray(goal_xy) - np.array([x, y])
            dist = np.linalg.norm(delta)
            along = direction * np.dot(delta, [np.cos(th), np.sin(th)])
            if dist < BASE_POS_TOL or along <= 0:
                break
            v = direction * min(BASE_K_V * along, BASE_V_MAX)
            w = 0.0
            if dist > 0.1:  # steering near the goal makes the robot circle it
                ref = math.atan2(delta[1], delta[0]) + (0 if direction > 0 else np.pi)
                w = float(np.clip(BASE_K_W * angle_wrap(ref - th), -BASE_W_MAX, BASE_W_MAX))
            self.step(v, w)
            if dist < best_dist - 1e-3:
                best_dist, no_progress = dist, 0
            else:
                no_progress += 1
                if no_progress >= 20:
                    self.stuck = True
                    break
        else:
            self.stuck = True


def move_to_pose(env, target_vec, k_col, floor_fixture_bounds_2d,
                 render_camera="robot0_head_camera", render_size=256,
                 settle_steps=20, gripper_action=None, use_rrt=True, mode="velocity"):
    """
    Navigates the RB-Y1 base to `target_vec` = [x, y, yaw].

    Plans like `nav_utils.move_to_pose` (straight line, RRT if blocked) and
    executes the plan as differential-drive maneuvers (rotate, drive straight,
    rotate; no lateral motion). With `mode="velocity"` the maneuvers are tracked
    in closed loop through base velocity commands. With `mode="kinematic"` the
    base is moved along the plan directly at `nav_utils.move_to_pose`'s
    speed limits, ignoring base dynamics. Penetrating robot-environment
    contacts are recorded at every step. After arrival the sim is stepped with
    a hold action to check the final pose is stable.
    """
    assert mode in ("velocity", "kinematic"), mode
    sim = env.sim
    target_vec = np.asarray(target_vec, dtype=float)
    init_vec = get_base_vec(sim)
    reset_joint_qpos = get_robot_joint_qpos(env)

    # restrict RRT sampling to the room
    xy_min = floor_fixture_bounds_2d.min(axis=0)
    xy_max = floor_fixture_bounds_2d.max(axis=0)
    nav_utils.X_MIN, nav_utils.X_MAX = xy_min[0], xy_max[0]
    nav_utils.Y_MIN, nav_utils.Y_MAX = xy_min[1], xy_max[1]

    used_rrt = False
    waypoints = [tuple(init_vec), tuple(target_vec)]
    if use_rrt and not check_path_collision(k_col, init_vec, target_vec, steps=100):
        print("[rby1_utils.py] Straight line path not collision free, using RRT...")
        waypoints = rrt_planner(k_col=k_col, start=tuple(init_vec), goal=tuple(target_vec))
        used_rrt = True
        # a failed RRT can return just [goal]
        if not np.allclose(waypoints[0][:2], init_vec[:2]):
            waypoints = [tuple(init_vec)] + list(waypoints)

    base_vec_history = [init_vec]
    collision_history = []
    images = []
    if render_camera is not None:
        images.append(render_cameras(sim, [render_camera], render_size)[0])

    def record_step():
        base_vec_history.append(get_base_vec(sim))
        collision_history.append(robot_env_contacts(sim))
        if render_camera is not None:
            images.append(render_cameras(sim, [render_camera], render_size)[0])

    stuck = False
    if mode == "kinematic":
        curr = init_vec.copy()
        for i, wp in enumerate(waypoints[1:]):
            is_last = i == len(waypoints) - 2
            for pose in _diff_drive_poses(curr, wp[:2], final_heading=target_vec[2] if is_last else None):
                set_base_vec(sim, pose)
                set_robot_joint_qpos(env, reset_joint_qpos)
                sim.forward()
                record_step()
                curr = pose
    else:
        driver = _BaseDriver(env, gripper_action, reset_joint_qpos, record_step)
        target_axle = driver.axle_xy(target_vec)
        for wp in waypoints[1:-1]:
            driver.go_to(wp[:2])
        driver.go_to(target_axle, final_heading=target_vec[2])
        # correct a residual offset once (turn toward it, drive, turn back)
        if np.linalg.norm(driver.pose()[:2] - target_axle) > 2 * BASE_POS_TOL:
            driver.go_to(target_axle, final_heading=target_vec[2])
        stuck = driver.stuck
    num_steps = len(base_vec_history) - 1

    arrived_vec = get_base_vec(sim)

    # let physics run with the robot holding still to check the pose is stable
    for _ in range(settle_steps):
        env.step(hold_action(env, gripper_action))
        if render_camera is not None:
            images.append(render_cameras(sim, [render_camera], render_size)[0])
    settled_vec = get_base_vec(sim)

    path_xy = np.array(base_vec_history)[:, :2]
    num_collision_steps = int(sum(len(c) > 0 for c in collision_history))
    colliding_geoms = sorted({pair[1] for c in collision_history for pair in c})

    info = dict(
        base_vec_history=np.array(base_vec_history),
        waypoints=np.array(waypoints),
        used_rrt=used_rrt,
        mode=mode,
        stuck=stuck,
        num_steps=num_steps,
        path_length=float(np.sum(np.linalg.norm(np.diff(path_xy, axis=0), axis=1))),
        num_collision_steps=num_collision_steps,
        colliding_geoms=colliding_geoms,
        final_contacts=robot_env_contacts(sim),
        arrived_vec=arrived_vec,
        settled_vec=settled_vec,
        settle_drift=np.abs(np.r_[settled_vec[:2] - arrived_vec[:2],
                                  angle_wrap(settled_vec[2] - arrived_vec[2])]),
        images=images,
    )
    err = np.abs(np.r_[settled_vec[:2] - target_vec[:2], angle_wrap(settled_vec[2] - target_vec[2])])
    print(
        f"[rby1_utils.py] Navigated to {settled_vec.round(3)} (target {target_vec.round(3)}, "
        f"error {err.round(3)}); {mode}, {num_steps} steps, {num_collision_steps} with contacts"
        + (", got stuck" if stuck else "")
    )
    return info
