#!/usr/bin/env python3

import argparse
import json
import os

import mujoco
import numpy as np
import robosuite
import robocasa  # registers RoboCasa environments
from scipy.spatial.transform import Rotation
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper

try:
    from robocasa.utils.env_utils import compute_robot_base_placement_pose
except Exception:
    compute_robot_base_placement_pose = None


from pathlib import Path

DEFAULT_CONTROLLER = str(
    Path(robosuite.__file__).resolve().parent
    / "controllers/config/robots/default_rby1_whole_body_ik.json"
)

# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def raw_model(sim):
    return sim.model._model if hasattr(sim.model, "_model") else sim.model


def raw_data(sim):
    return sim.data._data if hasattr(sim.data, "_data") else sim.data


def forward(sim):
    try:
        sim.forward()
    except Exception:
        mujoco.mj_forward(raw_model(sim), raw_data(sim))


def unwrap_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise RuntimeError(f"Cannot normalize vector {v}")
    return v / n


def smoothstep(t):
    return 3.0 * t**2 - 2.0 * t**3


def lerp_points(a, b, n):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.array([a + smoothstep(t) * (b - a) for t in np.linspace(0, 1, max(n, 2))])


def site_pose(sim, name):
    m, d = raw_model(sim), raw_data(sim)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f"Site not found: {name}")
    pos = np.asarray(d.site_xpos[sid], dtype=float).copy()
    mat = np.asarray(d.site_xmat[sid], dtype=float).reshape(3, 3).copy()
    return pos, mat


def rotation_error(R_actual, R_target):
    return np.linalg.norm(Rotation.from_matrix(R_target @ R_actual.T).as_rotvec())


def make_T(pos, rot):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T


def pose6(position, rotation_matrix):
    return np.concatenate([position, Rotation.from_matrix(rotation_matrix).as_rotvec()])


def interpolate_rotation(R0, R1, t):
    relative = R1 @ R0.T
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    return Rotation.from_rotvec(t * rotvec).as_matrix() @ R0


def other_arm(arm):
    return "right" if arm == "left" else "left"


def arm_site(arm):
    return f"gripper0_{arm}_grip_site"


def arm_wrist_site(arm):
    return f"gripper0_{arm}_ft_frame"


def arm_joint_names(arm):
    return [f"robot0_{arm}_arm_{i}" for i in range(7)]


# -----------------------------------------------------------------------------
# Environment / fixture discovery
# -----------------------------------------------------------------------------

def discover_target_fixture(env):
    ep_meta = env.get_ep_meta()
    print("\n===== Fixture discovery =====")
    print("instruction:", ep_meta.get("lang", ""))
    refs = getattr(env, "fixture_refs", {})
    print("fixture_refs keys:", list(refs.keys()))

    if "door_fxtr" in refs:
        fxtr = refs["door_fxtr"]
        print("Selected fixture_refs['door_fxtr']")
        print("type:", type(fxtr).__name__)
        print("name:", getattr(fxtr, "name", None))
        return fxtr

    for key, value in refs.items():
        if "door" in str(key).lower():
            print(f"Selected fallback fixture ref: {key}")
            print("type:", type(value).__name__)
            print("name:", getattr(value, "name", None))
            return value

    raise RuntimeError(f"Could not find a target door fixture. Available refs: {list(refs.keys())}")


def discover_hinge(base_env, fxtr):
    sim = base_env.sim
    m, d = raw_model(sim), raw_data(sim)
    qpos0 = np.asarray(d.qpos).copy()
    qvel0 = np.asarray(d.qvel).copy()
    fixture_name = str(getattr(fxtr, "name", "")).lower()

    fixture_joint_names = []
    for attr in ["joints", "joint_names", "door_joints"]:
        if not hasattr(fxtr, attr):
            continue
        try:
            value = getattr(fxtr, attr)
            if isinstance(value, str):
                fixture_joint_names.append(value)
            elif value is not None:
                fixture_joint_names.extend(list(value))
        except Exception:
            pass
    fixture_joint_names = [str(x) for x in fixture_joint_names]

    candidates = []
    for jid in range(m.njnt):
        if int(m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            continue
        lname = name.lower()
        score = 0
        if name in fixture_joint_names:
            score += 100
        if fixture_name and fixture_name in lname:
            score += 50
        if "door" in lname:
            score += 10
        if "hinge" in lname:
            score += 10
        if score > 0:
            candidates.append((score, jid, name))

    if not candidates:
        for jid in range(m.njnt):
            if int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
                candidates.append((0, jid, name))

    candidates.sort(reverse=True)
    print("\n===== Hinge candidates =====")
    for score, jid, name in candidates:
        adr = int(m.jnt_qposadr[jid])
        print(f"score={score:3d} id={jid:3d} name={name} q={float(d.qpos[adr]):.4f} range={np.asarray(m.jnt_range[jid])}")

    best = None
    for score, jid, name in candidates:
        adr = int(m.jnt_qposadr[jid])
        theta_start = float(qpos0[adr])
        lo, hi = float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])
        valid_tests = []
        for theta in [0.0, lo, hi]:
            if lo - 1e-6 <= theta <= hi + 1e-6 and not any(abs(theta - x) < 1e-6 for x in valid_tests):
                valid_tests.append(theta)
        for theta_test in valid_tests:
            d.qpos[:] = qpos0
            d.qvel[:] = qvel0
            d.qpos[adr] = theta_test
            forward(sim)
            try:
                success = bool(base_env._check_success())
            except Exception:
                success = False
            if success:
                best = (jid, name, theta_start, theta_test)
                break
        if best is not None:
            break

    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    forward(sim)

    if best is None:
        raise RuntimeError("Could not determine which hinge value closes the task door.")

    jid, joint_name, theta_start, theta_closed = best
    hinge_pos = np.asarray(d.xanchor[jid], dtype=float).copy()
    hinge_axis = normalize(np.asarray(d.xaxis[jid], dtype=float))

    print("\n===== Selected door hinge =====")
    print("joint:       ", joint_name)
    print("theta start: ", theta_start)
    print("theta closed:", theta_closed)
    print("delta:       ", theta_closed - theta_start)
    print("hinge pos:   ", np.round(hinge_pos, 4))
    print("hinge axis:  ", np.round(hinge_axis, 4))
    return jid, joint_name, theta_start, theta_closed, hinge_pos, hinge_axis


# -----------------------------------------------------------------------------
# Door-panel geometry. This intentionally does NOT require a handle.
# -----------------------------------------------------------------------------

def body_descendants(model, root_body_id):
    result = set()
    for bid in range(model.nbody):
        cur = bid
        while cur > 0:
            if cur == root_body_id:
                result.add(bid)
                break
            cur = int(model.body_parentid[cur])
    result.add(int(root_body_id))
    return result


def door_body_and_geom_ids(base_env, hinge_id):
    m = raw_model(base_env.sim)
    root_body = int(m.jnt_bodyid[hinge_id])
    body_ids = body_descendants(m, root_body)
    geom_ids = []
    preferred = []

    for gid in range(m.ngeom):
        bid = int(m.geom_bodyid[gid])
        if bid not in body_ids:
            continue
        geom_ids.append(gid)
        gname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
        bname = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
        if not any(tok in f"{gname} {bname}" for tok in ("handle", "knob", "grip")):
            preferred.append(gid)

    if preferred:
        geom_ids = preferred

    # Door panels are commonly represented by boxes. Prefer them when available
    # because their corners give a much tighter width/height estimate than a
    # mesh bounding sphere. Fall back to all door geoms for mesh-only assets.
    box_geoms = [gid for gid in geom_ids if int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX)]
    if box_geoms:
        geom_ids = box_geoms

    if not geom_ids:
        raise RuntimeError("No MuJoCo geoms found on the selected door body.")
    return root_body, body_ids, geom_ids


def geom_sample_points(base_env, gid, radial_dir, hinge_axis):
    m, d = raw_model(base_env.sim), raw_data(base_env.sim)
    center = np.asarray(d.geom_xpos[gid], dtype=float).copy()
    gtype = int(m.geom_type[gid])

    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        R = np.asarray(d.geom_xmat[gid], dtype=float).reshape(3, 3)
        size = np.asarray(m.geom_size[gid][:3], dtype=float)
        pts = []
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    pts.append(center + R @ (size * np.array([sx, sy, sz])))
        return pts

    rbound = float(m.geom_rbound[gid])
    if not np.isfinite(rbound) or rbound <= 0:
        rbound = 0.02
    return [
        center,
        center + rbound * radial_dir,
        center - rbound * radial_dir,
        center + rbound * hinge_axis,
        center - rbound * hinge_axis,
    ]


def discover_door_push_point(base_env, hinge_id, hinge_pos, hinge_axis, radius_fraction=0.72, height_fraction=0.50):
    m, d = raw_model(base_env.sim), raw_data(base_env.sim)
    root_body, body_ids, geom_ids = door_body_and_geom_ids(base_env, hinge_id)

    best_vec, best_norm = None, -1.0
    for gid in geom_ids:
        c = np.asarray(d.geom_xpos[gid], dtype=float)
        v = c - hinge_pos
        v = v - np.dot(v, hinge_axis) * hinge_axis
        n = np.linalg.norm(v)
        if n > best_norm:
            best_vec, best_norm = v, n

    if best_vec is None or best_norm < 1e-4:
        raise RuntimeError("Could not infer the radial direction of the moving door panel.")
    radial_dir = normalize(best_vec)

    samples = []
    for gid in geom_ids:
        samples.extend(geom_sample_points(base_env, gid, radial_dir, hinge_axis))
    samples = np.asarray(samples, dtype=float)

    rel = samples - hinge_pos
    radial_values = rel @ radial_dir
    height_values = rel @ hinge_axis
    positive_radial = radial_values[radial_values > 0.0]
    if len(positive_radial) == 0:
        radial_dir *= -1.0
        radial_values = rel @ radial_dir
        positive_radial = radial_values[radial_values > 0.0]
    if len(positive_radial) == 0:
        raise RuntimeError("Could not determine the outward side of the door panel.")

    r_max = float(np.max(positive_radial))
    r_target = float(np.clip(radius_fraction, 0.15, 0.95)) * r_max
    h_min, h_max = float(np.min(height_values)), float(np.max(height_values))
    h_target = h_min + float(np.clip(height_fraction, 0.05, 0.95)) * (h_max - h_min)
    push_point = hinge_pos + r_target * radial_dir + h_target * hinge_axis

    root_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root_body)
    geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) for gid in geom_ids]
    print("\n===== Door-panel geometry =====")
    print("door body:      ", root_name)
    print("door geoms:     ", geom_names)
    print("radial dir:     ", np.round(radial_dir, 4))
    print("estimated width:", round(r_max, 4))
    print("height range:   ", np.round([h_min, h_max], 4))
    print("push point:     ", np.round(push_point, 4))
    print("radius fraction:", radius_fraction)
    print("height fraction:", height_fraction)
    return push_point, radial_dir, body_ids, geom_ids


# -----------------------------------------------------------------------------
# Door motion and tool orientation
# -----------------------------------------------------------------------------

def door_point(hinge_pos, hinge_axis, initial_point, theta, theta_start):
    delta = theta - theta_start
    return hinge_pos + Rotation.from_rotvec(hinge_axis * delta).apply(initial_point - hinge_pos)


def closing_tangent(hinge_pos, hinge_axis, point, theta_delta_sign):
    tangent = np.cross(hinge_axis, point - hinge_pos)
    tangent *= theta_delta_sign
    return normalize(tangent)


def force_tangent_direction(tangent, push_x_direction):
    if push_x_direction != 0 and np.sign(tangent[0]) != push_x_direction:
        tangent = -tangent
    return tangent


def door_facing_rotation(hinge_axis, door_normal, T_tcp_wrist):
    up = normalize(hinge_axis)
    normal = normalize(door_normal)
    sign = np.sign(T_tcp_wrist[2, 3])
    if abs(sign) < 1e-6:
        sign = 1.0
    z_world = -sign * normal
    y_world = up - np.dot(up, z_world) * z_world
    y_world = normalize(y_world)
    x_world = normalize(np.cross(y_world, z_world))
    return np.column_stack([x_world, y_world, z_world])


# -----------------------------------------------------------------------------
# Robot commands, generalized to left or right arm
# -----------------------------------------------------------------------------

def set_ik_q0_from_current_pose(base_env, robot):
    if not isinstance(robot.composite_controller, WholeBody):
        return
    ik = robot.composite_controller.joint_action_policy
    ik.q0 = np.array([float(base_env.sim.data.get_joint_qpos(name)) for name in ik.joint_names])
    print("\nIK q0:", np.round(ik.q0, 3))


def tcp_to_wrist_target(tcp_pos, tcp_rot, T_tcp_wrist):
    T_world_wrist = make_T(tcp_pos, tcp_rot) @ T_tcp_wrist
    return pose6(T_world_wrist[:3, 3], T_world_wrist[:3, :3])


def create_robot_action(robot, active_arm, active_target, inactive_hold_q, inactive_hold_target, gripper_value=1.0):
    inactive_arm = other_arm(active_arm)

    def with_grippers(action_dict):
        for arm in robot.arms:
            dof = robot.gripper[arm].dof
            if dof > 0:
                action_dict[f"{arm}_gripper"] = np.repeat([gripper_value], dof)
        return action_dict

    joint_hold = with_grippers({active_arm: active_target, inactive_arm: inactive_hold_q})
    try:
        return robot.create_action_vector(joint_hold)
    except Exception as first_error:
        pose_hold = with_grippers({active_arm: active_target, inactive_arm: inactive_hold_target})
        try:
            return robot.create_action_vector(pose_hold)
        except Exception as second_error:
            raise RuntimeError(
                f"Controller/action mismatch for active arm '{active_arm}'. "
                f"Use a controller config that accepts an absolute IK target for {active_arm} and can hold {inactive_arm}. "
                f"Joint-hold error: {first_error}; pose-hold error: {second_error}"
            ) from second_error


def execute_target(env, base_env, robot, active_arm, active_tcp_pos, active_tcp_rot,
                   inactive_hold_q, inactive_hold_target, active_tcp_to_wrist,
                   gripper_value, repeats=1):
    active_target = tcp_to_wrist_target(active_tcp_pos, active_tcp_rot, active_tcp_to_wrist)
    for _ in range(repeats):
        action = create_robot_action(
            robot, active_arm, active_target, inactive_hold_q,
            inactive_hold_target, gripper_value,
        )
        env.step(action)
        if not np.all(np.isfinite(raw_data(base_env.sim).qpos)):
            raise RuntimeError("Robot state became NaN / Inf.")
    actual_tcp, _ = site_pose(base_env.sim, arm_site(active_arm))
    return actual_tcp, float(np.linalg.norm(actual_tcp - active_tcp_pos))


def door_surface_contacts(base_env, door_body_ids, push_point=None, radius=None):
    m, d = raw_model(base_env.sim), raw_data(base_env.sim)
    robot_tokens = ("robot0", "rby1", "gripper", "left_hand", "right_hand", "left_gripper", "right_gripper")
    contacts = []

    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        n1 = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or "").lower()
        n2 = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or "").lower()
        bn1 = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1) or "").lower()
        bn2 = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2) or "").lower()
        robot1 = any(tok in f"{n1} {bn1}" for tok in robot_tokens)
        robot2 = any(tok in f"{n2} {bn2}" for tok in robot_tokens)
        door1, door2 = b1 in door_body_ids, b2 in door_body_ids
        if not ((robot1 and door2) or (robot2 and door1)):
            continue
        p = np.asarray(c.pos, dtype=float)
        dist = None if push_point is None else float(np.linalg.norm(p - push_point))
        if radius is not None and dist is not None and dist > radius:
            continue
        name1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g1) or ""
        name2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g2) or ""
        contacts.append((name1, name2, dist))
    return contacts


# -----------------------------------------------------------------------------
# Dynamic robot placement
# -----------------------------------------------------------------------------

def make_config(args, controller_config, renderer, forced_placement=None):
    config = {
        "env_name": args.environment,
        "robots": "RBY1",
        "controller_configs": controller_config,
        "layout_ids": [args.layout],
        "style_ids": [args.style],
        "has_renderer": renderer,
        "has_offscreen_renderer": False,
        "render_camera": None,
        "ignore_done": True,
        "use_camera_obs": False,
        "control_freq": args.control_freq,
        "renderer": "mjviewer",
        "translucent_robot": True,
        "initialization_noise": None,
        "randomize_base_init_pose": None,
        "randomize_cameras": False,
        "seed": args.seed,
    }
    if forced_placement is not None:
        pos, ori = forced_placement
        config["force_robot_placement"] = (np.asarray(pos, dtype=float).tolist(), np.asarray(ori, dtype=float).tolist())
    return config


def create_env(args, controller_config, renderer, forced_placement=None, record=False):
    np.random.seed(args.seed)
    env = robosuite.make(**make_config(args, controller_config, renderer, forced_placement))
    if record and args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)
        env = DataCollectionWrapper(env, args.record_dir)
        print("Recording raw trajectory to:", args.record_dir)
    np.random.seed(args.seed)
    env.reset()
    return env


def fallback_door_spawn(push_point, tangent, args):
    tangent_xy = np.asarray(tangent[:2], dtype=float)
    if np.linalg.norm(tangent_xy) < 1e-4:
        raise RuntimeError("Closing tangent has almost no horizontal component; automatic floor placement is not valid for this door.")
    away = -normalize(tangent_xy)
    base_xy = np.asarray(push_point[:2], dtype=float) + args.spawn_distance * away
    lateral = np.array([-away[1], away[0]])
    base_xy += args.spawn_lateral * lateral
    yaw = np.arctan2(push_point[1] - base_xy[1], push_point[0] - base_xy[0])
    return np.array([base_xy[0], base_xy[1], args.robot_z]), np.array([0.0, 0.0, yaw])


def compute_spawn_pose(base_env, fxtr, push_point, tangent, args):
    if args.spawn_mode == "fixed":
        return (
            np.array([args.robot_x, args.robot_y, args.robot_z], dtype=float),
            np.array([0.0, 0.0, args.robot_yaw], dtype=float),
        )

    official = None
    if compute_robot_base_placement_pose is not None:
        try:
            official = compute_robot_base_placement_pose(
                base_env, fxtr, offset=[args.spawn_offset_x, args.spawn_offset_y]
            )
            official = (
                np.asarray(official[0], dtype=float),
                np.asarray(official[1], dtype=float),
            )
            print("\nRoboCasa base suggestion:", np.round(official[0], 3), np.round(official[1], 3))
        except Exception as e:
            print("RoboCasa base-placement helper failed:", e)

    # Preferred general solution: trust RoboCasa fixture-relative placement
    if args.spawn_mode in ("robocasa", "hybrid") and official is not None:
        pos, ori = official
        pos = pos.copy()
        ori = ori.copy()

        if args.face_push_point:
            ori[2] = np.arctan2(
                push_point[1] - pos[1],
                push_point[0] - pos[0],
            )

        return pos, ori

    # Geometry fallback only if RoboCasa placement is unavailable
    tangent_xy = np.asarray(tangent[:2], dtype=float)
    if np.linalg.norm(tangent_xy) < 1e-4:
        raise RuntimeError("Could not infer horizontal door direction.")

    away = -normalize(tangent_xy)
    base_xy = np.asarray(push_point[:2], dtype=float) + args.spawn_distance * away

    lateral = np.array([-away[1], away[0]])
    base_xy += args.spawn_lateral * lateral

    yaw = np.arctan2(
        push_point[1] - base_xy[1],
        push_point[0] - base_xy[0],
    )

    return (
        np.array([base_xy[0], base_xy[1], args.robot_z]),
        np.array([0.0, 0.0, yaw]),
    )


def probe_scene(args, controller_config):
    env = create_env(args, controller_config, renderer=False, forced_placement=None, record=False)
    base_env = unwrap_env(env)
    fxtr = discover_target_fixture(base_env)
    hinge_id, hinge_name, theta_start, theta_closed, hinge_pos, hinge_axis = discover_hinge(base_env, fxtr)
    push_start, radial_dir, door_body_ids, _ = discover_door_push_point(
        base_env, hinge_id, hinge_pos, hinge_axis,
        args.push_radius_fraction, args.push_height_fraction,
    )
    direction_sign = np.sign(theta_closed - theta_start)
    tangent0 = closing_tangent(hinge_pos, hinge_axis, push_start, direction_sign)
    tangent0 = force_tangent_direction(tangent0, args.push_x_direction)
    spawn_pos, spawn_ori = compute_spawn_pose(base_env, fxtr, push_start, tangent0, args)

    signature = {
        "fixture": str(getattr(fxtr, "name", "")),
        "hinge": hinge_name,
        "push_point": push_start.copy(),
    }
    env.close()

    print("\n===== Planned robot spawn =====")
    print("mode: ", args.spawn_mode)
    print("pos:  ", np.round(spawn_pos, 4))
    print("rpy:  ", np.round(spawn_ori, 4))
    print("door distance:", round(float(np.linalg.norm(spawn_pos[:2] - push_start[:2])), 4))
    return spawn_pos, spawn_ori, signature


# -----------------------------------------------------------------------------
# Main trajectory
# -----------------------------------------------------------------------------

def run(args):
    controller_path = args.controller_left if args.arm == "left" and args.controller_left else args.controller
    controller_path = args.controller_right if args.arm == "right" and args.controller_right else controller_path
    controller_config = load_composite_controller_config(controller=controller_path, robot="RBY1")

    if args.spawn_mode == "fixed":
        spawn_pos = np.array([args.robot_x, args.robot_y, args.robot_z], dtype=float)
        spawn_ori = np.array([0.0, 0.0, args.robot_yaw], dtype=float)
        probe_signature = None
    else:
        spawn_pos, spawn_ori, probe_signature = probe_scene(args, controller_config)

    if args.probe_only:
        print("\nProbe complete; not executing the manipulation.")
        return True

    env = create_env(
        args, controller_config, renderer=True,
        forced_placement=(spawn_pos, spawn_ori), record=True,
    )
    base_env = unwrap_env(env)
    robot = base_env.robots[0]
    set_ik_q0_from_current_pose(base_env, robot)

    fxtr = discover_target_fixture(base_env)
    hinge_id, hinge_name, theta_start, theta_closed, hinge_pos, hinge_axis = discover_hinge(base_env, fxtr)
    delta_total = theta_closed - theta_start
    if abs(delta_total) < 1e-4:
        raise RuntimeError("Door is already essentially closed.")
    direction_sign = np.sign(delta_total)

    door_push_start, radial_dir, door_body_ids, door_geom_ids = discover_door_push_point(
        base_env, hinge_id, hinge_pos, hinge_axis,
        args.push_radius_fraction, args.push_height_fraction,
    )

    if probe_signature is not None:
        mismatch = float(np.linalg.norm(door_push_start - probe_signature["push_point"]))
        if mismatch > args.max_probe_mismatch:
            raise RuntimeError(
                f"Probe/final scene mismatch is {mismatch:.3f} m. "
                "The base pose was computed for a different sampled scene; rerun with the same seed or increase --max-probe-mismatch only if this is expected."
            )

    tangent0 = closing_tangent(hinge_pos, hinge_axis, door_push_start, direction_sign)
    tangent0 = force_tangent_direction(tangent0, args.push_x_direction)
    if np.linalg.norm(tangent0[:2]) < 0.20:
        raise RuntimeError("Selected door hinge produces a mostly vertical push. This script currently supports floor-based pushing of mostly vertical door hinges.")

    active_arm = args.arm
    inactive_arm = other_arm(active_arm)
    active_site, active_wrist_site = arm_site(active_arm), arm_wrist_site(active_arm)
    inactive_site, inactive_wrist_site = arm_site(inactive_arm), arm_wrist_site(inactive_arm)

    active_start_pos, active_start_rot = site_pose(base_env.sim, active_site)
    active_wrist_pos, active_wrist_rot = site_pose(base_env.sim, active_wrist_site)
    inactive_start_pos, inactive_start_rot = site_pose(base_env.sim, inactive_site)
    inactive_wrist_pos, inactive_wrist_rot = site_pose(base_env.sim, inactive_wrist_site)

    T_active_tcp_wrist = np.linalg.inv(make_T(active_start_pos, active_start_rot)) @ make_T(active_wrist_pos, active_wrist_rot)
    T_inactive_tcp_wrist = np.linalg.inv(make_T(inactive_start_pos, inactive_start_rot)) @ make_T(inactive_wrist_pos, inactive_wrist_rot)
    inactive_hold_q = np.array([float(base_env.sim.data.get_joint_qpos(name)) for name in arm_joint_names(inactive_arm)])
    inactive_hold_target = tcp_to_wrist_target(inactive_start_pos, inactive_start_rot, T_inactive_tcp_wrist)

    print("\n===== Generalized setup =====")
    print("layout/style/seed:", args.layout, args.style, args.seed)
    print("fixture:          ", getattr(fxtr, "name", None), type(fxtr).__name__)
    print("active arm:       ", active_arm)
    print("inactive arm:     ", inactive_arm)
    print("controller:       ", controller_path)
    print("base pose:        ", np.round(spawn_pos, 4), np.round(spawn_ori, 4))
    print("push point:       ", np.round(door_push_start, 4))
    print("closing tangent:  ", np.round(tangent0, 4))

    contact_rot = door_facing_rotation(hinge_axis, tangent0, T_active_tcp_wrist)
    pre_contact = door_push_start - (args.standoff + args.contact_clearance) * tangent0
    far_pre_contact = door_push_start - (args.standoff + args.contact_clearance + args.extra_approach_clearance) * tangent0
    lift_pose = far_pre_contact.copy()
    lift_pose[2] = door_push_start[2] + args.via_z
    pre_contact_high = pre_contact.copy()
    pre_contact_high[2] = lift_pose[2]

    trajectory = []
    points = lerp_points(active_start_pos, far_pre_contact, args.approach_points // 3)
    for i, p in enumerate(points):
        t = i / max(len(points) - 1, 1)
        trajectory.append((p, interpolate_rotation(active_start_rot, contact_rot, t)))
    for p in lerp_points(far_pre_contact, lift_pose, args.approach_points // 3):
        trajectory.append((p, contact_rot))
    for p in lerp_points(lift_pose, pre_contact_high, args.approach_points // 3):
        trajectory.append((p, contact_rot))
    for p in lerp_points(pre_contact_high, pre_contact, args.approach_points // 4):
        trajectory.append((p, contact_rot))

    print("\n===== Executing approach =====")
    print("waypoints:", len(trajectory))
    for _ in range(args.settle_steps):
        execute_target(
            env, base_env, robot, active_arm,
            active_start_pos, active_start_rot,
            inactive_hold_q, inactive_hold_target,
            T_active_tcp_wrist, args.gripper,
        )

    last_target, last_rot = active_start_pos.copy(), active_start_rot.copy()
    for i, (target, target_rot) in enumerate(trajectory):
        actual, err = execute_target(
            env, base_env, robot, active_arm,
            target, target_rot, inactive_hold_q, inactive_hold_target,
            T_active_tcp_wrist, args.gripper, repeats=args.steps_per_waypoint,
        )
        last_target, last_rot = target.copy(), target_rot.copy()
        if i % 10 == 0:
            print(f"approach {i:3d}/{len(trajectory):3d} target={np.round(target,3)} actual={np.round(actual,3)} err={err:.3f}")

    m, d = raw_model(base_env.sim), raw_data(base_env.sim)
    door_qadr = int(m.jnt_qposadr[hinge_id])

    print("\n===== Aligning with door surface =====")
    for step in range(100):
        actual, err = execute_target(
            env, base_env, robot, active_arm,
            pre_contact, contact_rot, inactive_hold_q, inactive_hold_target,
            T_active_tcp_wrist, args.gripper,
        )
        _, actual_rot = site_pose(base_env.sim, active_site)
        ang_err = rotation_error(actual_rot, contact_rot)
        if step % 10 == 0:
            print(f"align {step:3d} pos_err={err:.3f} rot_err={np.degrees(ang_err):.1f} deg")
        if err < args.align_position_tolerance and ang_err < np.deg2rad(args.align_angle_tolerance_deg):
            break

    print("\n===== Acquiring physical DOOR contact =====")
    clearance = args.contact_clearance
    contact_found = False
    for step in range(args.contact_search_steps):
        theta = float(d.qpos[door_qadr])
        remaining = theta_closed - theta
        push_point = door_point(hinge_pos, hinge_axis, door_push_start, theta, theta_start)
        tangent = closing_tangent(hinge_pos, hinge_axis, push_point, np.sign(remaining))
        tangent = force_tangent_direction(tangent, args.push_x_direction)
        target_rot = door_facing_rotation(hinge_axis, tangent, T_active_tcp_wrist)
        target = push_point - clearance * tangent
        target[2] += args.contact_z_offset

        actual, err = execute_target(
            env, base_env, robot, active_arm,
            target, target_rot, inactive_hold_q, inactive_hold_target,
            T_active_tcp_wrist, args.gripper, repeats=2,
        )
        contacts = door_surface_contacts(base_env, door_body_ids, push_point, args.contact_radius)

        if step % 5 == 0:
            print(
                f"surface-search {step:3d} clearance={clearance:.3f} "
                f"target={np.round(target,3)} actual={np.round(actual,3)} err={err:.3f} contacts={contacts[:3]}"
            )

        if contacts:
            contact_tcp_pos, push_rot_ref = site_pose(base_env.sim, active_site)
            theta_contact_ref = float(d.qpos[door_qadr])
            surface_offset_ref = contact_tcp_pos - push_point
            print("\nDOOR CONTACT at clearance:", clearance)
            print("contacts:", contacts[:5])
            print("contact TCP:", np.round(contact_tcp_pos, 3))
            print("surface offset:", np.round(surface_offset_ref, 3))
            print("contact door angle:", theta_contact_ref)
            contact_found = True
            break

        if step > 0 and step % args.contact_search_interval == 0:
            clearance = max(clearance - args.contact_search_step, -args.max_contact_penetration)

    if not contact_found:
        raise RuntimeError("Could not establish physical contact with the selected door surface.")

    print("\n===== Closed-loop surface pushing =====")
    success = False
    finish_rot = None
    theta_progress_ref = float(d.qpos[door_qadr])
    lost_contact_steps = 0

    for step in range(args.max_push_steps):
        theta = float(d.qpos[door_qadr])
        remaining = theta_closed - theta
        if abs(remaining) < args.close_tolerance or base_env._check_success():
            print("\nSUCCESS predicate / closed angle reached.")
            success = True
            break

        finish_mode = abs(remaining) < args.finish_angle
        push_point = door_point(hinge_pos, hinge_axis, door_push_start, theta, theta_start)
        tangent = closing_tangent(hinge_pos, hinge_axis, push_point, np.sign(remaining))
        tangent = force_tangent_direction(tangent, args.push_x_direction)
        tcp_now, rot_now = site_pose(base_env.sim, active_site)

        if finish_mode and finish_rot is None:
            finish_rot = rot_now.copy()
            print("\nEntering finish mode: holding current reachable orientation.")

        target_rot = finish_rot if finish_mode else door_facing_rotation(
            hinge_axis, tangent, T_active_tcp_wrist
        )

        door_delta = theta - theta_contact_ref
        Rdoor = Rotation.from_rotvec(hinge_axis * door_delta).as_matrix()
        desired_surface_offset = Rdoor @ surface_offset_ref
        surface_goal = push_point + desired_surface_offset
        surface_error = surface_goal - tcp_now

        surface_correction = surface_error - np.dot(surface_error, tangent) * tangent
        corr_norm = np.linalg.norm(surface_correction)
        if corr_norm > args.max_surface_correction:
            surface_correction *= args.max_surface_correction / corr_norm

        normal_error = float(np.dot(surface_error, tangent))
        normal_command = float(np.clip(normal_error + args.push_lead, 0.0, args.max_normal_command))
        target = tcp_now + surface_correction + normal_command * tangent

        actual, err = execute_target(
            env, base_env, robot, active_arm,
            target, target_rot, inactive_hold_q, inactive_hold_target,
            T_active_tcp_wrist, args.gripper,
        )

        theta_after = float(d.qpos[door_qadr])
        push_point_after = door_point(hinge_pos, hinge_axis, door_push_start, theta_after, theta_start)
        contacts = door_surface_contacts(base_env, door_body_ids)
        in_contact = bool(contacts)
        lost_contact_steps = 0 if in_contact else lost_contact_steps + 1
        tangential_error = float(np.dot(target - actual, tangent))

        if step > 0 and step % args.progress_window == 0:
            progress = abs(theta_after - theta_progress_ref)
            print(f"{args.progress_window}-step door progress={progress:.4f} rad")
            theta_progress_ref = theta_after

        if step % 10 == 0:
            actual_surface_offset = actual - push_point_after
            mode = "door-contact" if in_contact else "free-push"
            print(
                f"push {step:4d} door={theta_after:.3f} remaining={theta_closed-theta_after:.3f} "
                f"normal_cmd={normal_command:.4f} tan_err={tangential_error:.3f} err={err:.3f} "
                f"contact={in_contact} mode={mode} finish={finish_mode} lost={lost_contact_steps}"
            )
            print(
                "   surface_offset=", np.round(actual_surface_offset, 3),
                "target=", np.round(target, 3), "actual=", np.round(actual, 3),
            )

        last_target, last_rot = target.copy(), target_rot.copy()

    print("\n===== Result =====")
    final_success = bool(base_env._check_success())
    final_angle = float(d.qpos[door_qadr])
    print("success:", final_success)
    print("door final angle:", final_angle)
    print("door target angle:", float(theta_closed))

    metadata = {
        "layout": args.layout,
        "style": args.style,
        "seed": args.seed,
        "fixture_name": str(getattr(fxtr, "name", "")),
        "fixture_type": type(fxtr).__name__,
        "hinge_name": hinge_name,
        "active_arm": active_arm,
        "controller": controller_path,
        "spawn_mode": args.spawn_mode,
        "robot_base_position": np.asarray(spawn_pos).tolist(),
        "robot_base_orientation": np.asarray(spawn_ori).tolist(),
        "door_push_start": np.asarray(door_push_start).tolist(),
        "push_radius_fraction": args.push_radius_fraction,
        "push_height_fraction": args.push_height_fraction,
        "push_lead": args.push_lead,
        "success": final_success,
        "door_final_angle": final_angle,
        "door_target_angle": float(theta_closed),
    }

    if args.record_dir:
        metadata_dir = getattr(env, "ep_directory", None) or args.record_dir
        try:
            with open(os.path.join(metadata_dir, "generalized_close_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
            print("metadata:", os.path.join(metadata_dir, "generalized_close_metadata.json"))
        except Exception as e:
            print("Could not save metadata:", e)

    if args.keep_open:
        print("\nPress Ctrl+C to finish.")
        try:
            while True:
                execute_target(
                    env, base_env, robot, active_arm,
                    last_target, last_rot, inactive_hold_q, inactive_hold_target,
                    T_active_tcp_wrist, args.gripper,
                )
        except KeyboardInterrupt:
            pass

    env.close()
    return success or final_success


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="CloseSingleDoor")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER, help="Fallback controller config")
    parser.add_argument("--controller-left", default=None, help="Optional controller override when --arm left")
    parser.add_argument("--controller-right", default=None, help="Optional controller override when --arm right")

    parser.add_argument("--spawn-mode", choices=["hybrid", "robocasa", "door", "fixed"], default="hybrid")
    parser.add_argument("--spawn-distance", type=float, default=0.50, help="Robot-base distance from the open-door push point [m]")
    parser.add_argument("--spawn-lateral", type=float, default=0.0, help="Lateral base offset around the push point [m]")
    parser.add_argument("--spawn-offset-x", type=float, default=0.0, help="Offset passed to RoboCasa base-placement helper")
    parser.add_argument("--spawn-offset-y", type=float, default=0.0, help="Offset passed to RoboCasa base-placement helper")
    parser.add_argument("--face-push-point", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--robot-x", type=float, default=0.55)
    parser.add_argument("--robot-y", type=float, default=-1.95)
    parser.add_argument("--robot-z", type=float, default=0.0)
    parser.add_argument("--robot-yaw", type=float, default=np.pi / 2)
    parser.add_argument("--max-probe-mismatch", type=float, default=0.05)
    parser.add_argument("--probe-only", action="store_true")

    parser.add_argument("--push-radius-fraction", type=float, default=0.72, help="Radial location on door panel, 0=hinge and 1=outer edge")
    parser.add_argument("--push-height-fraction", type=float, default=0.50, help="Height on door panel, 0=bottom and 1=top")
    parser.add_argument("--push-x-direction", type=int, choices=[-1, 0, 1], default=0, help="Normally leave at 0; world-X override only for debugging")

    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--standoff", type=float, default=0.10)
    parser.add_argument("--extra-approach-clearance", type=float, default=0.10)
    parser.add_argument("--via-z", type=float, default=0.12)
    parser.add_argument("--approach-points", type=int, default=80)
    parser.add_argument("--steps-per-waypoint", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--gripper", type=float, default=1.0)

    parser.add_argument("--contact-clearance", "--handle-clearance", dest="contact_clearance", type=float, default=0.053)
    parser.add_argument("--contact-z-offset", type=float, default=0.0)
    parser.add_argument("--contact-radius", "--near-handle-contact-radius", dest="contact_radius", type=float, default=0.12)
    parser.add_argument("--contact-search-steps", type=int, default=500)
    parser.add_argument("--contact-search-interval", type=int, default=4)
    parser.add_argument("--contact-search-step", type=float, default=0.002)
    parser.add_argument("--max-contact-penetration", type=float, default=0.060)
    parser.add_argument("--align-position-tolerance", type=float, default=0.05)
    parser.add_argument("--align-angle-tolerance-deg", type=float, default=8.0)

    parser.add_argument("--push-lead", type=float, default=0.010, help="Persistent normal penetration into the door surface [m]")
    parser.add_argument("--max-normal-command", type=float, default=0.030, help="Maximum one-step normal target offset [m]")
    parser.add_argument("--max-surface-correction", type=float, default=0.002, help="Maximum lateral anti-slide correction per control step [m]")
    parser.add_argument("--finish-angle", type=float, default=0.12)
    parser.add_argument("--close-tolerance", type=float, default=0.020)
    parser.add_argument("--progress-window", type=int, default=10)
    parser.add_argument("--max-push-steps", type=int, default=5000)

    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--keep-open", action="store_true")
    args = parser.parse_args()
    run(args)
