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

from pathlib import Path
import time
from robocasa.models.fixtures import FixtureType

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
        "base_types": "RBY1Mount",
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
        "door_id": FixtureType.MICROWAVE,
        "initial_door_angle_deg": args.initial_door_angle_deg,
        # Start off the straight-elbow singularity; this is also the stow goal.
        "initial_arm_qpos": {args.arm: [0, 0, 0, -0.7, 0, 0.7, 0]},
    }
    if forced_placement is not None:
        pos, ori = forced_placement
        config["force_robot_placement"] = (np.asarray(pos, dtype=float).tolist(), np.asarray(ori, dtype=float).tolist())
    return config


def create_env(args, controller_config, renderer, forced_placement=None, record=False):
    np.random.seed(args.seed)
    config = make_config(args, controller_config, renderer, forced_placement)
    env = robosuite.make(**config)
    if record and args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)
        (Path(args.record_dir) / "env_kwargs.json").write_text(json.dumps(config, indent=2) + "\n")
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

    if args.spawn_mode in ("fixture", "robocasa", "hybrid"):
        fixture = fixture_pose(base_env, fxtr)
        pos, ori = base_env.compute_robot_base_placement_pose(
            fxtr, offset=[args.spawn_offset_x, args.spawn_offset_y])
        if args.base_local is not None:
            local = make_T([*args.base_local[:2], 0], Rotation.from_euler("z", np.deg2rad(args.base_local[2])).as_matrix())
            world = fixture @ local
            pos[:2] = world[:2, 3]
            ori[2] = np.arctan2(world[1, 0], world[0, 0])
        # Base Z comes from the floor / robot mount, independent of fixture height.
        return np.asarray(pos), np.asarray(ori)

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


def fixture_pose(env, fixture):
    body = env.sim.model.body_name2id(fixture.root_body)
    return make_T(env.sim.data.body_xpos[body].copy(), env.sim.data.body_xmat[body].reshape(3, 3).copy())


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
    try:
        from .door_motion import DoorMotion, DemoFailure
    except ImportError:
        from door_motion import DoorMotion, DemoFailure
    output = Path(args.output_dir or args.record_dir or f"logs/rby1_source/run_{time.time_ns()}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    controller_path = args.controller_left if args.arm == "left" and args.controller_left else args.controller
    controller_path = args.controller_right if args.arm == "right" and args.controller_right else controller_path
    controller_config = load_composite_controller_config(controller=controller_path, robot="RBY1")
    # Keep the other arm in its absolute joint hold, matching the Cartesian policy datasets.
    ik_config = controller_config["composite_controller_specific_configs"]
    ik_config["actuation_part_names"] = [args.arm]
    # Preserve the saved 31-D source action layout, including both reference
    # sites. Only the selected arm's joints participate in IK.
    result = dict(success=False, layout=args.layout, style=args.style, seed=args.seed,
                  initial_door_angle_deg=args.initial_door_angle_deg, failure_reason=None,
                  event_indexing="0-based post-action trace rows; raw states include the initial state at index 0",
                  parameters=vars(args).copy())
    env = motion = None
    try:
        spawn_pos, spawn_ori, signature = probe_scene(args, controller_config)
        result.update(base_position=spawn_pos.tolist(), base_orientation=spawn_ori.tolist())
        if args.probe_only:
            result["success"] = True
            return True
        env = create_env(args, controller_config, renderer=not args.headless,
                         forced_placement=(spawn_pos, spawn_ori), record=True)
        base_env = unwrap_env(env)
        robot = base_env.robots[0]
        set_ik_q0_from_current_pose(base_env, robot)
        fixture = discover_target_fixture(base_env)
        fixture_world = fixture_pose(base_env, fixture)
        if fixture.name != signature["fixture"]:
            raise DemoFailure("setup_error", "Probe and recorded environment selected different fixtures")
        base_world = make_T(spawn_pos, Rotation.from_euler("xyz", spawn_ori).as_matrix())
        result.update(fixture=fixture.name, base_in_fixture=(np.linalg.inv(fixture_world) @ base_world).tolist())
        hinge, hinge_name, theta_start, theta_closed, anchor, axis = discover_hinge(base_env, fixture)
        if abs(theta_closed) > 1e-6:
            raise DemoFailure("setup_error", "Microwave controller expects a zero closed angle")
        result.update(hinge=hinge_name, actual_initial_angle_deg=float(-np.rad2deg(theta_start)))
        if abs(-np.rad2deg(theta_start) - args.initial_door_angle_deg) > 0.5:
            raise DemoFailure("initial_angle_mismatch", "Requested door angle changed during reset; check base/door clearance")
        point, _, door_bodies, _ = discover_door_push_point(base_env, hinge, anchor, axis,
                                                         args.push_radius_fraction, args.push_height_fraction)
        active = args.arm
        inactive = other_arm(active)
        start_p, start_r = site_pose(base_env.sim, arm_site(active))
        wrist_p, wrist_r = site_pose(base_env.sim, arm_wrist_site(active))
        tcp_wrist = np.linalg.inv(make_T(start_p, start_r)) @ make_T(wrist_p, wrist_r)
        inactive_p, inactive_r = site_pose(base_env.sim, arm_site(inactive))
        inactive_wp, inactive_wr = site_pose(base_env.sim, arm_wrist_site(inactive))
        inactive_transform = np.linalg.inv(make_T(inactive_p, inactive_r)) @ make_T(inactive_wp, inactive_wr)
        inactive_target = tcp_to_wrist_target(inactive_p, inactive_r, inactive_transform)
        inactive_q = np.array([base_env.sim.data.get_joint_qpos(n) for n in arm_joint_names(inactive)])
        base_site = robot.robot_model.base.correct_naming("center")

        def base_pose():
            return make_T(*site_pose(base_env.sim, base_site))

        def command(p, r):
            execute_target(env, base_env, robot, active, p, r, inactive_q,
                           inactive_target, tcp_wrist, args.gripper)

        motion = DoorMotion(base_env, command, lambda: site_pose(base_env.sim, arm_site(active)),
                            base_pose, hinge, point, args)
        motion.run(lambda tangent: door_facing_rotation(axis, tangent, tcp_wrist))
        result.update(success=True, final_angle_deg=float(-np.rad2deg(motion.angle)))
        return True
    except DemoFailure as error:
        result.update(failure_reason=error.reason, detail=str(error))
        return False
    except KeyboardInterrupt:
        result.update(failure_reason="interrupted", detail="Source collection interrupted")
        return False
    except Exception as error:
        result.update(failure_reason="setup_error", detail=f"{type(error).__name__}: {error}")
        return False
    finally:
        if motion is not None:
            motion.save_trace(output / "trace.npz")
            result.update(events=motion.events, steps=len(motion.trace), phase=motion.phase,
                          final_angle_deg=float(-np.rad2deg(motion.angle)), stow_base=motion.home.tolist())
            if motion.trace:
                home = motion.base_pose() @ motion.home
                position, rotation = motion.tcp_pose()
                result.update(final_stow_position_error_m=float(np.linalg.norm(position-home[:3, 3])),
                              final_stow_rotation_error_deg=float(np.rad2deg(rotation_error(rotation, home[:3, :3]))),
                              max_tracking_error_m=max(row["tracking_error"] for row in motion.trace))
                result["base_drift_m"] = max(float(np.linalg.norm(row["base"][:3, 3]-motion.initial_base[:3, 3]))
                                             for row in motion.trace)
        if env is not None:
            if args.record_dir:
                result["raw_demo"] = getattr(env, "ep_directory", None)
            env.close()
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        print(f"Source result: {output}", flush=True)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parser():
    parser = argparse.ArgumentParser(description="RB-Y1 microwave source demos with hinge-following and release/stow")
    parser.add_argument("--environment", default="CloseSingleDoor")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER, help="Fallback controller config")
    parser.add_argument("--controller-left", default=None, help="Optional controller override when --arm left")
    parser.add_argument("--controller-right", default=None, help="Optional controller override when --arm right")

    parser.add_argument("--spawn-mode", choices=["fixture", "hybrid", "robocasa", "door", "fixed"], default="fixture")
    parser.add_argument("--spawn-distance", type=float, default=0.50, help="Robot-base distance from the open-door push point [m]")
    parser.add_argument("--spawn-lateral", type=float, default=0.0, help="Lateral base offset around the push point [m]")
    parser.add_argument("--spawn-offset-x", type=float, default=0.0, help="Offset passed to RoboCasa base-placement helper")
    parser.add_argument("--spawn-offset-y", type=float, default=-0.35, help="Fixture-side setback for the RB-Y1 footprint [m]")
    parser.add_argument("--robot-x", type=float, default=None)
    parser.add_argument("--robot-y", type=float, default=None)
    parser.add_argument("--robot-z", type=float, default=0.0)
    parser.add_argument("--robot-yaw", type=float, default=None)
    parser.add_argument("--probe-only", action="store_true")

    parser.add_argument("--push-radius-fraction", type=float, default=0.72, help="Radial location on door panel, 0=hinge and 1=outer edge")
    parser.add_argument("--push-height-fraction", type=float, default=0.50, help="Height on door panel, 0=bottom and 1=top")
    parser.add_argument("--push-x-direction", type=int, choices=[-1, 0, 1], default=0, help="Normally leave at 0; world-X override only for debugging")

    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--standoff", type=float, default=0.0)
    parser.add_argument("--gripper", type=float, default=1.0)

    parser.add_argument("--contact-clearance", "--handle-clearance", dest="contact_clearance", type=float, default=0.053)
    parser.add_argument("--contact-z-offset", type=float, default=0.0)
    parser.add_argument("--contact-search-steps", type=int, default=500)
    parser.add_argument("--contact-search-step", type=float, default=0.002)
    parser.add_argument("--max-contact-penetration", type=float, default=0.060)

    parser.add_argument("--push-lead", type=float, default=0.010, help="Maximum arc-length closing lead, tapered near closure [m]")
    parser.add_argument("--max-contact-relief", "--max-normal-command", dest="max_normal_command", type=float,
                        default=0.010, help="Maximum unloading offset during brief contact loss; sustained loss fails [m]")
    parser.add_argument("--finish-angle", type=float, default=0.12)
    parser.add_argument("--close-tolerance", type=float, default=0.020)
    parser.add_argument("--max-push-steps", type=int, default=5000)

    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--initial-door-angle-deg", type=float, default=60)
    parser.add_argument("--base-local", nargs=3, type=float, metavar=("X", "Y", "YAW_DEG"),
                        help="Base XY/yaw in the stationary microwave frame; floor height is retained")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", default=None, help="Result JSON and trace directory")
    parser.add_argument("--tcp-speed", type=float, default=0.10)
    parser.add_argument("--lift-before-turn", action="store_true",
                        help="Use a 90-degree shoulder pre-lift; --pre-lift-deg selects a smaller lift")
    parser.add_argument("--pre-lift-deg", type=float, default=None,
                        help="Shoulder pre-lift angle [0, 90] before the normal approach; also used during stow")
    parser.add_argument("--direct-approach", action="store_true",
                        help="Reach the pre-contact point along a straight TCP path while aligning")
    parser.add_argument("--orientation-speed-deg", type=float, default=15)
    parser.add_argument("--orientation-follow", type=float, default=0.6)
    parser.add_argument("--max-orientation-deg", type=float, default=45)
    parser.add_argument("--max-slide", type=float, default=0.08)
    parser.add_argument("--closed-dwell", type=float, default=0.5)
    parser.add_argument("--retreat-distance", type=float, default=0.15)
    parser.add_argument("--hold-seconds", type=float, default=0.75)
    return parser


def main():
    options = parser()
    args = options.parse_args()
    if not 0 < args.initial_door_angle_deg <= 90:
        options.error("Use an initial door angle between 0 and 90 degrees (exclusive of zero)")
    if args.pre_lift_deg is not None and not 0 <= args.pre_lift_deg <= 90:
        options.error("Use --pre-lift-deg between 0 and 90 degrees")
    if args.push_x_direction:
        options.error("World-X forcing is incompatible with the door-relative source controller")
    if args.keep_open:
        options.error("Use the recorded final hold; --keep-open would append unlabelled actions")
    if args.spawn_mode == "fixed" and any(value is None for value in (args.robot_x, args.robot_y, args.robot_yaw)):
        options.error("Fixed placement requires explicit --robot-x, --robot-y, and --robot-yaw")
    if any(value <= 0 for value in (args.tcp_speed, args.orientation_speed_deg, args.closed_dwell,
                                    args.hold_seconds, args.push_lead, args.finish_angle, args.close_tolerance)):
        options.error("Motion rates, dwell times, lead, and tolerances must be positive")
    if not 0 <= args.orientation_follow <= 1 or min(args.max_slide, args.max_orientation_deg, args.max_normal_command) < 0:
        options.error("Use orientation-follow in [0, 1] and nonnegative slide/orientation bounds")
    raise SystemExit(0 if run(args) else 2)


if __name__ == "__main__":
    main()
