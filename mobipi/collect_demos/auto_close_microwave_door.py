#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import mujoco
import numpy as np
import robosuite
import robocasa  # registers RoboCasa environments
from scipy.spatial.transform import Rotation
from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import WholeBody
from robosuite.wrappers import DataCollectionWrapper

DEFAULT_CONTROLLER = str(Path(robosuite.__file__).resolve().parent / "controllers/config/robots/default_rby1_whole_body_ik.json")
DOOR_SURFACE_INSET = 0.06  # 6 cm from handle toward hinge: flat door surface

def raw_model(sim):
    return sim.model._model if hasattr(sim.model, '_model') else sim.model

def raw_data(sim):
    return sim.data._data if hasattr(sim.data, '_data') else sim.data

def forward(sim):
    try:
        sim.forward()
    except Exception:
        mujoco.mj_forward(raw_model(sim), raw_data(sim))

def unwrap_env(env):
    base = env
    seen = set()
    while True:
        if base is None:
            raise RuntimeError("Environment unwrapping reached None.")
        if id(base) in seen:
            raise RuntimeError("Detected a cycle while unwrapping environment wrappers.")
        seen.add(id(base))
        inner = getattr(base, "env", None)
        if inner is None or inner is base:
            return base
        base = inner

def normalize(v, eps=1e-09):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        raise RuntimeError(f'Cannot normalize vector {v}')
    return v / n

def smoothstep(t):
    return 3.0 * t ** 2 - 2.0 * t ** 3

def lerp_points(a, b, n):
    a, b = (np.asarray(a), np.asarray(b))
    return np.array([a + smoothstep(t) * (b - a) for t in np.linspace(0, 1, n)])

def discover_target_fixture(env):
    ep_meta = env.get_ep_meta()
    print('\n===== Fixture discovery =====')
    print('instruction:', ep_meta.get('lang', ''))
    refs = getattr(env, 'fixture_refs', {})
    print('fixture_refs keys:', list(refs.keys()))
    if 'door_fxtr' in refs:
        fxtr = refs['door_fxtr']
        print("Selected fixture_refs['door_fxtr']")
        print('type:', type(fxtr).__name__)
        print('name:', getattr(fxtr, 'name', None))
        print('\nUseful fixture attributes:')
        for attr in ['name', 'joints', 'joint_names', 'door_joints', 'handle_name', 'handles', 'sites']:
            if hasattr(fxtr, attr):
                try:
                    print(f'  {attr}: {getattr(fxtr, attr)}')
                except Exception:
                    pass
        return fxtr
    raise RuntimeError(f"Could not find 'door_fxtr'. Available refs: {list(refs.keys())}")

def rotation_error(R_actual, R_target):
    Rerr = R_target @ R_actual.T
    return np.linalg.norm(Rotation.from_matrix(Rerr).as_rotvec())

def site_pose(sim, name):
    m, d = (raw_model(sim), raw_data(sim))
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    if sid < 0:
        raise RuntimeError(f'Site not found: {name}')
    pos = np.asarray(d.site_xpos[sid], dtype=float).copy()
    mat = np.asarray(d.site_xmat[sid], dtype=float).reshape(3, 3).copy()
    return (pos, mat)

def object_position(sim, name):
    """Try site -> geom -> body."""
    m, d = (raw_model(sim), raw_data(sim))
    for obj_type, array in [(mujoco.mjtObj.mjOBJ_SITE, d.site_xpos), (mujoco.mjtObj.mjOBJ_GEOM, d.geom_xpos), (mujoco.mjtObj.mjOBJ_BODY, d.xpos)]:
        idx = mujoco.mj_name2id(m, obj_type, name)
        if idx >= 0:
            return np.asarray(array[idx], dtype=float).copy()
    return None

def dump_fixture_candidates(sim, fixture_name):
    m = raw_model(sim)
    token = fixture_name.lower()
    print('\nFixture-related joints:')
    for jid in range(m.njnt):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name and token in name.lower():
            print(' ', jid, name)
    print('\nFixture-related sites:')
    for sid in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid)
        if name and (token in name.lower() or 'handle' in name.lower()):
            print(' ', sid, name)
    print('\nFixture-related geoms:')
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if name and (token in name.lower() or 'handle' in name.lower()):
            print(' ', gid, name)

def discover_hinge(base_env, fxtr):
    """
    Finds the door hinge belonging to the task fixture and determines
    which joint value corresponds to the closed state.
    Does not require fxtr.open_door() / fxtr.close_door().
    """
    sim = base_env.sim
    m, d = (raw_model(sim), raw_data(sim))
    qpos0 = np.asarray(d.qpos).copy()
    qvel0 = np.asarray(d.qvel).copy()
    fixture_name = str(getattr(fxtr, 'name', '')).lower()
    print('\n===== Hinge discovery =====')
    print('fixture name:', fixture_name)
    fixture_joint_names = []
    for attr in ['joints', 'joint_names', 'door_joints']:
        if hasattr(fxtr, attr):
            try:
                value = getattr(fxtr, attr)
                if isinstance(value, str):
                    fixture_joint_names.append(value)
                elif value is not None:
                    fixture_joint_names.extend(list(value))
            except Exception:
                pass
    fixture_joint_names = [str(x) for x in fixture_joint_names]
    print('fixture-provided joints:', fixture_joint_names)
    candidates = []
    for jid in range(m.njnt):
        if int(m.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name is None:
            continue
        name_lower = name.lower()
        score = 0
        if name in fixture_joint_names:
            score += 100
        if fixture_name and fixture_name in name_lower:
            score += 50
        if 'door' in name_lower:
            score += 10
        if 'hinge' in name_lower:
            score += 10
        if score > 0:
            candidates.append((score, jid, name))
    if not candidates:
        print('No named candidates; considering every hinge.')
        for jid in range(m.njnt):
            if int(m.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_HINGE):
                name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
                candidates.append((0, jid, name))
    candidates.sort(reverse=True)
    print('\nHinge candidates:')
    for score, jid, name in candidates:
        adr = int(m.jnt_qposadr[jid])
        print(f'  score={score:3d}', f'id={jid:3d}', f'name={name}', f'q={float(d.qpos[adr]):.4f}', f'range={np.asarray(m.jnt_range[jid])}')
    best = None
    for score, jid, name in candidates:
        adr = int(m.jnt_qposadr[jid])
        theta_start = float(qpos0[adr])
        lo = float(m.jnt_range[jid][0])
        hi = float(m.jnt_range[jid][1])
        tests = [0.0, lo, hi]
        valid_tests = []
        for theta in tests:
            if theta < lo - 1e-06 or theta > hi + 1e-06:
                continue
            if not any((abs(theta - x) < 1e-06 for x in valid_tests)):
                valid_tests.append(theta)
        print(f'\nTesting {name}')
        for theta_test in valid_tests:
            d.qpos[:] = qpos0
            d.qvel[:] = qvel0
            d.qpos[adr] = theta_test
            forward(sim)
            try:
                success = bool(base_env._check_success())
            except Exception as e:
                print(' success check failed:', e)
                success = False
            print(f'  theta={theta_test:.4f}', f'success={success}')
            if success:
                best = (jid, name, theta_start, theta_test)
                break
        if best is not None:
            break
    d.qpos[:] = qpos0
    d.qvel[:] = qvel0
    forward(sim)
    if best is None:
        raise RuntimeError('Could not determine which hinge value closes the task door.')
    jid, joint_name, theta_start, theta_closed = best
    hinge_pos = np.asarray(d.xanchor[jid], dtype=float).copy()
    hinge_axis = normalize(np.asarray(d.xaxis[jid], dtype=float))
    print('\n===== Selected door hinge =====')
    print('joint:        ', joint_name)
    print('theta start:  ', theta_start)
    print('theta closed: ', theta_closed)
    print('delta:        ', theta_closed - theta_start)
    print('hinge pos:    ', np.round(hinge_pos, 4))
    print('hinge axis:   ', np.round(hinge_axis, 4))
    return (jid, theta_start, theta_closed, hinge_pos, hinge_axis)

def discover_handle(base_env, fxtr, hinge_pos, ee_pos):
    """
    Prefer fixture-provided handle names.
    Otherwise search MuJoCo sites/geoms containing both fixture name
    and "handle".
    """
    sim = base_env.sim
    m = raw_model(sim)
    candidates = []
    for attr in ['handle_name', 'left_handle_name', 'right_handle_name']:
        try:
            name = getattr(fxtr, attr)
        except Exception:
            continue
        if isinstance(name, str):
            p = object_position(sim, name)
            if p is not None:
                candidates.append((name, p, 'fixture property'))
            p = object_position(sim, name + '_default_site')
            if p is not None:
                candidates.append((name + '_default_site', p, 'fixture property'))
    token = fxtr.name.lower()
    for sid in range(m.nsite):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, sid)
        if not name:
            continue
        lname = name.lower()
        if token in lname and 'handle' in lname:
            p = np.asarray(raw_data(sim).site_xpos[sid], dtype=float).copy()
            candidates.append((name, p, 'site'))
    for gid in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid)
        if not name:
            continue
        lname = name.lower()
        if token in lname and 'handle' in lname:
            p = np.asarray(raw_data(sim).geom_xpos[gid], dtype=float).copy()
            candidates.append((name, p, 'geom'))
    unique = {}
    for name, p, source in candidates:
        unique[name] = (name, p, source)
    candidates = list(unique.values())
    if not candidates:
        dump_fixture_candidates(sim, fxtr.name)
        raise RuntimeError(f"Could not automatically find a handle for '{fxtr.name}'.")
    print('\n===== Handle candidates =====')
    for name, p, source in candidates:
        print(f'{name:60s}', np.round(p, 4), f'r_hinge={np.linalg.norm(p - hinge_pos):.3f}', f'd_robot={np.linalg.norm(p - ee_pos):.3f}', source)
    valid = [c for c in candidates if 0.05 < np.linalg.norm(c[1] - hinge_pos) < 1.5]
    if not valid:
        valid = candidates
    valid.sort(key=lambda c: (-np.linalg.norm(c[1] - hinge_pos), np.linalg.norm(c[1] - ee_pos)))
    name, handle_pos, source = valid[0]
    print('\nSelected handle:', name)
    print('handle pos:     ', np.round(handle_pos, 4))
    return (name, handle_pos)

def door_point(hinge_pos, hinge_axis, initial_point, theta, theta_start):
    delta = theta - theta_start
    rot = Rotation.from_rotvec(hinge_axis * delta)
    r0 = initial_point - hinge_pos
    return hinge_pos + rot.apply(r0)

def closing_tangent(hinge_pos, hinge_axis, point, theta_delta_sign):
    r = point - hinge_pos
    tangent = np.cross(hinge_axis, r)
    tangent *= theta_delta_sign
    return normalize(tangent)

def door_facing_rotation(hinge_axis, door_normal, T_tcp_wrist):
    """
    Orient RBY1 gripper perpendicular to the door.
    Assumes the gripper/tool length is along local Z, which is consistent
    with the ~25 cm TCP <-> wrist offset in the RBY1 gripper model.
    hinge_axis: vertical door hinge axis
    door_normal: direction perpendicular to door surface
    """
    up = normalize(hinge_axis)
    normal = normalize(door_normal)
    tcp_to_wrist_z = T_tcp_wrist[2, 3]
    sign = np.sign(tcp_to_wrist_z)
    if abs(sign) < 1e-06:
        sign = 1.0
    z_world = -sign * normal
    y_world = up - np.dot(up, z_world) * z_world
    y_world = normalize(y_world)
    x_world = normalize(np.cross(y_world, z_world))
    R = np.column_stack([x_world, y_world, z_world])
    return R

def pose6(position, rotation_matrix):
    rotvec = Rotation.from_matrix(rotation_matrix).as_rotvec()
    return np.concatenate([position, rotvec])

def interpolate_rotation(R0, R1, t):
    relative = R1 @ R0.T
    rotvec = Rotation.from_matrix(relative).as_rotvec()
    return Rotation.from_rotvec(t * rotvec).as_matrix() @ R0

def set_ik_q0_from_current_pose(base_env, robot):
    if not isinstance(robot.composite_controller, WholeBody):
        return
    ik = robot.composite_controller.joint_action_policy
    ik.q0 = np.array([float(base_env.sim.data.get_joint_qpos(name)) for name in ik.joint_names])
    print('\nIK q0:', np.round(ik.q0, 3))

def make_T(pos, rot):
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos
    return T

def pose_target_for_ref(tcp_pos, tcp_rot, T_tcp_ref):
    T_world_ref = make_T(tcp_pos, tcp_rot) @ T_tcp_ref
    return pose6(T_world_ref[:3, 3], T_world_ref[:3, :3])

def arm_tcp_site(arm):
    return f"gripper0_{arm}_grip_site"

def arm_default_ref_site(arm):
    return f"gripper0_{arm}_ft_frame"

def arm_joint_names(arm):
    return [f"robot0_{arm}_arm_{i}" for i in range(7)]

def whole_body_arm_refs(controller_config):
    specific = controller_config.get("composite_controller_specific_configs", {})
    parts = specific.get("actuation_part_names", [])
    refs = specific.get("ref_name", [])
    if isinstance(parts, str):
        parts = [parts]
    if isinstance(refs, str):
        refs = [refs]
    parts, refs = list(parts or []), list(refs or [])
    if refs and len(refs) != len(parts):
        raise RuntimeError(f"WHOLE_BODY_IK config has {len(parts)} actuation_part_names but {len(refs)} ref_name entries.")
    ref_map = {part: ref for part, ref in zip(parts, refs)}
    for arm in ("left", "right"):
        if arm in parts and arm not in ref_map:
            ref_map[arm] = arm_default_ref_site(arm)
    return parts, ref_map

def choose_active_arm(requested_arm, robot, ik_parts):
    available = [arm for arm in ("left", "right") if arm in robot.arms]
    if requested_arm == "auto":
        for arm in ("left", "right"):
            if arm in available and arm in ik_parts:
                return arm
        raise RuntimeError(f"No arm is controlled by WHOLE_BODY_IK. Robot arms={available}, actuation_part_names={ik_parts}")
    if requested_arm not in available:
        raise RuntimeError(f"Requested arm '{requested_arm}' is not available. Robot arms={available}")
    if requested_arm not in ik_parts:
        raise RuntimeError(
            f"Requested active arm '{requested_arm}' is not in WHOLE_BODY_IK actuation_part_names={ik_parts}. "
            "Use a controller config that actuates this arm, or choose another --arm."
        )
    return requested_arm

def joint_hold_command(base_env, controller_config, arm):
    arm_cfg = controller_config.get("body_parts_controller_configs", {}).get("arms", {}).get(arm, {})
    controller_type = str(arm_cfg.get("type", "JOINT_POSITION")).upper()
    input_type = str(arm_cfg.get("input_type", "absolute")).lower()
    q = np.array([float(base_env.sim.data.get_joint_qpos(name)) for name in arm_joint_names(arm)])
    if controller_type == "JOINT_POSITION":
        if input_type == "absolute":
            return q, "joint-position absolute hold"
        if input_type == "relative":
            return np.zeros_like(q), "joint-position relative zero hold"
        raise RuntimeError(f"Unsupported input_type '{input_type}' for {arm} JOINT_POSITION controller.")
    if controller_type in ("JOINT_VELOCITY", "JOINT_TORQUE"):
        return np.zeros_like(q), f"{controller_type.lower()} zero hold"
    raise RuntimeError(
        f"Inactive arm '{arm}' uses unsupported controller type '{controller_type}'. "
        "This script currently supports inactive WHOLE_BODY_IK arms and JOINT_POSITION / JOINT_VELOCITY / JOINT_TORQUE holds."
    )

def build_controller_interface(base_env, robot, controller_config, requested_arm):
    if not isinstance(robot.composite_controller, WholeBody):
        raise RuntimeError(f"Expected a WholeBody composite controller, got {type(robot.composite_controller).__name__}.")
    ik_parts, ref_map = whole_body_arm_refs(controller_config)
    active_arm = choose_active_arm(requested_arm, robot, ik_parts)
    active_tcp = arm_tcp_site(active_arm)
    active_ref = ref_map.get(active_arm, arm_default_ref_site(active_arm))
    tcp_pos, tcp_rot = site_pose(base_env.sim, active_tcp)
    ref_pos, ref_rot = site_pose(base_env.sim, active_ref)
    T_tcp_ref = np.linalg.inv(make_T(tcp_pos, tcp_rot)) @ make_T(ref_pos, ref_rot)
    hold_commands = {}
    hold_descriptions = {}
    for arm in robot.arms:
        if arm == active_arm:
            continue
        if arm in ik_parts:
            ref_site = ref_map.get(arm, arm_default_ref_site(arm))
            ref_hold_pos, ref_hold_rot = site_pose(base_env.sim, ref_site)
            hold_commands[arm] = pose6(ref_hold_pos, ref_hold_rot)
            hold_descriptions[arm] = f"WHOLE_BODY_IK pose hold at {ref_site}"
        else:
            hold_commands[arm], hold_descriptions[arm] = joint_hold_command(base_env, controller_config, arm)
    print("\n===== Controller interface =====")
    print("WHOLE_BODY_IK actuation parts:", ik_parts)
    print("WHOLE_BODY_IK ref map:        ", ref_map)
    print("active arm:                   ", active_arm)
    print("active TCP site:              ", active_tcp)
    print("active controller ref:        ", active_ref)
    print("TCP->ref translation:         ", np.round(T_tcp_ref[:3, 3], 4))
    for arm, desc in hold_descriptions.items():
        print(f"inactive {arm} hold:             {desc}; dim={hold_commands[arm].size}")
    return active_arm, active_tcp, active_ref, tcp_pos, tcp_rot, T_tcp_ref, hold_commands

def create_robot_action(robot, active_arm, active_target, hold_commands, gripper_value=1.0):
    action_dict = dict(hold_commands)
    action_dict[active_arm] = np.asarray(active_target, dtype=float)
    for arm in robot.arms:
        dof = robot.gripper[arm].dof
        if dof > 0:
            action_dict[f"{arm}_gripper"] = np.repeat([gripper_value], dof)
    try:
        action = np.asarray(robot.create_action_vector(action_dict), dtype=float)
    except Exception as e:
        dims = {k: int(np.asarray(v).size) for k, v in action_dict.items()}
        raise RuntimeError(f"Could not create robot action from controller-aware action_dict dims={dims}: {e}") from e
    if not np.all(np.isfinite(action)):
        raise RuntimeError("Controller produced a non-finite action vector.")
    return action

def execute_target(env, base_env, robot, active_arm, active_tcp_site, tcp_pos, tcp_rot, hold_commands, T_tcp_ref, gripper_value, repeats=1):
    active_target = pose_target_for_ref(tcp_pos, tcp_rot, T_tcp_ref)
    for _ in range(repeats):
        action = create_robot_action(robot, active_arm, active_target, hold_commands, gripper_value)
        env.step(action)
        if not np.all(np.isfinite(raw_data(base_env.sim).qpos)):
            raise RuntimeError('Robot state became NaN / Inf.')
    actual_tcp, _ = site_pose(base_env.sim, active_tcp_site)
    return actual_tcp, float(np.linalg.norm(actual_tcp - tcp_pos))

def door_surface_contacts(base_env, fixture_name, push_point, radius=0.10):
    """Robot <-> microwave/fixture contacts near the chosen flat-door push point."""
    m, d = raw_model(base_env.sim), raw_data(base_env.sim)
    fixture_token = str(fixture_name).lower()
    robot_tokens = ("robot0", "rby1", "gripper", "left_hand", "right_hand", "left_gripper", "right_gripper")
    contacts = []
    for i in range(d.ncon):
        c = d.contact[i]
        g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
        g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
        b1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom1])) or ""
        b2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[c.geom2])) or ""
        s1, s2 = f"{g1} {b1}".lower(), f"{g2} {b2}".lower()
        fixture1 = (fixture_token and fixture_token in s1) or "microwave" in s1
        fixture2 = (fixture_token and fixture_token in s2) or "microwave" in s2
        robot1 = any(tok in s1 for tok in robot_tokens)
        robot2 = any(tok in s2 for tok in robot_tokens)
        if not ((robot1 and fixture2) or (robot2 and fixture1)):
            continue
        p = np.asarray(c.pos, dtype=float)
        dist = float(np.linalg.norm(p - push_point))
        if dist <= radius:
            contacts.append((g1, g2, dist))
    return contacts

def force_tangent_direction(tangent, push_x_direction):
    if push_x_direction != 0 and np.sign(tangent[0]) != push_x_direction:
        tangent = -tangent
    return tangent

def run(args):
    controller_config = load_composite_controller_config(controller=args.controller, robot="RBY1")
    config = {
        "env_name": args.environment,
        "robots": "RBY1",
        "controller_configs": controller_config,
        "layout_ids": [args.layout],
        "style_ids": [args.style],
        "has_renderer": True,
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
    if not args.no_fixed_placement:
        config["force_robot_placement"] = ([args.robot_x, args.robot_y, args.robot_z], [0.0, 0.0, args.robot_yaw])
    np.random.seed(args.seed)
    env = robosuite.make(**config)
    if args.record_dir:
        os.makedirs(args.record_dir, exist_ok=True)
        env = DataCollectionWrapper(env, args.record_dir)
        print("Recording raw trajectory to:", args.record_dir)
    np.random.seed(args.seed)
    env.reset()
    base_env = unwrap_env(env)
    robot = base_env.robots[0]
    set_ik_q0_from_current_pose(base_env, robot)
    active_arm, active_tcp_site, active_ref_site, active_start_pos, active_start_rot, T_active_tcp_ref, hold_commands = build_controller_interface(
        base_env, robot, controller_config, args.arm
    )
    fxtr = discover_target_fixture(base_env)
    hinge_id, theta_start, theta_closed, hinge_pos, hinge_axis = discover_hinge(base_env, fxtr)
    delta_total = theta_closed - theta_start
    if abs(delta_total) < 1e-4:
        raise RuntimeError("Door is already essentially closed.")
    direction_sign = np.sign(delta_total)
    # Handle is used only as a landmark. We never try to touch or track it.
    handle_name, handle_start = discover_handle(base_env, fxtr, hinge_pos, active_start_pos)
    radial = handle_start - hinge_pos
    radial -= np.dot(radial, hinge_axis) * hinge_axis
    radial_dir = normalize(radial)
    door_push_start = handle_start - DOOR_SURFACE_INSET * radial_dir
    door_radius = np.linalg.norm(np.cross(door_push_start - hinge_pos, hinge_axis))
    tangent0 = closing_tangent(hinge_pos, hinge_axis, door_push_start, direction_sign)
    tangent0 = force_tangent_direction(tangent0, args.push_x_direction)
    contact_rot = door_facing_rotation(hinge_axis, tangent0, T_active_tcp_ref)
    print("\n===== Flat-door push point =====")
    print("active arm:      ", active_arm)
    print("handle landmark: ", np.round(handle_start, 4))
    print("door push point: ", np.round(door_push_start, 4))
    print("inset from handle:", DOOR_SURFACE_INSET, "m toward hinge")
    print("door radius:      ", round(float(door_radius), 4))
    print("closing tangent:  ", np.round(tangent0, 4))
    # Approach the flat door surface, not the handle.
    pre_contact = door_push_start - (args.standoff + args.handle_clearance) * tangent0
    far_pre_contact = door_push_start - (args.standoff + args.handle_clearance + 0.10) * tangent0
    lift_pose = far_pre_contact.copy()
    lift_pose[2] = door_push_start[2] + args.via_z
    pre_contact_high = pre_contact.copy()
    pre_contact_high[2] = lift_pose[2]
    trajectory = []
    points = lerp_points(active_start_pos, far_pre_contact, max(args.approach_points // 3, 2))
    for i, p in enumerate(points):
        t = i / max(len(points) - 1, 1)
        trajectory.append((p, interpolate_rotation(active_start_rot, contact_rot, t)))
    for p in lerp_points(far_pre_contact, lift_pose, max(args.approach_points // 3, 2)):
        trajectory.append((p, contact_rot))
    for p in lerp_points(lift_pose, pre_contact_high, max(args.approach_points // 3, 2)):
        trajectory.append((p, contact_rot))
    for p in lerp_points(pre_contact_high, pre_contact, max(args.approach_points // 4, 2)):
        trajectory.append((p, contact_rot))
    print("\n===== Executing approach =====")
    print("waypoints:", len(trajectory))
    print("door angle:", theta_start, "->", theta_closed)
    for _ in range(args.settle_steps):
        execute_target(env, base_env, robot, active_arm, active_tcp_site, active_start_pos, active_start_rot, hold_commands, T_active_tcp_ref, args.gripper)
    last_target, last_rot = active_start_pos.copy(), active_start_rot.copy()
    for i, (target, target_rot) in enumerate(trajectory):
        actual, err = execute_target(
            env, base_env, robot, active_arm, active_tcp_site, target, target_rot,
            hold_commands, T_active_tcp_ref, args.gripper, repeats=args.steps_per_waypoint,
        )
        last_target, last_rot = target.copy(), target_rot.copy()
        if i % 10 == 0:
            qdoor = float(raw_data(base_env.sim).qpos[int(raw_model(base_env.sim).jnt_qposadr[hinge_id])])
            print(f"approach {i:3d}/{len(trajectory):3d}", "target=", np.round(target, 3), "actual=", np.round(actual, 3), f"err={err:.3f}", f"door={qdoor:.3f}")
    m, d = raw_model(base_env.sim), raw_data(base_env.sim)
    door_qadr = int(m.jnt_qposadr[hinge_id])
    print("\n===== Aligning with flat door surface =====")
    for step in range(100):
        actual, err = execute_target(env, base_env, robot, active_arm, active_tcp_site, pre_contact, contact_rot, hold_commands, T_active_tcp_ref, args.gripper)
        _, actual_rot = site_pose(base_env.sim, active_tcp_site)
        ang_err = rotation_error(actual_rot, contact_rot)
        if step % 10 == 0:
            print(f"align {step:3d}", f"pos_err={err:.3f}", f"rot_err={np.degrees(ang_err):.1f} deg")
        if err < 0.05 and ang_err < np.deg2rad(8):
            break
    print("\n===== Acquiring FLAT DOOR contact =====")
    clearance = args.handle_clearance
    contact_found = False
    for step in range(500):
        theta = float(d.qpos[door_qadr])
        remaining = theta_closed - theta
        push_point = door_point(hinge_pos, hinge_axis, door_push_start, theta, theta_start)
        tangent = closing_tangent(hinge_pos, hinge_axis, push_point, np.sign(remaining))
        tangent = force_tangent_direction(tangent, args.push_x_direction)
        target_rot = door_facing_rotation(hinge_axis, tangent, T_active_tcp_ref)
        target = push_point - clearance * tangent
        target[2] += args.contact_z_offset
        actual, err = execute_target(
            env, base_env, robot, active_arm, active_tcp_site, target, target_rot,
            hold_commands, T_active_tcp_ref, args.gripper, repeats=2,
        )
        contacts = door_surface_contacts(base_env, fxtr.name, push_point, args.near_handle_contact_radius)
        if step % 5 == 0:
            print(f"surface-search {step:3d}", f"clearance={clearance:.3f}", "target=", np.round(target, 3), "actual=", np.round(actual, 3), f"err={err:.3f}", "contacts=", contacts[:3])
        if contacts:
            contact_tcp_pos, push_rot_ref = site_pose(
                base_env.sim, active_tcp_site
            )
            theta_contact_ref = float(d.qpos[door_qadr])
            surface_offset_ref = contact_tcp_pos - push_point
            print("\nFLAT DOOR CONTACT at clearance:", clearance)
            print("contacts:", contacts[:5])
            print("Contact TCP:", np.round(contact_tcp_pos, 3))
            print("Contact door point:", np.round(push_point, 3))
            print("Contact door angle:", theta_contact_ref)
            contact_found = True
            break
        if step > 0 and step % 4 == 0:
            clearance = max(clearance - 0.002, -0.060)
    if not contact_found:
        raise RuntimeError("Could not establish physical contact with the flat door surface.")
    print("\n===== Closed-loop FLAT-SURFACE pushing =====")
    print("The handle is no longer tracked or recovered.")
    print("Sliding on the door surface is allowed.")
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
        tcp_now, rot_now = site_pose(base_env.sim, active_tcp_site)
        if finish_mode and finish_rot is None:
            finish_rot = rot_now.copy()
            print("\nEntering finish mode: freeze orientation, keep pushing the door surface.")
        target_rot = finish_rot if finish_mode else door_facing_rotation(
            hinge_axis, tangent, T_active_tcp_ref
        )
        # Where the originally measured physical surface contact should be now.
        door_delta = theta - theta_contact_ref
        Rdoor = Rotation.from_rotvec(hinge_axis * door_delta).as_matrix()
        desired_surface_offset = Rdoor @ surface_offset_ref
        surface_goal = push_point + desired_surface_offset
        # Correct sliding ALONG the door surface, but only gently.
        surface_error = surface_goal - tcp_now
        surface_correction = surface_error - np.dot(surface_error, tangent) * tangent
        corr_norm = np.linalg.norm(surface_correction)
        if corr_norm > 0.002:
            surface_correction *= 0.002 / corr_norm
        # Maintain persistent penetration into the door.
        # This is what gives us continued pushing pressure instead of just
        # commanding another tiny step from the current TCP.
        normal_error = float(np.dot(surface_error, tangent))
        normal_command = normal_error + args.push_lead
        target = tcp_now + surface_correction + normal_command * tangent
        actual, err = execute_target(
            env, base_env, robot, active_arm, active_tcp_site, target, target_rot,
            hold_commands, T_active_tcp_ref, args.gripper, repeats=1,
        )
        theta_after = float(d.qpos[door_qadr])
        push_point_after = door_point(hinge_pos, hinge_axis, door_push_start, theta_after, theta_start)
        contacts = door_surface_contacts(base_env, fxtr.name, push_point_after, args.near_handle_contact_radius)
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
                f"push {step:4d}", f"door={theta_after:.3f}", f"remaining={theta_closed - theta_after:.3f}",
                #f"step={push_step:.4f}", f"tan_err={tangential_error:.3f}", f"err={err:.3f}",
                f"contact={in_contact}", f"mode={mode}", f"finish_mode={finish_mode}", f"lost={lost_contact_steps}",
            )
            print("   surface_offset=", np.round(actual_surface_offset, 3), "target=", np.round(target, 3), "actual=", np.round(actual, 3))
        # IMPORTANT: no handle recovery. If contact disappears, keep advancing in
        # the current closing tangent; the broad door surface can be re-contacted.
        last_target, last_rot = target.copy(), target_rot.copy()
    print("\n===== Result =====")
    print("success:", base_env._check_success())
    print("door final angle:", float(d.qpos[door_qadr]))
    print("door target angle:", float(theta_closed))
    if args.record_dir and hasattr(env, "ep_directory"):
        print("raw demo:", env.ep_directory)
    if args.keep_open:
        print("\nPress Ctrl+C to finish.")
        try:
            while True:
                execute_target(env, base_env, robot, active_arm, active_tcp_site, last_target, last_rot, hold_commands, T_active_tcp_ref, args.gripper)
        except KeyboardInterrupt:
            pass
    env.close()
    return success

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="CloseSingleDoor")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--arm", choices=["auto", "left", "right"], default="auto", help="Arm used for Cartesian door pushing; auto prefers left when available in WHOLE_BODY_IK actuation_part_names")
    parser.add_argument("--layout", type=int, default=1)
    parser.add_argument("--style", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="RoboCasa / NumPy episode seed")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--standoff", type=float, default=0.10)
    parser.add_argument("--push-lead", type=float, default=0.006, help="Normal incremental push step cap [m]")
    parser.add_argument("--via-z", type=float, default=0.12)
    parser.add_argument("--approach-points", type=int, default=80)
    parser.add_argument("--steps-per-waypoint", type=int, default=2)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--gripper", type=float, default=1.0)
    parser.add_argument("--no-fixed-placement", action="store_true")
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--robot-x", type=float, default=0.55)
    parser.add_argument("--robot-y", type=float, default=-1.95)
    parser.add_argument("--robot-z", type=float, default=0.0)
    parser.add_argument("--robot-yaw", type=float, default=np.pi / 2)
    parser.add_argument("--push-x-direction", type=int, choices=[-1, 0, 1], default=0)
    parser.add_argument("--contact-z-offset", type=float, default=0.0)
    parser.add_argument("--progress-window", type=int, default=10)
    parser.add_argument("--lost-contact-penetration-step", type=float, default=0.001, help="Incremental final push step [m]")
    parser.add_argument("--near-handle-contact-radius", type=float, default=0.10, help="Door-contact detection radius around flat push point [m]")
    parser.add_argument("--finish-angle", type=float, default=0.12)
    parser.add_argument("--close-tolerance", type=float, default=0.020)
    parser.add_argument("--max-push-steps", type=int, default=5000)
    parser.add_argument("--handle-clearance", type=float, default=0.053, help="Initial clearance for flat-door contact search [m]")
    parser.add_argument("--recovery-step", type=float, default=0.004, help="Normal incremental push step limit [m]")
    args = parser.parse_args()
    raise SystemExit(0 if run(args) else 2)
