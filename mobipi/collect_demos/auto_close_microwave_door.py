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

# Keep these utility functions import-compatible with the drawer collector.
# Microwave source motion is implemented once in the general entrypoint.
if __package__:
    from .auto_close_door_general import run, parser, main
else:
    from auto_close_door_general import run, parser, main


if __name__ == "__main__":
    main()
