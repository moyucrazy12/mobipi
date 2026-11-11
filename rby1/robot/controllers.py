"""Existing RB-Y1 WholeBody controller and TCP/reference-frame adapters."""
import numpy as np
from scipy.spatial.transform import Rotation
from robosuite.controllers.composite.composite_controller import WholeBody

from rby1.robot.sim_utils import (
    raw_model, raw_data, site_pose, make_T, arm_joint_names,
)


def pose6(position, rotation_matrix):
    rotvec = Rotation.from_matrix(rotation_matrix).as_rotvec()
    return np.concatenate([position, rotvec])

def set_ik_q0_from_current_pose(base_env, robot):
    if not isinstance(robot.composite_controller, WholeBody):
        return
    ik = robot.composite_controller.joint_action_policy
    ik.q0 = np.array([float(base_env.sim.data.get_joint_qpos(name)) for name in ik.joint_names])
    print('\nIK q0:', np.round(ik.q0, 3))

def pose_target_for_ref(tcp_pos, tcp_rot, T_tcp_ref):
    T_world_ref = make_T(tcp_pos, tcp_rot) @ T_tcp_ref
    return pose6(T_world_ref[:3, 3], T_world_ref[:3, :3])

def arm_tcp_site(arm):
    return f"gripper0_{arm}_grip_site"

def arm_default_ref_site(arm):
    return f"gripper0_{arm}_ft_frame"

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
