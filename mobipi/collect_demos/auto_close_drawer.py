#!/usr/bin/env python3
"""
Scripted RB-Y1 demo for CloseDrawerLeft: push the open drawer shut with the left arm.

The robot starts at RoboCasa's drawer placement, always to the right of the
drawer (mobipi.envs.CloseDrawerLeft), squatting with the head turned toward
the drawer (the env resets into rby1_utils.TORSO_POSES / HEAD_POSES). The left
gripper approaches the drawer front near the edge closest to the robot, finds
contact, and pushes along the drawer's slide axis until the task succeeds.
--base-pos-noise / --base-yaw-noise and --drawer-open-min / -max randomize the
robot placement and the initial drawer opening. With --record-dir the
environment kwargs are saved to env_kwargs.json next to the recording.

Exit codes: 0 success, 2 failure, 3 scene skipped (no left placement, or the
start pose collides with a fixture).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import robosuite
from scipy.spatial.transform import Rotation
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import DataCollectionWrapper

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import mobipi.envs  # noqa: F401  registers CloseDrawerLeft
from mobipi.utils import rby1_utils
from auto_close_microwave_door import (
    DEFAULT_CONTROLLER,
    build_controller_interface,
    door_facing_rotation,
    door_surface_contacts,
    execute_target,
    interpolate_rotation,
    lerp_points,
    normalize,
    raw_data,
    raw_model,
    rotation_error,
    set_ik_q0_from_current_pose,
    site_pose,
    unwrap_env,
)

EXIT_SUCCESS, EXIT_FAILURE, EXIT_SKIPPED = 0, 2, 3
UP = np.array([0.0, 0.0, 1.0])


def drawer_geometry(base_env, push_edge_inset):
    """Slide joint, closing direction, and the push point on the drawer front."""
    sim = base_env.sim
    m, d = raw_model(sim), raw_data(sim)
    drawer = base_env.drawer
    jid = m.joint(f"{drawer.name}_slidejoint").id
    qadr = int(m.jnt_qposadr[jid])
    q_open = float(d.qpos[qadr])
    # closed is q = 0; the drawer moves along the joint axis as q changes
    close_dir = normalize(np.asarray(d.xaxis[jid]) * np.sign(0.0 - q_open))
    handle = sim.data.get_geom_xpos(drawer.handle_name).copy()
    # horizontal direction along the drawer front, toward the robot
    to_robot = rby1_utils.get_base_vec(sim)[:2] - handle[:2]
    lateral = np.r_[to_robot, 0.0]
    lateral -= np.dot(lateral, close_dir) * close_dir
    lateral -= np.dot(lateral, UP) * UP
    lateral = normalize(lateral)
    push_point = handle + (drawer.width / 2 - push_edge_inset) * lateral
    return drawer, qadr, q_open, close_dir, handle, push_point


def run(args):
    controller_config = load_composite_controller_config(controller=args.controller, robot="RBY1")
    config = {
        "env_name": args.environment,
        "robots": "RBY1",
        "controller_configs": controller_config,
        "layout_ids": [args.layout],
        "style_ids": [args.style],
        "has_renderer": not args.no_render,
        "has_offscreen_renderer": False,
        "render_camera": None,
        "ignore_done": True,
        "use_camera_obs": False,
        "control_freq": args.control_freq,
        "renderer": "mjviewer",
        "translucent_robot": False,
        "initialization_noise": None,
        "randomize_cameras": False,
        "seed": args.seed,
        "drawer_open_range": [args.drawer_open_min, args.drawer_open_max],
        "robot_offset": [args.robot_side_gap, args.robot_front_offset],
    }
    if args.base_pos_noise > 0 or args.base_yaw_noise > 0:
        config["randomize_base_init_pose"] = [args.base_pos_noise, args.base_pos_noise, np.deg2rad(args.base_yaw_noise)]
    np.random.seed(args.seed)
    # CloseDrawerLeft raises when no sampled scene has room to the right of the
    # drawer, both while loading the model in make() and on reset()
    try:
        env = robosuite.make(**config)
        if args.record_dir:
            os.makedirs(args.record_dir, exist_ok=True)
            # gather_demos_hdf5.py recreates the environment from these
            with open(os.path.join(args.record_dir, "env_kwargs.json"), "w") as f:
                json.dump(config, f, indent=4)
            env = DataCollectionWrapper(env, args.record_dir)
            print("Recording raw trajectory to:", args.record_dir)
        env.reset()
    except RuntimeError as e:
        print("SKIP:", e)
        return EXIT_SKIPPED
    base_env = unwrap_env(env)
    robot = base_env.robots[0]

    # The env resets into the task's torso/head pose. An override is applied
    # after the reset, so the recorded initial state still has the default one.
    if args.torso_squat is not None:
        rby1_utils.set_task_pose(base_env, args.environment, torso_pose=rby1_utils.squat_torso_pose(args.torso_squat))
    contacts = rby1_utils.robot_env_contacts(base_env.sim)
    if contacts:
        print("SKIP: start pose collides with the scene:", contacts[:3])
        env.close()
        return EXIT_SKIPPED

    head_pose = [base_env.sim.data.get_joint_qpos(j) for j in rby1_utils.HEAD_JOINTS]
    set_ik_q0_from_current_pose(base_env, robot)
    active_arm, tcp_site, _, start_pos, start_rot, T_tcp_ref, hold_commands = build_controller_interface(
        base_env, robot, controller_config, args.arm
    )
    # Hold head and torso explicitly: parts missing from the action are commanded
    # to zero, which for absolute joint-position control means the default pose.
    hold_commands["head"] = np.asarray(head_pose, dtype=float)
    torso = robot.composite_controller.part_controllers.get("torso")
    if torso is not None:
        hold_commands["torso"] = np.array([base_env.sim.data.get_joint_qpos(n) for n in torso.joint_names])

    drawer, qadr, q_open, close_dir, handle, push_point = drawer_geometry(base_env, args.push_edge_inset)
    # Gripper axis along the closing direction, optionally pitched down about the
    # horizontal axis along the drawer front
    pitch_axis = normalize(np.cross(close_dir, UP))
    contact_rot = Rotation.from_rotvec(np.deg2rad(args.tool_pitch) * pitch_axis).as_matrix() @ door_facing_rotation(
        UP, close_dir, T_tcp_ref
    )
    print("\n===== Drawer =====")
    print("drawer:         ", drawer.name, f"(width {drawer.width:.2f} m)", base_env.get_ep_meta()["lang"])
    print("slide qpos:     ", round(q_open, 4), "-> 0")
    print("close direction:", np.round(close_dir, 3))
    print("handle:         ", np.round(handle, 3))
    print("push point:     ", np.round(push_point, 3), f"({args.push_edge_inset} m in from the robot-side edge)")

    # Approach from behind the drawer front (on the open side), passing above it
    push_point = push_point + np.array([0.0, 0.0, args.contact_z_offset])
    pre_contact = push_point - (args.standoff + args.clearance) * close_dir
    far_pre_contact = pre_contact - 0.10 * close_dir
    far_high = far_pre_contact + np.array([0.0, 0.0, args.via_z])
    pre_high = pre_contact + np.array([0.0, 0.0, args.via_z])
    trajectory = []
    points = lerp_points(start_pos, far_high, max(args.approach_points // 2, 2))
    for i, p in enumerate(points):
        trajectory.append((p, interpolate_rotation(start_rot, contact_rot, i / max(len(points) - 1, 1))))
    for p in lerp_points(far_high, pre_high, max(args.approach_points // 4, 2)):
        trajectory.append((p, contact_rot))
    for p in lerp_points(pre_high, pre_contact, max(args.approach_points // 4, 2)):
        trajectory.append((p, contact_rot))

    def step_to(target, rot, repeats=1):
        return execute_target(env, base_env, robot, active_arm, tcp_site, target, rot, hold_commands,
                              T_tcp_ref, args.gripper, repeats=repeats)

    d = raw_data(base_env.sim)
    print("\n===== Approach =====")
    for _ in range(args.settle_steps):
        step_to(start_pos, start_rot)
    for i, (target, rot) in enumerate(trajectory):
        actual, err = step_to(target, rot, repeats=args.steps_per_waypoint)
        if i % 10 == 0:
            print(f"approach {i:3d}/{len(trajectory)}", "target=", np.round(target, 3), "actual=", np.round(actual, 3),
                  f"err={err:.3f}", f"drawer={float(d.qpos[qadr]):.3f}")
    for step in range(100):
        actual, err = step_to(pre_contact, contact_rot)
        ang_err = rotation_error(site_pose(base_env.sim, tcp_site)[1], contact_rot)
        if err < 0.03 and ang_err < np.deg2rad(8):
            break
    print(f"aligned: pos_err={err:.3f} rot_err={np.degrees(ang_err):.1f} deg")

    print("\n===== Contact search =====")
    contact_found = False
    clearance = args.clearance
    for step in range(400):
        target = push_point - clearance * close_dir
        actual, err = step_to(target, contact_rot, repeats=2)
        contacts = door_surface_contacts(base_env, drawer.name, push_point, args.contact_radius)
        if step % 10 == 0:
            print(f"search {step:3d} clearance={clearance:.3f} err={err:.3f} drawer={float(d.qpos[qadr]):.3f} contacts={contacts[:2]}")
        if contacts:
            print("contact:", contacts[:3])
            contact_found = True
            break
        clearance = max(clearance - 0.002, -0.10)
    if not contact_found:
        print("No contact with the drawer front.")
        env.close()
        return EXIT_FAILURE

    print("\n===== Push =====")
    success = False
    for step in range(args.max_push_steps):
        if base_env._check_success():
            success = True
            break
        tcp, _ = site_pose(base_env.sim, tcp_site)
        # stay on the line through the push point, a little ahead of the gripper
        along = float(np.dot(tcp - push_point, close_dir))
        target = push_point + (along + args.push_lead) * close_dir
        actual, err = step_to(target, contact_rot)
        if step % 20 == 0:
            blocking = [c[1] for c in rby1_utils.robot_env_contacts(base_env.sim)]
            print(f"push {step:4d} drawer={float(d.qpos[qadr]):.3f} along={along:.3f} err={err:.3f} contacts={blocking[:3]}")
    print("\n===== Result =====")
    print("success:", success)
    print("drawer final qpos:", round(float(d.qpos[qadr]), 4))
    if args.record_dir and hasattr(env, "ep_directory"):
        print("raw demo:", env.ep_directory)
    if args.keep_open:
        print("\nPress Ctrl+C to finish.")
        tcp, rot = site_pose(base_env.sim, tcp_site)
        try:
            while True:
                step_to(tcp, rot)
        except KeyboardInterrupt:
            pass
    env.close()
    return EXIT_SUCCESS if success else EXIT_FAILURE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--environment", default="CloseDrawerLeft", choices=["CloseDrawerLeft"])
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER)
    parser.add_argument("--arm", choices=["left"], default="left")
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--style", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="RoboCasa / NumPy episode seed")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--base-pos-noise", type=float, default=0.0, help="Std of the robot base x / y placement noise [m]")
    parser.add_argument("--base-yaw-noise", type=float, default=0.0, help="Std of the robot base yaw placement noise [deg]")
    parser.add_argument("--robot-side-gap", type=float, default=0.3, help="Gap between the drawer's right edge and the robot [m]")
    parser.add_argument("--robot-front-offset", type=float, default=-0.23, help="Robot offset along the drawer's front direction [m]")
    parser.add_argument("--drawer-open-min", type=float, default=0.9, help="Min initial drawer opening (fraction of fully open)")
    parser.add_argument("--drawer-open-max", type=float, default=1.0, help="Max initial drawer opening (fraction of fully open)")
    parser.add_argument("--push-edge-inset", type=float, default=0.10, help="Push point distance from the drawer's robot-side edge [m]")
    parser.add_argument("--contact-z-offset", type=float, default=0.0, help="Push point height relative to the handle [m]")
    parser.add_argument("--torso-squat", type=float, default=None, help="Override the task's torso pose with a squat of this angle [rad]")
    parser.add_argument("--tool-pitch", type=float, default=0.0, help="Gripper pitch below horizontal [deg]")
    parser.add_argument("--standoff", type=float, default=0.08, help="Pre-contact distance in front of the drawer face [m]")
    parser.add_argument("--clearance", type=float, default=0.06, help="Initial clearance for the contact search [m]")
    parser.add_argument("--via-z", type=float, default=0.15, help="Height above the push point while approaching [m]")
    parser.add_argument("--approach-points", type=int, default=80)
    parser.add_argument("--steps-per-waypoint", type=int, default=4)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--push-lead", type=float, default=0.03, help="How far ahead of the gripper the push target is [m]")
    parser.add_argument("--contact-radius", type=float, default=0.15, help="Drawer-contact detection radius around the push point [m]")
    parser.add_argument("--max-push-steps", type=int, default=1500)
    parser.add_argument("--gripper", type=float, default=1.0)
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--no-render", action="store_true", help="Run without the MuJoCo viewer")
    raise SystemExit(run(parser.parse_args()))
