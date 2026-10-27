r"""Select a base pose with Mobi-pi, navigate RB-Y1 there, then run its policy.

Run from the repository root in an environment with the policy and mobilization
dependencies installed. The dataset must contain RGB observations and a train
mask; the scene directory must contain pc.ply and a trained splatfacto model.
--task selects the task module (mobipi/pipeline/tasks) and its default
checkpoint and dataset.

    python -m mobipi.scripts.run_rby1_pipeline --task close_microwave \
        --scene-dir scene_data/close_single_door/layout1_style1 \
        --skip-detection --vis
"""

import argparse
import copy
import importlib.util
import json
from pathlib import Path
import time

from mobipi.pipeline.tasks import TASKS, get_task

ROOT = Path(__file__).resolve().parents[2]


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--task", choices=sorted(TASKS), default="close_microwave")
    result.add_argument("--checkpoint", type=Path, default=None, help="Policy checkpoint (default: the task's)")
    result.add_argument("--dataset", type=Path, default=None,
                        help="RGB demonstration HDF5 used for DINO scoring (default: the task's)")
    result.add_argument("--scene-dir", type=Path, required=True, help="Scene folder containing pc.ply and model/")
    result.add_argument("--filter-key", default="train")
    result.add_argument("--layout-id", type=int, default=1)
    result.add_argument("--style-id", type=int, default=1)
    result.add_argument("--seed", type=int, default=1)
    result.add_argument("--camera-names", nargs="+", default=["robot0_head_camera"])
    result.add_argument("--scene-model-step", type=int, default=29999)
    result.add_argument("--num-init-samples", type=int, default=2500)
    result.add_argument("--bo-num-samples", type=int, default=500, help="Additional BO candidates, in batches of five")
    result.add_argument("--k", type=int, default=5)
    result.add_argument("--skip-detection", action="store_true")
    result.add_argument("--collision-radius-scale", type=float, default=None,
                        help="Scoring collision radius as a fraction of the robot radius (default: the task's)")
    result.add_argument("--no-scene-overrides", action="store_true",
                        help="Keep the dataset's scene settings instead of matching the 3DGS capture "
                             "(e.g. which drawer is open)")
    result.add_argument("--nav-mode", choices=("velocity", "kinematic"), default="velocity")
    result.add_argument("--policy-device", choices=("cuda", "cpu"), default="cuda")
    result.add_argument("--policy-steps", type=int, default=2000, help="Maximum learned-policy steps before timeout")
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--vis", action="store_true", help="Show the simulator during navigation and manipulation")
    return result


def apply_task_defaults(args):
    """Fills the task-dependent defaults into parsed arguments and returns the task."""
    task = get_task(args.task)
    args.checkpoint = args.checkpoint or task.default_checkpoint
    args.dataset = args.dataset or task.default_dataset
    if args.collision_radius_scale is None:
        args.collision_radius_scale = task.collision_threshold_multiplier
    return task


def make_env(metadata, args, task, overrides=None):
    """Use the same mobile RB-Y1 scene setup as test_mobilization.py."""
    import robosuite
    import robocasa
    import mobipi.envs  # registers the Mobi-pi task variants (e.g. CloseDrawerLeft)

    kwargs = copy.deepcopy(metadata["env_kwargs"])
    layouts = kwargs.pop("layout_ids", []) or []
    kwargs.pop("style_ids", None)
    in_training_layout = args.layout_id in layouts
    if not in_training_layout:
        kwargs.pop("force_robot_placement", None)
    task.env_overrides(kwargs, in_training_layout)
    kwargs.update(
        base_types="RBY1MobileBase", layout_and_style_ids=[[args.layout_id, args.style_id]],
        has_renderer=args.vis, has_offscreen_renderer=True, use_camera_obs=False,
        ignore_done=True, renderer="mjviewer" if args.vis else "mujoco", render_camera=None, seed=args.seed,
        **(overrides or {}),
    )
    return robosuite.make(metadata["env_name"], **kwargs)


def select_base_pose(env, args, scene_run, fixture_bounds, floor_bounds, robot_size, default_pose, output):
    """Reuse the 3DGS/DINO score, Bayesian optimizer, and initial-pose sampler."""
    import numpy as np
    import torch
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
    from mobipi.scene_model.scene_model import BatchSceneModel
    from mobipi.utils.encoder_utils import DinoDenseDescriptorEncoder
    from mobipi.utils.env_utils import compute_camera_extrinsics
    from mobipi.utils.opt_utils import optimize_pose_batch
    from mobipi.utils.score_utils import HybridDistribution
    from mobipi.utils import rby1_utils

    encoder = DinoDenseDescriptorEncoder(device="cuda", model_name="vit_base_patch16_224_dino")
    config = {"train": {"data": [{"path": str(args.dataset), "filter_key": args.filter_key}]}}
    distribution = HybridDistribution(
        config, encoder, k=args.k, object_detection_model_name="openbmb/MiniCPM-V-2",
        ply_path=str(args.scene_dir / "pc.ply"), floor_fixture_bounds_2d=floor_bounds,
        robot_size=robot_size, max_z=1.0, image_size=224, skip_detection=args.skip_detection,
        collision_threshold_multiplier=args.collision_radius_scale,
    )
    distribution.set_obs_keys([name + "_image" for name in args.camera_names])
    distribution.set_object(env.get_object_name())
    distribution.set_task_name(env.get_ep_meta()["lang"])

    intrinsics, relative_positions, relative_rotations, images, masks = [], [], [], [], []
    for name in args.camera_names:
        matrix = get_camera_intrinsic_matrix(env.sim, name, camera_height=224, camera_width=224)
        intrinsics.append(dict(w=224, h=224, fl_x=matrix[0, 0], fl_y=matrix[1, 1], cx=matrix[0, 2], cy=matrix[1, 2]))
        position, rotation, image, mask = rby1_utils.render_image_with_robot_mask(env.sim, name)
        relative_positions.append(torch.as_tensor(position, dtype=torch.float32, device="cuda"))
        relative_rotations.append(torch.as_tensor(rotation, dtype=torch.float32, device="cuda"))
        images.append(image)
        masks.append(mask)
    robot_images = torch.as_tensor(np.asarray(images) / 255, dtype=torch.float32, device="cuda")
    robot_masks = torch.as_tensor(np.asarray(masks)[..., None], dtype=torch.float32, device="cuda")
    scene_model = BatchSceneModel(str(scene_run), intrinsics)
    initial_pose = rby1_utils.sample_nav_init_pose(
        env, fixture_bounds, floor_bounds, default_pose, np.random.default_rng(args.seed * 10000),
        k_col=distribution.k_col,
    )

    def render(pose):
        with torch.no_grad():
            pose = torch.as_tensor(pose, device="cuda", dtype=torch.float32)
            cameras = [compute_camera_extrinsics(pose, p, r) for p, r in zip(relative_positions, relative_rotations)]
            rendered = scene_model.render(cameras, image_size=224)
            return rendered, rendered * (1 - robot_masks) + robot_images * robot_masks

    def score(poses):
        values, _, _ = distribution.compute_score(
            [lambda pose=pose: render(pose) for pose in poses],
            torch.as_tensor(np.asarray(poses), device="cuda", dtype=torch.float32),
            env=env, base_pos=np.r_[initial_pose[:2], 0.0], base_heading=initial_pose[2],
            check_fov=np.pi * 75 / 180,
        )
        # BO needs scores and optional plotting data; avoid retaining every RGB render.
        return values.detach().cpu().tolist(), {}

    bounds = [
        (floor_bounds[2, 0], floor_bounds[0, 0]),
        (floor_bounds[0, 1], floor_bounds[1, 1]),
        (initial_pose[2] - np.pi / 2, initial_pose[2] + np.pi / 2),
    ]
    initial_samples = np.random.RandomState(args.seed * 10000).uniform(0, 1, (args.num_init_samples, 3))
    selected_pose, history = optimize_pose_batch(
        score, bounds, algorithm="bayesian", base_estimator="ET", n_iterations=args.bo_num_samples // 5,
        initial_samples=initial_samples, normalize_data=True, track_info=True, batch_size=5,
        seed=args.seed * 10000, acq_func="LCB", acq_func_kwargs=dict(kappa=1.96),
    )
    selected_pose[2] = rby1_utils.angle_wrap(selected_pose[2])
    scores = np.asarray(history["sampled_scores"])
    np.savez(output / "mobilization.npz", poses=history["sampled_points"], scores=scores)
    if not np.isfinite(scores).all() or scores.max() <= 0:
        raise RuntimeError("Mobilization found no valid pose with a positive image score")
    print(f"Selected base pose {selected_pose.round(3)}; score={scores.max():.4f}", flush=True)
    # The large scene and image models are released on return, before policy loading.
    return initial_pose, selected_pose, distribution.k_col, float(scores.max())


def run_policy(env, policy, task, seed, max_steps, vis=False, progress=None):
    """Start a fresh policy history at the reached pose, without resetting the scene."""
    import gc
    import numpy as np
    import torch
    import mujoco
    from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter
    from robomimic.scripts.evaluate_rby1_cartesian_checkpoints import execute_loop, observation_reader

    # Flush deferred cleanup from earlier simulator instances, then rebuild the
    # camera buffers. Rebinding alone cannot repair buffers invalidated by cleanup.
    gc.collect()
    context = env.sim._render_context_offscreen
    context.gl_ctx.make_current()
    context.con.free()
    context._set_mujoco_context_and_buffers()

    for controller in env.robots[0].composite_controller.part_controllers.values():
        controller.update(force=True)
        controller.reset_goal()
    adapter = RBY1CartesianAdapter(env)
    task_progress = task.progress_fn(env)
    base_start = adapter.site_transform(adapter.base_site).copy()
    policy.policy.inference_noise_generator = torch.Generator(device="cpu").manual_seed(seed)
    policy.start_episode()
    started = last_frame = time.perf_counter()
    done_steps = 0
    required_done_steps = max(1, round(0.5 * env.control_freq))
    if progress is None:
        progress = {}
    initial = task_progress()
    progress.update(completed=False, success=False, steps_executed=0, max_steps=max_steps,
                    initial_progress=initial, final_progress=initial, success_timestep=None)
    print(f"Running up to {max_steps} policy steps ({max_steps / env.control_freq:.0f} seconds of simulation), "
          "stopping once the task stays done for 0.5 seconds. Viewer playback is capped at real time.", flush=True)

    def observe(step, progress_after, **unused):
        nonlocal last_frame, done_steps
        progress.update(steps_executed=step + 1, final_progress=float(progress_after),
                        success=bool(progress_after >= task.success_threshold))
        if progress["success"] and progress["success_timestep"] is None:
            progress["success_timestep"] = step + 1
        done_steps = done_steps + 1 if progress["success"] else 0
        if vis:
            if not env.viewer.viewer.is_running():
                raise KeyboardInterrupt
            delay = env.control_timestep - (time.perf_counter() - last_frame)
            if delay > 0:
                time.sleep(delay)
            last_frame = time.perf_counter()
        if (step + 1) % 100 == 0:
            print(f"Policy step {step + 1}/{max_steps}: {task.progress_label}={progress_after:.3f}, "
                  f"done={progress['success']}, elapsed={time.perf_counter() - started:.1f}s", flush=True)
        if done_steps >= required_done_steps:
            print(f"Task done for 0.5 seconds at policy step {step + 1}; stopping learned actions.", flush=True)
            return False

    native_read = observation_reader(env, adapter)

    def read_observation():
        # 3DGS and the interactive viewer can leave another GL context current.
        # Robosuite's render method does not rebind its offscreen context itself.
        context = env.sim._render_context_offscreen
        context.gl_ctx.make_current()
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, context.con)
        return native_read()

    result = execute_loop(policy, env, adapter, read_observation, None, max_steps, observer=observe,
                          progress_fn=task_progress, success_threshold=task.success_threshold)
    result["closure_confirmed"] = done_steps >= required_done_steps
    result["success"] = bool(result["success"] and result["closure_confirmed"])
    if not result["closure_confirmed"] and result["failure_classification"] == "success":
        result["failure_classification"] = "closure_not_stable"
    result["base_drift_m"] = float(np.linalg.norm(adapter.site_transform(adapter.base_site)[:2, 3] - base_start[:2, 3]))
    return result


def return_arm_to_start(env, task, home_qpos, grippers, vis=False, progress=None):
    """Use the existing joint-position servos to return the left arm smoothly."""
    import numpy as np
    from mobipi.utils import rby1_utils

    controller = env.robots[0].composite_controller
    left = controller.part_controllers["left"]
    ik = controller.joint_action_policy
    ik_joints = list(ik.joint_names)
    ik_parts = [part for part in controller.part_controllers.values()
                if set(getattr(part, "joint_names", ())) & set(ik_joints)]
    if (not set(left.joint_names) <= set(ik_joints)
            or any(part.input_type != "absolute" for part in ik_parts)):
        raise ValueError("Arm return expects IK over the left arm and absolute joint controllers")
    task_progress = task.progress_fn(env)
    if task_progress() < task.success_threshold:
        raise ValueError("Arm return requires a completed task")
    start = env.sim.data.qpos[left.qpos_index].copy()
    home_qpos = np.asarray(home_qpos, dtype=float)
    # Cubic interpolation has peak slope 1.5; limit joint targets to 0.5 rad/s.
    duration = max(3.0, 1.5 * float(np.max(np.abs(home_qpos - start))) / 0.5)
    move_steps = int(np.ceil(duration * env.control_freq))
    settle_steps = max(1, round(env.control_freq))
    hold = rby1_utils.hold_action(env, grippers)
    base_start = rby1_utils.get_base_vec(env.sim)
    original_solve = ik.solve
    target = start.copy()
    # IK over both arms (e.g. the drawer checkpoint): hold the other joints where they are.
    ik_target = env.sim.data.qpos[[env.sim.model.get_joint_qpos_addr(j) for j in ik_joints]].copy()
    left_slots = [ik_joints.index(j) for j in left.joint_names]
    if progress is None:
        progress = {}
    progress.update(completed=False, success=False, steps_executed=0, home_qpos=home_qpos.copy(),
                    task_stayed_done=True, num_contact_steps=0)

    # Supply joint targets to the same low-level servos instead of Cartesian IK
    # during this phase. env.step still advances normal physics and base holds.
    def solve(unused):
        ik_target[left_slots] = target
        return ik_target.copy()

    ik.solve = solve
    try:
        for step in range(move_steps + settle_steps):
            frame_start = time.perf_counter()
            u = min((step + 1) / move_steps, 1.0)
            target = start + (3 * u * u - 2 * u * u * u) * (home_qpos - start)
            env.step(hold)
            if not np.isfinite(env.sim.data.qpos).all():
                raise RuntimeError("Nonfinite simulator state during arm return")
            value = task_progress()
            progress.update(steps_executed=step + 1, final_progress=value,
                            final_qpos=env.sim.data.qpos[left.qpos_index].copy())
            progress["num_contact_steps"] += int(bool(rby1_utils.robot_env_contacts(env.sim)))
            if value < task.success_threshold:
                progress["task_stayed_done"] = False
                progress["stop_reason"] = "Task undone during arm return"
                break
            if vis:
                if not env.viewer.viewer.is_running():
                    raise KeyboardInterrupt
                time.sleep(max(0.0, env.control_timestep - (time.perf_counter() - frame_start)))
    finally:
        ik.solve = original_solve

    error = float(np.max(np.abs(env.sim.data.qpos[left.qpos_index] - home_qpos)))
    progress.update(completed=True, max_joint_error_rad=error,
                    base_drift_m=float(np.linalg.norm(rby1_utils.get_base_vec(env.sim)[:2] - base_start[:2])),
                    final_contacts=rby1_utils.robot_env_contacts(env.sim))
    progress["success"] = bool(progress["task_stayed_done"] and error <= 0.02 and not progress["final_contacts"])
    print(f"Arm return: max joint error={error:.4f} rad, task still done={progress['task_stayed_done']}", flush=True)
    return progress


def navigate_to_pose(env, target, collision, floor_bounds, grippers, mode):
    """Drive to the goal through a staging pose behind it (rby1_utils.approach_pose).

    The robot turns to the final heading in open space and ends with a straight
    drive in, so it does not sweep its torso into an open door or drawer. The
    collision checks stay active on the whole drive.
    """
    from mobipi.utils import rby1_utils

    return rby1_utils.move_to_pose(
        env, target, collision, floor_bounds, render_camera=None, gripper_action=grippers, mode=mode,
        use_approach=True,
    )


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    task = apply_task_defaults(args)
    if not args.collision_radius_scale > 0:
        arguments.error("collision-radius-scale must be positive")
    if (args.policy_steps < 1 or args.k < 1 or args.num_init_samples < 5
            or args.num_init_samples % 5 or args.bo_num_samples < 0 or args.bo_num_samples % 5):
        arguments.error("Use positive policy-steps/k and sample counts divisible by five (at least five initial samples)")
    args.dataset, args.scene_dir, args.checkpoint = [p.expanduser().resolve() for p in (args.dataset, args.scene_dir, args.checkpoint)]
    for path in (args.dataset, args.checkpoint, args.scene_dir / "pc.ply"):
        if not path.is_file():
            arguments.error(f"Missing required input: {path}")
    checkpoints = sorted(args.scene_dir.glob(f"model/**/nerfstudio_models/step-{args.scene_model_step:09d}.ckpt"))
    if not checkpoints:
        arguments.error(f"No scene checkpoint for step {args.scene_model_step} under {args.scene_dir / 'model'}")
    scene_run = checkpoints[-1].parent.parent
    dependencies = ("nerfstudio", "gsplat", "timm", "open3d", "skopt", "sklearn", "plotly", "cma", "skvideo")
    missing = [name for name in dependencies if importlib.util.find_spec(name) is None]
    if missing:
        arguments.error("Mobilization dependencies are missing from this environment: " + ", ".join(missing))

    import h5py
    import numpy as np
    import torch
    from robomimic.utils import file_utils as F
    from robomimic.scripts.evaluate_rby1_cartesian_checkpoints import check_policy, seed_rollout
    from mobipi.utils import rby1_utils
    from mobipi.utils.env_utils import get_env_map_and_default_robot_init_pose

    with h5py.File(args.dataset, "r") as dataset:
        if f"mask/{args.filter_key}" not in dataset:
            arguments.error(f"Dataset needs a nonempty mask/{args.filter_key} for scoring reference images")
        names = dataset[f"mask/{args.filter_key}"].asstr()[:]
        if len(names) < args.k:
            arguments.error("The reference split has fewer demonstrations than k")
        group = dataset["data/" + names[0]]
        for camera in args.camera_names:
            if f"obs/{camera}_image" not in group:
                arguments.error(f"Dataset has no RGB observations for scoring camera {camera}")
        metadata = json.loads(dataset["data"].attrs["env_args"])
        first_action = group["actions"][0].copy()
    if metadata["env_name"] != task.env_name or metadata["env_kwargs"]["robots"] != "RBY1":
        arguments.error(f"Task {task.name} expects an RB-Y1 {task.env_name} reference dataset, "
                        f"got {metadata['env_kwargs']['robots']} {metadata['env_name']}")
    if not torch.cuda.is_available():
        arguments.error("The 3DGS/DINO mobilization stage requires a CUDA-capable environment")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.policy_device == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    seed_rollout(args.seed)
    output = args.output_dir or ROOT / "logs/rby1_pipeline" / f"run_{time.time_ns()}"
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(str(args.checkpoint), map_location="cpu", weights_only=False)
    env = None
    result = dict(task=task.name, checkpoint=str(args.checkpoint), dataset=str(args.dataset), scene_dir=str(args.scene_dir),
                  layout_id=args.layout_id, style_id=args.style_id, seed=args.seed, success=False,
                  policy_device=args.policy_device, policy_steps=args.policy_steps, vis=args.vis,
                  collision_radius_scale=args.collision_radius_scale, stage="environment", policy=None)
    try:
        env = make_env(metadata, args, task)
        overrides = {} if args.no_scene_overrides else task.scene_overrides(env, args.scene_dir)
        if overrides:
            # Fixtures are chosen when the model is loaded, so the env is rebuilt.
            env.close()
            env = None
            try:
                env = make_env(metadata, args, task, overrides)
            except RuntimeError as error:
                raise RuntimeError(f"{task.env_name} cannot be set up as captured in {args.scene_dir} "
                                   f"({overrides}): {error}") from error
        result["scene_overrides"] = overrides
        env.reset()
        fixture_bounds, floor_bounds, _, _, robot_size = get_env_map_and_default_robot_init_pose(env=env)
        default_pose = rby1_utils.get_base_vec(env.sim)
        posture = rby1_utils.get_robot_joint_qpos(env)
        split = env.robots[0].composite_controller._whole_body_controller_action_split_indexes
        shift = len(first_action) - env.action_dim
        # Gripper slots follow the base slot, whose size changes from two wheels to SE(2).
        grippers = {part: first_action[start + shift:end + shift] for part, (start, end) in split.items() if part.endswith("gripper")}

        result["stage"] = "mobilization"
        print("Stage 1/3: select a manipulation base pose", flush=True)
        initial_pose, selected_pose, collision, best_score = select_base_pose(
            env, args, scene_run, fixture_bounds, floor_bounds, robot_size, default_pose, output,
        )
        result["stage"] = "navigation"
        print("Stage 2/3: navigate to the selected pose", flush=True)
        navigation = navigate_to_pose(env, selected_pose, collision, floor_bounds, grippers, args.nav_mode)
        navigation.pop("images")
        np.savez(output / "navigation.npz", poses=navigation.pop("base_vec_history"), waypoints=navigation.pop("waypoints"))
        reached = navigation["settled_vec"]
        position_error = float(np.linalg.norm(reached[:2] - selected_pose[:2]))
        heading_error = float(abs(rby1_utils.angle_wrap(reached[2] - selected_pose[2])))
        failures = []
        if navigation["stuck"]:
            failures.append("the base controller reported a stall")
        if navigation["num_collision_steps"] or navigation["final_contacts"]:
            failures.append(f"robot-environment contacts ({navigation['num_collision_steps']} navigation steps)")
        if position_error > 0.05 or heading_error > 0.05:
            failures.append(f"goal error {position_error:.3f} m / {heading_error:.3f} rad")
        nav_success = not failures
        result.update(default_pose=default_pose, initial_pose=initial_pose, selected_pose=selected_pose,
                      best_score=best_score, navigation=navigation, navigation_success=bool(nav_success),
                      navigation_position_error_m=position_error, navigation_heading_error_rad=heading_error)
        if nav_success:
            # Restore the carried arm posture only. The reached base and task state stay in place.
            rby1_utils.set_robot_joint_qpos(env, posture)
            env.sim.forward()
            del collision
            torch.cuda.empty_cache()
            result["stage"] = "policy_loading"
            policy, checkpoint = F.policy_from_checkpoint(ckpt_dict=checkpoint, device=torch.device(args.policy_device))
            check_policy(policy, checkpoint, metadata)
            del checkpoint
            result["stage"] = "policy"
            print("Stage 3/3: run the Cartesian manipulation policy", flush=True)
            left = env.robots[0].composite_controller.part_controllers["left"]
            home_qpos = env.sim.data.qpos[left.qpos_index].copy()
            result["policy"] = {}
            result["policy"] = run_policy(env, policy, task, args.seed, args.policy_steps, args.vis,
                                          progress=result["policy"])
            if result["policy"]["error"]:
                raise RuntimeError(f"{result['policy']['error_stage']}: {result['policy']['error']}")
            if result["policy"]["success"]:
                result["stage"] = "arm_return"
                print("Returning the left arm to its initial joint posture.", flush=True)
                result["arm_return"] = {}
                result["arm_return"] = return_arm_to_start(env, task, home_qpos, grippers, args.vis,
                                                          progress=result["arm_return"])
                result["success"] = bool(result["arm_return"]["success"])
        else:
            result["policy"] = None
            result["stop_reason"] = "Navigation rejected: " + "; ".join(failures)
    except KeyboardInterrupt:
        result["interrupted"] = True
        result["stop_reason"] = f"Interrupted during {result['stage']}"
        if result["stage"] == "policy" and result["policy"]:
            progress = result["policy"]
            progress["interrupted"] = True
            result["stop_reason"] += f" at step {progress['steps_executed']}/{args.policy_steps}"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        if env is not None:
            env.close()
        text = json.dumps(result, indent=2, default=lambda value: value.tolist())
        (output / "result.json").write_text(text + "\n")
        print(text, flush=True)
        print(f"Results: {output}", flush=True)


if __name__ == "__main__":
    main()
