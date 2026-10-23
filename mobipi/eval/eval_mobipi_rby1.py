"""
Full Mobi-pi pipeline on RB-Y1: mobilization followed by a policy rollout.

The mobilization stage is the one of `test_mobilization.py`: from a random
initial base pose, candidate base poses are scored against the initial head
camera frames of the demonstration dataset (3DGS renders + DINO kNN + collision
/ visibility checks), the best pose is picked with BO and the robot navigates
there. The Cartesian Diffusion Policy (robomimic checkpoint, e.g. CloseDrawerLeft)
is then rolled out from the reached pose, as in
`external/robomimic/robomimic/scripts/evaluate_rby1_rollouts.py`.

The environment is created from the checkpoint's metadata (training env_args,
with camera observations), so the task, controller, start arm / torso / head
pose and cameras match the policy's training data; only the layout and style
are overridden. Use the policy's training dataset for --dataset, so scoring
compares against the initial frames the policy was trained on.

The scene models are static captures of RoboCasa's task env, with the drawer
that env happened to open baked in. Unless --no_pin_scene_drawer is given, the
open drawer of the scene model is found in its point cloud and the env is set up
with that drawer as the target, opened as far as in the capture, so the renders
show the scene the robot is in. The default pose (the base pose right after the
reset, where every demo starts) is recomputed per episode.

Usage:
python eval_mobipi_rby1.py \
    --checkpoint ../../external/robomimic/experiment_results/rby1_close_drawer_cartesian_cmd_look300/20261020114821/models/model_epoch_10.pth \
    --dataset ../../datasets/close_drawer_look/close_drawer_look_300_im128_cmd.hdf5 \
    --layout_id 0 --style_id 0 --num_episodes 3 --skip_detection --vis
"""
import os
import sys
import copy
import json
import time
import click
import torch
import warnings
import numpy as np
from glob import glob
from datetime import datetime

import robocasa  # registers RoboCasa environments
import mobipi.envs  # registers the Mobi-pi task variants (e.g. CloseDrawerLeft)
import open3d as o3d
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix
from robocasa.models.fixtures.cabinets import Drawer

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.rby1_cartesian import RBY1CartesianAdapter

from mobipi.macros import *
from mobipi.utils.io_utils import DualStream, camel_to_snake_case
from mobipi.utils.env_utils import (
    get_env_map_and_default_robot_init_pose,
    compute_camera_extrinsics,
)
from mobipi.utils.media_utils import save_video
from mobipi.scene_model.scene_model import BatchSceneModel
from mobipi.utils.encoder_utils import DinoDenseDescriptorEncoder, DinoEncoder
from mobipi.utils.score_utils import HybridDistribution
from mobipi.utils.opt_utils import optimize_pose_batch
from mobipi.utils.nav_utils import angle_wrap
from mobipi.utils.vis_utils import (
    plot_best_metrics,
    plot_best_renders,
    plot_pose_histograms,
    plot_scores_on_topdown_image,
)
from mobipi.utils import rby1_utils
from test_mobilization import (
    load_dataset_info,
    gripper_action_from_dataset,
    get_topdown_image,
    pose_error,
)

warnings.filterwarnings("ignore", message=".*default value of the antialias parameter.*")

VIDEO_CAMERAS = ["robot0_head_camera", "robot0_left_eye_in_hand"]


def make_policy_env(ckpt_dict, config, layout_id, style_id, seed, extra_env_kwargs=None):
    """
    Environment of the checkpoint's training data (with camera observations and
    frame stacking) in the given layout / style, with optional env kwarg
    overrides. Returns the robomimic env and the underlying robosuite env.
    """
    env_meta = copy.deepcopy(ckpt_dict["env_metadata"])
    env_kwargs = env_meta["env_kwargs"]
    env_kwargs.pop("base_types", None)  # mobilization needs the default mobile base
    train_layouts = env_kwargs.pop("layout_ids", None) or []
    env_kwargs.pop("style_ids", None)
    if layout_id not in train_layouts:
        print(f"Layout {layout_id} is not a training layout {train_layouts}.")
        if env_kwargs.pop("force_robot_placement", None) is not None:
            print("Dropping the training data's forced robot placement.")
    env_kwargs.update(layout_and_style_ids=[[layout_id, style_id]], seed=seed, **(extra_env_kwargs or {}))
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=ckpt_dict["shape_metadata"]["use_images"],
    )
    env = EnvUtils.wrap_env_from_config(env, config=config)
    base_env = env.env.env if hasattr(env.env, "env") else env.env
    return env, base_env


def find_scene_model(scene_model_root_dir, base_env, layout_id, style_id, ckpt_idx):
    """
    Scene model laid out as by download_scene_models.py. Task variants (e.g.
    CloseDrawerLeft) share the scene models of the RoboCasa task they derive from.
    """
    tried = []
    for cls in type(base_env).__mro__:
        scene_dir = os.path.join(
            scene_model_root_dir, camel_to_snake_case(cls.__name__), f"layout{layout_id}_style{style_id}"
        )
        regex = os.path.join(scene_dir, "model/splatfacto/*/nerfstudio_models", f"step-{ckpt_idx:09d}.ckpt")
        paths = sorted(glob(regex))
        if len(paths) > 0:
            return scene_dir, os.path.dirname(os.path.dirname(paths[-1]))
        tried.append(regex)
        if cls.__name__ == "Kitchen":
            break
    raise FileNotFoundError(
        "No scene model matches any of\n  " + "\n  ".join(tried)
        + "\nDownload or train it first (see mobipi/scripts/download_scene_models.py)."
    )


def find_scene_open_drawer(env, ply_path, min_points=200):
    """
    The drawer that is open in a scene model's capture, and its opening (fraction
    of RoboCasa's fully open drawer, 0.55 x drawer depth), from the scene's point
    cloud: an open drawer leaves points in front of its cabinet face between the
    drawer's bottom and top, a closed one only its handle (~5 cm).
    """
    pc = np.asarray(o3d.io.read_point_cloud(ply_path).points)
    candidates = []
    for name, fixture in env.fixtures.items():
        if not isinstance(fixture, Drawer):
            continue
        w, d, h = fixture.size
        c, s = np.cos(fixture.rot), np.sin(fixture.rot)
        rel = pc - fixture.pos
        local_x = c * rel[:, 0] + s * rel[:, 1]
        local_y = -s * rel[:, 0] + c * rel[:, 1]  # the drawer front faces -y
        in_front = (
            (np.abs(local_x) < w / 2 - 0.03) & (np.abs(rel[:, 2]) < h / 2 - 0.02)
            & (local_y < -d / 2 - 0.01) & (local_y > -d / 2 - 0.6)
        )
        if in_front.sum() < min_points:
            continue
        opening = np.percentile(-local_y[in_front] - d / 2, 90) / (0.55 * d)
        if opening <= 1.1:  # anything further out is other furniture in front of the drawer
            candidates.append((int(in_front.sum()), name, float(np.clip(opening, 0.0, 1.0))))
    if len(candidates) == 0:
        raise RuntimeError(f"Found no open drawer in {ply_path}; pass --no_pin_scene_drawer to sample drawers instead.")
    if len(candidates) > 1:
        print(f"Several open drawer candidates in {ply_path}: {sorted(candidates, reverse=True)}; using the densest.")
    _, name, opening = max(candidates)
    return name, opening


def target_fixture(env):
    for name in ("drawer", "door_fxtr"):
        if getattr(env, name, None) is not None:
            return getattr(env, name)
    return None


def target_opening(env):
    fixture = target_fixture(env)
    if fixture is None or not hasattr(fixture, "get_door_state"):
        return None
    return float(max(fixture.get_door_state(env=env).values()))


def restart_obs_history(env):
    """Frame-stacked observation of the current state, as right after a reset."""
    obs = env.env.get_observation() if hasattr(env, "obs_history") else env.get_observation()
    if not hasattr(env, "obs_history"):
        return obs
    env.timestep = 0
    env.update_obs(obs, reset=True)
    env.obs_history = env._get_initial_obs_history(init_obs=obs)
    return env._get_stacked_obs_from_history()


def run_policy(env, base_env, policy, horizon, video_skip, video_size=256):
    """Rolls out the Cartesian policy from the current state (no reset)."""
    policy.start_episode()
    adapter = RBY1CartesianAdapter(base_env)  # holds the parts the policy does not control where they are now
    obs = restart_obs_history(env)
    frames = []
    success, steps = False, 0
    for steps in range(1, horizon + 1):
        command = policy(ob=obs)
        obs, _, _, _ = env.step(adapter(command))
        if video_skip > 0 and steps % video_skip == 0:
            frames.append(np.concatenate([
                env.render(mode="rgb_array", height=video_size, width=video_size, camera_name=c)
                for c in VIDEO_CAMERAS
            ], axis=1))
        if env.is_success()["task"]:
            success = True
            break
    return dict(success=success, steps=steps, frames=frames)


@click.command()
@click.option("--checkpoint", required=True, type=str, help="robomimic Cartesian policy checkpoint (.pth).")
@click.option("--dataset", default=os.path.join(MOBIPI_DIR, "datasets/close_drawer_look/close_drawer_look_300_im128_cmd.hdf5"),
              type=str, help="Demonstration dataset whose initial frames are used for scoring (the policy's training data).")
@click.option("--filter_key", default="train", type=str, help="Dataset mask whose initial frames are used for scoring.")
@click.option("--camera_names", default="robot0_head_camera", type=str, help="Comma-separated cameras to score.")
@click.option("--seed", default=1, type=int)
@click.option("--layout_id", default=0, type=int)
@click.option("--style_id", default=0, type=int)
@click.option("--log_root_dir", default=LOG_ROOT_DIR, type=str)
@click.option("--scene_model_root_dir", default=SCENE_MODEL_ROOT_DIR, type=str)
@click.option("--scene_model_ckpt_idx", default=29999, type=int)
@click.option("--num_episodes", default=3, type=int)
@click.option("--bo_num_samples", default=500, type=int)
@click.option("--num_init_samples", default=2500, type=int)
@click.option("--k", default=5, type=int)
@click.option("--skip_detection", is_flag=True)
@click.option("--skip_collision", is_flag=True)
@click.option("--collision_radius_scale", default=0.4, type=float,
              help="Scoring collision radius as a fraction of the robot radius (0.5 in Mobi-pi). The CloseDrawerLeft "
                   "demos start 0.23-0.33 m from the nearest scene point, at the edge of the 0.5 x 0.44 m default.")
@click.option("--encoder_name", default="dino_dense_descriptor")
@click.option("--detector_name", default="openbmb/MiniCPM-V-2")
@click.option("--detector_view_robot", is_flag=True)
@click.option("--base_estimator", default="ET")
@click.option("--pos_threshold", default=0.3, type=float, help="Max distance (m) to the demo start pose to count as nav success.")
@click.option("--heading_threshold", default=0.35, type=float, help="Max heading error (rad) to the demo start pose to count as nav success.")
@click.option("--nav_mode", default="velocity", type=click.Choice(["velocity", "kinematic"]), help="Drive the base with velocity commands, or move it along the plan directly.")
@click.option("--horizon", default=1000, type=int, help="Max policy steps per episode (demos take ~600).")
@click.option("--video_skip", default=2, type=int, help="Record every n-th policy step (0: no rollout video).")
@click.option("--no_pin_scene_drawer", is_flag=True,
              help="Sample the target drawer per reset as in training, instead of using the scene model's open drawer.")
@click.option("--vis", is_flag=True)
def main(
    checkpoint,
    dataset,
    filter_key,
    camera_names,
    seed,
    layout_id,
    style_id,
    log_root_dir,
    scene_model_root_dir,
    scene_model_ckpt_idx,
    num_episodes,
    bo_num_samples,
    num_init_samples,
    k,
    skip_detection,
    skip_collision,
    collision_radius_scale,
    encoder_name,
    detector_name,
    detector_view_robot,
    base_estimator,
    pos_threshold,
    heading_threshold,
    nav_mode,
    horizon,
    video_skip,
    no_pin_scene_drawer,
    vis,
):
    checkpoint = os.path.abspath(os.path.expanduser(checkpoint))
    dataset = os.path.abspath(os.path.expanduser(dataset))
    _, first_action = load_dataset_info(dataset)
    camera_names = [c.strip() for c in camera_names.split(",")]
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load the policy first: it sets up robomimic's observation processing for the env
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint, device=device)
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
    env_name = ckpt_dict["env_metadata"]["env_name"]

    # Logging
    run_name = f"{os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(checkpoint))))}" \
               f"_{os.path.splitext(os.path.basename(checkpoint))[0]}"
    log_dir = os.path.join(
        log_root_dir, env_name, "mobipi_rby1", run_name,
        f"layout{layout_id}_style{style_id}_seed{seed}",
    )
    os.makedirs(log_dir, exist_ok=True)
    file_stream = open(
        os.path.join(log_dir, f"console_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), "a"
    )
    sys.stdout = DualStream(file_stream, sys.stdout)
    sys.stderr = DualStream(file_stream, sys.stderr)
    print(f"Policy: {checkpoint} (env {env_name})")
    print(f"Dataset: {dataset}")

    env, base_env = make_policy_env(ckpt_dict, config, layout_id, style_id, seed)
    scene_dir, scene_model_dir = find_scene_model(
        scene_model_root_dir, base_env, layout_id, style_id, scene_model_ckpt_idx
    )
    ply_path = os.path.join(scene_dir, "pc.ply")
    print(f"Using scene model {scene_model_dir}")

    if not no_pin_scene_drawer:
        # The env has to be rebuilt: the target fixture is chosen when the model is loaded
        drawer_name, opening = find_scene_open_drawer(base_env, ply_path)
        print(f"Open drawer in the scene model: {drawer_name} (opening {opening:.2f}); using it as the target")
        env.env.env.close() if hasattr(env.env, "env") else env.env.close()
        try:
            env, base_env = make_policy_env(
                ckpt_dict, config, layout_id, style_id, seed,
                extra_env_kwargs=dict(drawer_id=drawer_name, drawer_open_range=(opening, opening)),
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"The scene model's open drawer {drawer_name} cannot be used for {env_name} "
                f"(layout {layout_id}, style {style_id}): {e}"
            ) from e
    gripper_action = gripper_action_from_dataset(base_env, first_action)

    # Score function components
    if encoder_name == "dino_dense_descriptor":
        encoder = DinoDenseDescriptorEncoder(device="cuda", model_name="vit_base_patch16_224_dino")
    elif encoder_name == "dino":
        encoder = DinoEncoder(device="cuda", model_name="vit_base_patch16_224_dino")
    else:
        raise ValueError(f"Unrecognized encoder name [{encoder_name}]")

    env.reset()
    _, floor_fixture_bounds_2d, _, _, robot_size = get_env_map_and_default_robot_init_pose(env=base_env)

    # Only the dataset path / filter key are read from the config
    score_config = {"train": {"data": [{"path": dataset, "filter_key": filter_key}]}}
    dist = HybridDistribution(
        score_config,
        encoder,
        k=k,
        object_detection_model_name=detector_name,
        ply_path=ply_path,
        floor_fixture_bounds_2d=floor_fixture_bounds_2d,
        robot_size=robot_size,
        max_z=1.0,
        image_size=224,
        skip_detection=skip_detection,
        skip_collision=skip_collision,
        detector_view_robot=detector_view_robot,
        collision_threshold_multiplier=collision_radius_scale,
    )

    image_size = 224
    camera_intrinsics_dict = []
    for camera_name in camera_names:
        K = get_camera_intrinsic_matrix(base_env.sim, camera_name, camera_height=image_size, camera_width=image_size)
        camera_intrinsics_dict.append(
            {"w": image_size, "h": image_size, "fl_x": K[0, 0], "fl_y": K[1, 1], "cx": K[0, -1], "cy": K[1, -1]}
        )
    scene_model = BatchSceneModel(scene_model_dir, camera_intrinsics_dict)

    all_results, all_selected_poses, all_default_poses = [], [], []
    for ep in range(num_episodes):
        print(f"\n===== Episode {ep} =====")
        rng = np.random.default_rng(seed * 10000 + ep)
        # The reset applies the task's start pose (torso / head / left arm), as in the demos
        env.reset()
        sim = base_env.sim
        base_fixture_bounds_2d, floor_fixture_bounds_2d, _, _, _ = get_env_map_and_default_robot_init_pose(env=base_env)
        # Where the demos start: the base pose after the reset. RoboCasa's placement
        # (env._init_robot_pos) can overlap a fixture (e.g. a fridge next to the
        # drawer), and the reset's physics then pushes the robot away from it.
        default_pose = rby1_utils.get_base_vec(sim)
        fixture = target_fixture(base_env)
        initial_open = target_opening(base_env)
        print(f"Default (demo start) pose: {default_pose.round(3)}; target fixture "
              f"{getattr(fixture, 'name', None)} (opening {initial_open}), task '{base_env.get_ep_meta()['lang']}'")

        # Camera pose / robot appearance relative to the base, captured at the default pose
        rel_cam_positions, rel_cam_mats, robot_imgs, robot_masks = [], [], [], []
        for camera_name in camera_names:
            rel_pos, rel_mat, robot_img, robot_mask = rby1_utils.render_image_with_robot_mask(
                sim, camera_name, image_size, image_size
            )
            rel_cam_positions.append(rel_pos)
            rel_cam_mats.append(rel_mat)
            robot_imgs.append(robot_img)
            robot_masks.append(robot_mask)
        robot_imgs_torch = torch.tensor(np.array(robot_imgs) / 255, dtype=torch.float32, device="cuda")
        robot_masks_torch = torch.tensor(np.array(robot_masks)[..., None], dtype=torch.float32, device="cuda")
        rel_cam_positions = torch.tensor(np.array(rel_cam_positions), dtype=torch.float32, device="cuda")
        rel_cam_mats = torch.tensor(np.array(rel_cam_mats), dtype=torch.float32, device="cuda")

        # Random initial base pose from which the default pose is in view
        init_pose = rby1_utils.sample_nav_init_pose(
            base_env, base_fixture_bounds_2d, floor_fixture_bounds_2d, default_pose, rng,
            k_col=None if skip_collision else dist.k_col,
        )
        print(f"Initial pose: {init_pose.round(3)}")
        base_pos = np.array([init_pose[0], init_pose[1], 0.0])
        base_heading = init_pose[2]

        dist.set_obs_keys([c + "_image" for c in camera_names])
        dist.set_object(base_env.get_object_name())
        dist.set_task_name(base_env.get_ep_meta()["lang"])

        def render_images(robot_pose):
            with torch.no_grad():
                pose_torch = torch.tensor(robot_pose, device="cuda", dtype=torch.float32)
                extrinsics = [
                    compute_camera_extrinsics(pose_torch, p, m)
                    for p, m in zip(rel_cam_positions, rel_cam_mats)
                ]
                rendered = scene_model.render(extrinsics, image_size=image_size)
                edited = rendered * (1 - robot_masks_torch) + robot_imgs_torch * robot_masks_torch
            return rendered, edited

        def score_function(robot_poses):
            lazy_renders = [lambda pose=pose: render_images(pose) for pose in robot_poses]
            scores, scores_per_view, rendered_images = dist.compute_score(
                lazy_renders,
                torch.tensor(np.array(robot_poses), device="cuda", dtype=torch.float32),
                env=base_env,
                base_pos=base_pos,
                base_heading=base_heading,
                check_fov=np.pi * 75 / 180,
            )
            score_info = {
                "scores_per_view": [s.detach().cpu().numpy().tolist() for s in scores_per_view],
                "rendered_images": [im for im in rendered_images],
            }
            return scores.detach().cpu().numpy().tolist(), score_info

        # Sanity check: the demo start pose should score well (FOV check off)
        default_score = dist.compute_score(
            [lambda: render_images(default_pose)],
            torch.tensor(default_pose[None], device="cuda", dtype=torch.float32),
        )[0].item()
        print(f"Score of default pose: {default_score:.4f}")

        bounds = [
            (floor_fixture_bounds_2d[2, 0], floor_fixture_bounds_2d[0, 0]),
            (floor_fixture_bounds_2d[0, 1], floor_fixture_bounds_2d[1, 1]),
            (base_heading - np.pi / 2, base_heading + np.pi / 2),
        ]
        normalized_initial_samples = np.random.RandomState(seed * 10000 + ep).uniform(
            0, 1, size=(num_init_samples, 3)
        )
        val2str = lambda a: f"[{a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f}]"

        t0 = time.time()
        best_pose, history = optimize_pose_batch(
            score_function,
            bounds,
            algorithm="bayesian",
            base_estimator=base_estimator,
            n_iterations=bo_num_samples // 5,
            normalize_data=True,
            track_info=True,
            initial_samples=normalized_initial_samples,
            acq_func="LCB",
            acq_func_kwargs=dict(kappa=1.96),
            seed=seed * 10000,
            batch_size=5,
            print_fn=val2str,
        )
        scoring_time = time.time() - t0
        best_pose[2] = angle_wrap(best_pose[2])
        sampled_scores = np.array(history["sampled_scores"])
        print(f"Best pose: {best_pose.round(3)} (score {sampled_scores.max():.4f}, {scoring_time:.0f}s)")

        if len(history["rendered_images"]) > 0:
            plot_best_renders(
                np.array(history["rendered_images"]), history["sampled_scores"],
                save_path=os.path.join(log_dir, f"ep{ep}_renders.pdf"),
            )
        del history["rendered_images"]
        plot_best_metrics(
            default_pose, history["sampled_points"], sampled_scores,
            save_path=os.path.join(log_dir, f"ep{ep}_progress.pdf"),
        )

        # Navigate to the selected pose
        t0 = time.time()
        nav_info = rby1_utils.move_to_pose(
            base_env, best_pose, dist.k_col, floor_fixture_bounds_2d,
            render_camera=camera_names[0], gripper_action=gripper_action, mode=nav_mode,
        )
        nav_time = time.time() - t0
        save_video(nav_info.pop("images"), os.path.join(log_dir, f"ep{ep}_nav.mp4"))

        final_pose = nav_info["settled_vec"]
        sel_pos_err, sel_head_err = pose_error(best_pose, default_pose)
        fin_pos_err, fin_head_err = pose_error(final_pose, default_pose)
        nav_pos_err, nav_head_err = pose_error(final_pose, best_pose)
        nav_success = bool(fin_pos_err < pos_threshold and fin_head_err < heading_threshold)
        print(
            f"Episode {ep}: selected pose is {sel_pos_err:.3f} m / {sel_head_err:.3f} rad from the demo start pose; "
            f"final pose {fin_pos_err:.3f} m / {fin_head_err:.3f} rad -> nav {'success' if nav_success else 'failure'}"
        )

        # Run the policy from the reached pose
        t0 = time.time()
        rollout = run_policy(env, base_env, policy, horizon, video_skip)
        rollout_time = time.time() - t0
        save_video(rollout.pop("frames"), os.path.join(log_dir, f"ep{ep}_rollout.mp4"),
                   fps=max(1, 20 // max(video_skip, 1)))
        final_open = target_opening(base_env)
        policy_pos_drift, policy_head_drift = pose_error(rby1_utils.get_base_vec(sim), final_pose)
        print(f"Episode {ep}: policy {'success' if rollout['success'] else 'failure'} after {rollout['steps']} steps "
              f"(opening {initial_open} -> {final_open}, {rollout_time:.0f}s)")

        result = dict(
            episode=ep,
            layout=layout_id,
            style=style_id,
            fixture=getattr(fixture, "name", None),
            init_pose=init_pose.tolist(),
            default_pose=default_pose.tolist(),
            selected_pose=best_pose.tolist(),
            final_pose=final_pose.tolist(),
            default_pose_score=default_score,
            best_score=float(sampled_scores.max()),
            num_scored=int(len(sampled_scores)),
            num_valid_scored=int((sampled_scores > 0).sum()),
            selected_to_default_pos_err=sel_pos_err,
            selected_to_default_heading_err=sel_head_err,
            final_to_default_pos_err=fin_pos_err,
            final_to_default_heading_err=fin_head_err,
            nav_pos_err=nav_pos_err,
            nav_heading_err=nav_head_err,
            nav_mode=nav_mode,
            nav_stuck=nav_info["stuck"],
            nav_used_rrt=nav_info["used_rrt"],
            nav_approach_pose=None if nav_info["approach_pose"] is None else nav_info["approach_pose"].tolist(),
            nav_num_steps=nav_info["num_steps"],
            nav_path_length=nav_info["path_length"],
            nav_num_collision_steps=nav_info["num_collision_steps"],
            nav_colliding_geoms=nav_info["colliding_geoms"],
            nav_final_contacts=[list(c) for c in nav_info["final_contacts"]],
            nav_settle_drift=nav_info["settle_drift"].tolist(),
            nav_success=nav_success,
            initial_open=initial_open,
            final_open=final_open,
            policy_steps=rollout["steps"],
            policy_base_drift=[policy_pos_drift, policy_head_drift],
            scoring_time=scoring_time,
            nav_time=nav_time,
            rollout_time=rollout_time,
            success=rollout["success"],
        )
        all_results.append(result)
        all_selected_poses.append(best_pose)
        all_default_poses.append(default_pose)
        with open(os.path.join(log_dir, f"ep{ep}_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        history.update({k: v for k, v in nav_info.items() if isinstance(v, np.ndarray)})
        np.savez(os.path.join(log_dir, f"ep{ep}_info.npz"), **history)

        if vis:
            topdown_image, image_corners = get_topdown_image(base_env, bounds)
            plot_scores_on_topdown_image(
                topdown_image,
                image_corners,
                bounds,
                history["sampled_points"],
                sampled_scores,
                vis_size=0.05,
                save_path=os.path.join(log_dir, f"ep{ep}_topdown_scores.pdf"),
                render_heading=True,
                default_pose=default_pose,
                init_pose=init_pose,
                selected_pose=best_pose,
                nav_pose=final_pose,
            )

    # Summary
    summary = dict(
        checkpoint=checkpoint,
        dataset=dataset,
        num_episodes=len(all_results),
        success_rate=float(np.mean([r["success"] for r in all_results])),
        nav_success_rate=float(np.mean([r["nav_success"] for r in all_results])),
        success_rate_given_nav_success=float(np.mean([r["success"] for r in all_results if r["nav_success"]]))
        if any(r["nav_success"] for r in all_results) else None,
        mean_final_to_default_pos_err=float(np.mean([r["final_to_default_pos_err"] for r in all_results])),
        mean_final_to_default_heading_err=float(np.mean([r["final_to_default_heading_err"] for r in all_results])),
        mean_nav_pos_err=float(np.mean([r["nav_pos_err"] for r in all_results])),
        episodes_with_nav_contacts=int(sum(r["nav_num_collision_steps"] > 0 for r in all_results)),
    )
    print("\n===== Summary =====")
    print(json.dumps(summary, indent=2))
    with open(os.path.join(log_dir, "summary.json"), "w") as f:
        json.dump(dict(summary=summary, episodes=all_results), f, indent=2)
    plot_pose_histograms(
        np.array(all_default_poses),
        np.array(all_selected_poses),
        os.path.join(log_dir, "hist.pdf"),
    )


if __name__ == "__main__":
    main()
