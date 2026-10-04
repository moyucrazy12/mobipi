"""
Tests the Mobi-pi mobilization stage on RB-Y1 without running a policy:
score candidate base poses against the initial frames of a demonstration
dataset (3DGS renders + DINO kNN + collision / visibility checks), pick the
best pose with BO, and navigate the robot there.

Follows `eval_mobipi.py`, but reads env metadata and training images directly
from the dataset instead of a policy checkpoint, and uses the RB-Y1 helpers in
`mobipi.utils.rby1_utils` for base pose access, init placement and navigation.

Since every demo in the dataset starts from the same base pose (the env's
forced robot placement), the distance between the selected pose and that
default pose is reported as the main quality metric.

Usage:
python test_mobilization.py --dataset ../../datasets/demo_im128_head_wrist.hdf5 \
    --layout_id 1 --style_id 1 --num_episodes 3 --skip_detection --vis

@yjy0625 (original eval script), adapted for RB-Y1
"""
import os
import sys
import json
import h5py
import time
import click
import torch
import warnings
import numpy as np
from glob import glob
from datetime import datetime

import robosuite
import robocasa  # registers RoboCasa environments
from robocasa.models.fixtures import FixtureType
from robosuite.utils.camera_utils import get_camera_intrinsic_matrix

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
    setup_orthographic_camera,
    unproject_image_corners_on_floor,
)
from mobipi.utils import rby1_utils

warnings.filterwarnings("ignore", message=".*default value of the antialias parameter.*")


def load_dataset_info(dataset_path):
    with h5py.File(dataset_path, "r") as f:
        env_args = json.loads(f["data"].attrs["env_args"])
        demo_key = sorted(f["data"].keys())[0]
        first_action = f["data"][demo_key]["actions"][0].copy()
    return env_args, first_action


def make_env(env_args, layout_id, style_id):
    env_kwargs = dict(env_args["env_kwargs"])
    dataset_layouts = env_kwargs.pop("layout_ids", None) or []
    env_kwargs.pop("style_ids", None)
    if layout_id not in dataset_layouts:
        # The dataset pins a fixture name and robot placement of its own layout.
        # Elsewhere, target the same fixture type and let RoboCasa place the
        # robot in front of it as usual; that placement is the reference pose.
        door_id = env_kwargs.get("door_id")
        if isinstance(door_id, str) and "microwave" in door_id:
            env_kwargs["door_id"] = FixtureType.MICROWAVE
        elif door_id is not None:
            env_kwargs.pop("door_id")
        env_kwargs.pop("force_robot_placement", None)
        print(f"Layout {layout_id} is not in the dataset layouts {dataset_layouts}: "
              f"using door_id={env_kwargs.get('door_id')} and RoboCasa's default robot placement.")
    env_kwargs.update(
        layout_and_style_ids=[[layout_id, style_id]],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=False,
    )
    return robosuite.make(env_args["env_name"], **env_kwargs)


def gripper_action_from_dataset(env, first_action):
    split = env.robots[0].composite_controller._whole_body_controller_action_split_indexes
    return {
        part: first_action[start:end]
        for part, (start, end) in split.items() if part.endswith("gripper")
    }


def get_topdown_image(env, bounds, image_size=1024):
    xy_center = ((bounds[0][0] + bounds[0][1]) / 2, (bounds[1][0] + bounds[1][1]) / 2)
    camera_params = setup_orthographic_camera(
        env, "freeview", xy_center, 450, np.pi, fovy=1.0, image_size=image_size
    )
    image = env.sim.render(width=image_size, height=image_size, camera_name="freeview")[::-1]
    corners = unproject_image_corners_on_floor(
        camera_params["pos"], camera_params["quat"], camera_params["fovy"],
        (image_size, image_size),
    )
    return image, corners


def pose_error(a, b):
    return float(np.linalg.norm(a[:2] - b[:2])), float(abs(angle_wrap(a[2] - b[2])))


@click.command()
@click.option("--dataset", default=os.path.join(MOBIPI_DIR, "datasets/demo_im128_head_wrist.hdf5"), type=str)
@click.option("--filter_key", default="train", type=str, help="Dataset mask whose initial frames are used for scoring.")
@click.option("--camera_names", default="robot0_head_camera", type=str, help="Comma-separated cameras to score.")
@click.option("--seed", default=1, type=int)
@click.option("--layout_id", default=1, type=int)
@click.option("--style_id", default=1, type=int)
@click.option("--log_root_dir", default=LOG_ROOT_DIR, type=str)
@click.option("--scene_model_root_dir", default=SCENE_MODEL_ROOT_DIR, type=str)
@click.option("--scene_model_ckpt_idx", default=29999, type=int)
@click.option("--num_episodes", default=3, type=int)
@click.option("--bo_num_samples", default=500, type=int)
@click.option("--num_init_samples", default=2500, type=int)
@click.option("--k", default=5, type=int)
@click.option("--skip_detection", is_flag=True)
@click.option("--skip_collision", is_flag=True)
@click.option("--encoder_name", default="dino_dense_descriptor")
@click.option("--detector_name", default="openbmb/MiniCPM-V-2")
@click.option("--detector_view_robot", is_flag=True)
@click.option("--base_estimator", default="ET")
@click.option("--pos_threshold", default=0.3, type=float, help="Max distance (m) to the demo start pose to count as success.")
@click.option("--heading_threshold", default=0.35, type=float, help="Max heading error (rad) to the demo start pose to count as success.")
@click.option("--nav_mode", default="wheels", type=click.Choice(["wheels", "kinematic"]), help="Drive the wheels, or move the chassis along the plan directly.")
@click.option("--vis", is_flag=True)
def main(
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
    encoder_name,
    detector_name,
    detector_view_robot,
    base_estimator,
    pos_threshold,
    heading_threshold,
    nav_mode,
    vis,
):
    dataset = os.path.abspath(os.path.expanduser(dataset))
    env_args, first_action = load_dataset_info(dataset)
    env_name = env_args["env_name"]
    camera_names = [c.strip() for c in camera_names.split(",")]

    # Logging
    log_dir = os.path.join(
        log_root_dir, env_name, "mobilization_test",
        f"layout{layout_id}_style{style_id}_seed{seed}",
    )
    os.makedirs(log_dir, exist_ok=True)
    file_stream = open(
        os.path.join(log_dir, f"console_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"), "a"
    )
    sys.stdout = DualStream(file_stream, sys.stdout)
    sys.stderr = DualStream(file_stream, sys.stderr)
    print(f"Dataset: {dataset} (env {env_name}, robot {env_args['env_kwargs']['robots']})")

    # Find scene model, laid out as by download_scene_models.py
    scene_dir = os.path.join(
        scene_model_root_dir, camel_to_snake_case(env_name), f"layout{layout_id}_style{style_id}"
    )
    scene_model_regex = os.path.join(
        scene_dir, "model/splatfacto/*/nerfstudio_models", f"step-{scene_model_ckpt_idx:09d}.ckpt"
    )
    scene_model_paths = sorted(glob(scene_model_regex))
    if len(scene_model_paths) == 0:
        raise FileNotFoundError(
            f"No scene model matches {scene_model_regex}. "
            "Download it with mobipi/scripts/download_scene_models.py."
        )
    scene_model_dir = os.path.dirname(os.path.dirname(scene_model_paths[-1]))
    ply_path = os.path.join(scene_dir, "pc.ply")
    print(f"Using scene model {scene_model_dir}")

    # Score function components
    if encoder_name == "dino_dense_descriptor":
        encoder = DinoDenseDescriptorEncoder(device="cuda", model_name="vit_base_patch16_224_dino")
    elif encoder_name == "dino":
        encoder = DinoEncoder(device="cuda", model_name="vit_base_patch16_224_dino")
    else:
        raise ValueError(f"Unrecognized encoder name [{encoder_name}]")

    env = make_env(env_args, layout_id, style_id)
    gripper_action = gripper_action_from_dataset(env, first_action)
    env.reset()
    (
        base_fixture_bounds_2d,
        floor_fixture_bounds_2d,
        default_robot_pos,
        default_robot_ori,
        robot_size,
    ) = get_env_map_and_default_robot_init_pose(env=env)
    default_pose = np.array([default_robot_pos[0], default_robot_pos[1], default_robot_ori[-1]])
    print(f"Default (demo start) pose: {default_pose.round(3)}; robot size {robot_size}; "
          f"target fixture {env.door_fxtr.name}, task '{env.get_ep_meta()['lang']}'")

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
    )

    image_size = 224
    camera_intrinsics_dict = []
    for camera_name in camera_names:
        K = get_camera_intrinsic_matrix(env.sim, camera_name, camera_height=image_size, camera_width=image_size)
        camera_intrinsics_dict.append(
            {"w": image_size, "h": image_size, "fl_x": K[0, 0], "fl_y": K[1, 1], "cx": K[0, -1], "cy": K[1, -1]}
        )
    scene_model = BatchSceneModel(scene_model_dir, camera_intrinsics_dict)

    all_results, all_selected_poses = [], []
    for ep in range(num_episodes):
        print(f"\n===== Episode {ep} =====")
        rng = np.random.default_rng(seed * 10000 + ep)
        env.reset()

        # Camera pose / robot appearance relative to the base, captured at the default pose
        rel_cam_positions, rel_cam_mats, robot_imgs, robot_masks = [], [], [], []
        for camera_name in camera_names:
            rel_pos, rel_mat, robot_img, robot_mask = rby1_utils.render_image_with_robot_mask(
                env.sim, camera_name, image_size, image_size
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
            env, base_fixture_bounds_2d, floor_fixture_bounds_2d, default_pose, rng,
            k_col=None if skip_collision else dist.k_col,
        )
        print(f"Initial pose: {init_pose.round(3)}")
        base_pos = np.array([init_pose[0], init_pose[1], 0.0])
        base_heading = init_pose[2]

        dist.set_obs_keys([c + "_image" for c in camera_names])
        dist.set_object(env.get_object_name())
        dist.set_task_name(env.get_ep_meta()["lang"])

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
                env=env,
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
            env, best_pose, dist.k_col, floor_fixture_bounds_2d,
            render_camera=camera_names[0], gripper_action=gripper_action, mode=nav_mode,
        )
        nav_time = time.time() - t0
        save_video(nav_info.pop("images"), os.path.join(log_dir, f"ep{ep}_nav.mp4"))

        final_pose = nav_info["settled_vec"]
        sel_pos_err, sel_head_err = pose_error(best_pose, default_pose)
        fin_pos_err, fin_head_err = pose_error(final_pose, default_pose)
        nav_pos_err, nav_head_err = pose_error(final_pose, best_pose)
        success = bool(fin_pos_err < pos_threshold and fin_head_err < heading_threshold)
        result = dict(
            episode=ep,
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
            nav_num_steps=nav_info["num_steps"],
            nav_path_length=nav_info["path_length"],
            nav_num_collision_steps=nav_info["num_collision_steps"],
            nav_colliding_geoms=nav_info["colliding_geoms"],
            nav_final_contacts=[list(c) for c in nav_info["final_contacts"]],
            nav_settle_drift=nav_info["settle_drift"].tolist(),
            scoring_time=scoring_time,
            nav_time=nav_time,
            success=success,
        )
        all_results.append(result)
        all_selected_poses.append(best_pose)
        print(
            f"Episode {ep}: selected pose is {sel_pos_err:.3f} m / {sel_head_err:.3f} rad from the demo start pose; "
            f"final pose {fin_pos_err:.3f} m / {fin_head_err:.3f} rad -> {'success' if success else 'failure'}"
        )
        with open(os.path.join(log_dir, f"ep{ep}_result.json"), "w") as f:
            json.dump(result, f, indent=2)
        history.update({k: v for k, v in nav_info.items() if isinstance(v, np.ndarray)})
        np.savez(os.path.join(log_dir, f"ep{ep}_info.npz"), **history)

        if vis:
            topdown_image, image_corners = get_topdown_image(env, bounds)
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
        num_episodes=len(all_results),
        success_rate=float(np.mean([r["success"] for r in all_results])),
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
        np.array([default_pose] * len(all_selected_poses)),
        np.array(all_selected_poses),
        os.path.join(log_dir, "hist.pdf"),
    )


if __name__ == "__main__":
    main()
