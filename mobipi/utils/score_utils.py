"""
Utilities related to computing scores for sampled robot poses.

@yjy0625
"""
import numpy as np
import os
import torch
import time
import h5py
import json
import open3d as o3d
from tqdm import tqdm
from glob import glob
from torch.cuda.amp import autocast
from scipy.spatial import cKDTree
from contextlib import nullcontext
from sklearn.decomposition import PCA
from torchvision.transforms import ToPILImage, ToTensor
from transformers import AutoModel, AutoTokenizer

from mobipi.utils.env_utils import (
    check_robot_positions,
    compute_camera_extrinsics,
    render_image_with_robot_mask,
)


# Normalize function
def normalize(vectors):
    return vectors / torch.norm(vectors, dim=1, keepdim=True)


def normalize_np(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class Distribution(object):
    def __init__(self, config, encoder, obs_key=None):
        assert obs_key is not None
        data_path = config["train"]["data"][0]["path"]
        filter_key = self.config["train"]["data"][0]["filter_key"]
        initial_images = []
        with h5py.File(data_path, "r") as df:
            demo_keys = list(df["mask"][filter_key])
            for demo_key in demo_keys:
                initial_images.append(df["data"][demo_key]["obs"][obs_key][0])
        initial_images = np.array(initial_images)

        BS = 256

        # Initialize an empty list to store embeddings
        all_embeddings = []

        # Process in batches
        with torch.no_grad():
            for i in range(0, len(initial_images), BS):
                batch = (
                    torch.tensor(initial_images[i : i + BS] / 255, dtype=torch.float32)
                    .permute(0, 3, 1, 2)
                    .to("cuda")
                )
                embeddings = encoder(batch)
                all_embeddings.append(
                    embeddings.detach().cpu().numpy()
                )  # Move to CPU to free up GPU memory

            # Concatenate all the embeddings into a single tensor
            all_embeddings = np.concatenate(all_embeddings, axis=0)
        self.training_data = all_embeddings
        self.normalized_training_data = normalize_np(self.training_data)

        self.encoder = encoder

    def compute_score(self, eval_images):
        # torch images with shape (N, 3, H, W)
        raise NotImplementedError


class KNNDenseDescriptorDistribution(Distribution):
    def __init__(self, config, encoder, detector_wrapper, k=5, pca_components=128):
        # Initialize the distribution
        self.config = config
        self.encoder = encoder
        self.k = k
        self.pca_components = pca_components
        self.detector_wrapper = detector_wrapper
        self._dense_descriptors_pca_dict = dict()
        self._pca_dict = dict()
        self._task_langs = dict()
        self.use_pca = True
        self._descriptor_dim = None

        self.obs_key = None

        print(f"PCA reduced descriptor size to {self.pca_components}")

    def _process_images(self, initial_images, batch_size=256):
        # Initialize an empty list to store dense descriptors
        dense_descriptors = []

        # Process in batches
        with torch.no_grad():
            for i in tqdm(
                range(0, len(initial_images), batch_size),
                desc="Process training embeddings",
            ):
                batch = (
                    torch.tensor(
                        initial_images[i : i + batch_size] / 255, dtype=torch.float32
                    )
                    .permute(0, 3, 1, 2)
                    .to("cuda")
                )
                descriptors = self.encoder(batch)  # Shape: (B, H'*W', C)
                dense_descriptors.append(descriptors.detach().cpu())
            print(f"Descriptor has size {descriptors.shape}")

        # Concatenate all dense descriptors
        dense_descriptors = torch.cat(dense_descriptors, dim=0).view(
            len(initial_images), -1, dense_descriptors[0].shape[-1]
        )  # Shape: (N, H' * W', C)
        dense_descriptors_np = dense_descriptors.view(
            -1, dense_descriptors.shape[-1]
        ).numpy()  # Flatten descriptors for PCA

        # Perform PCA on the training dense descriptors
        print("Fitting PCA to dense descriptors...")
        if dense_descriptors_np.shape[-1] > self.pca_components:
            self._pca_dict[self.obs_key] = PCA(n_components=self.pca_components)
            dense_descriptors_pca = self._pca_dict[self.obs_key].fit_transform(
                dense_descriptors_np
            )  # Shape: (N_train * H'W', pca_components)
            dense_descriptors_pca = torch.tensor(dense_descriptors_pca).view(
                len(initial_images), -1, self.pca_components
            )  # Shape: (N, H' * W', pca_components)
        else:
            dense_descriptors_pca = dense_descriptors_np
            self.use_pca = False
            self._descriptor_dim = dense_descriptors_np.shape[-1]
            dense_descriptors_pca = torch.tensor(dense_descriptors_pca).view(
                len(initial_images), -1, self._descriptor_dim
            )
        return dense_descriptors_pca  # Save PCA-transformed descriptors

    @property
    def _dense_descriptors_pca(self):
        return self._dense_descriptors_pca_dict[self.obs_key]

    def set_task(self, task_name):
        self.task_name = task_name
        if hasattr(self.encoder, "set_lang"):
            self.encoder.set_lang(task_name)

    def set_obs_key(self, obs_key):
        self.obs_key = obs_key
        if hasattr(self.encoder, "set_obs_key"):
            self.encoder.set_obs_key(obs_key)
        if self.obs_key not in self._dense_descriptors_pca_dict:
            if self.obs_key == "real":
                # parse real data
                data_dir = self.config["data_dir"]
                num_demos = self.config["num_demos"]
                initial_images = []
                for data_path in list(
                    sorted(glob(os.path.join(data_dir, "ep*_t0000.npz")))
                )[:num_demos]:
                    initial_images.append(np.load(data_path)["rgb"])
                self._task_langs = (
                    None  # we don't use language in real robot experiments
                )
            else:
                data_path = self.config["train"]["data"][0]["path"]
                filter_key = self.config["train"]["data"][0]["filter_key"]
                initial_images = []
                task_names = []
                total_num_images = 0
                with h5py.File(data_path, "r") as df:
                    demo_keys = list(df["mask"][filter_key])
                    print(f"Demo key length for obs key {obs_key} is {len(demo_keys)}.")
                    for demo_key in tqdm(
                        demo_keys, desc=f"Load initial images for key [{self.obs_key}]"
                    ):
                        img = df["data"][demo_key]["obs"][self.obs_key][0]
                        lang = json.loads(dict(df["data"][demo_key].attrs)["ep_meta"])[
                            "lang"
                        ]
                        if self.detector_wrapper is None:
                            initial_images.append(img)
                            task_names.append(lang)
                        else:
                            img_tensor = (
                                torch.tensor(img / 255, dtype=torch.float32)
                                .permute(2, 0, 1)
                                .unsqueeze(0)
                                .to("cuda")
                            )
                            total_num_images += 1
                            detector_score = (
                                self.detector_wrapper.compute_score(
                                    img_tensor[None][None], None, None
                                )[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .item()
                            )
                            if detector_score > 0:
                                initial_images.append(img)
                                task_names.append(lang)
                self._task_langs[self.obs_key] = task_names
            initial_images = np.array(initial_images)
            # TODO: debugging {
            # initial_images[:, :, 640:, :] = 0
            # }
            if self.detector_wrapper is not None:
                print(
                    f"[score_utils.py] {len(initial_images)} out of {total_num_images} images have the detected object inside."
                )
            else:
                print(f"[score_utils.py] loaded {len(initial_images)} images to k_id.")
            self._dense_descriptors_pca_dict[self.obs_key] = self._process_images(
                initial_images
            )

    def compute_score(
        self, rendered_images, edited_rendered_images, poses, masks=None, return_top_k=False
    ):
        """
        Compute kNN scores based on dense descriptors.
        Optionally, return the top-k rendered poses and images with the best scores.
        :param rendered_images: Input images tensor with shape (N, V, 3, H, W).
        :param edited_rendered_images: Edited images tensor (can be the same as rendered_images).
        :param poses: Tensor containing poses corresponding to rendered_images, shape (N, ...).
        :param masks: Tensor containing image masks with shape (N, H, W).
        :param return_top_k: Boolean, whether to return top-k poses and images with best scores.
        :return: kNN scores with shape (N,) if return_top_k is False.
                 Tuple of (kNN scores, top-k poses, top-k images) if return_top_k is True.
        """

        # Compute dense descriptors for eval images
        dense_eval_features = self.encoder(
            rendered_images.view(-1, *rendered_images.shape[2:])
        )  # Shape: (N * V, H' * W', C)
        dense_eval_features = dense_eval_features.view(
            rendered_images.shape[0],
            rendered_images.shape[1],
            -1,
            dense_eval_features.shape[-1],
        )  # Shape: (N, V, H' * W', C)

        # Apply PCA to evaluation descriptors
        dense_eval_features_np = (
            dense_eval_features.reshape(-1, dense_eval_features.shape[-1]).cpu().numpy()
        )
        if self.use_pca:
            dense_eval_features_pca = self._pca_dict[self.obs_key].transform(
                dense_eval_features_np
            )  # Shape: (N * V * H'W', pca_components)
        else:
            dense_eval_features_pca = dense_eval_features_np
        dense_eval_features_pca = (
            torch.tensor(dense_eval_features_pca)
            .view(
                rendered_images.shape[0],
                rendered_images.shape[1],
                -1,
                self.pca_components if self.use_pca else self._descriptor_dim,
            )
            .to("cuda")
        )  # Shape: (N, V, H' * W', pca_components)

        # Flatten dense descriptors for training data
        # NOTE: we filter training episodes based on object name
        #       we assume task name formatted as [verb .... the "object we care about"]
        #       so we extract out the object of interest by splitting the task name
        if self._task_langs is not None:
            train_episode_filter = np.array(
                [
                    task_name.split(" the ")[-1] == self.task_name.split(" the ")[-1]
                    for task_name in self._task_langs[self.obs_key]
                ]
            )
            binary_tensor = torch.from_numpy(train_episode_filter).bool()
            train_descriptors = self._dense_descriptors_pca[binary_tensor]
        else:
            train_descriptors = self._dense_descriptors_pca
        train_descriptors = train_descriptors.to(
            "cuda"
        )  # Shape: (N_train, H' * W', pca_components)

        scores = []

        # Storage for top-k top_k_scores and images if needed
        top_k_scores = []
        top_k_images = []

        # Process object mask if exists
        if masks is not None:
            assert rendered_images.shape[-2] == 224, "The code is only checked with image size 224."
            assert rendered_images.shape[-1] == 224, "The code is only checked with image size 224."
            stride = int((masks.shape[-2] * masks.shape[-1] // dense_eval_features_pca.shape[-2]) ** 0.5)

            # (N, H', W')
            patch_masks = torch.nn.functional.avg_pool2d(masks.float(), kernel_size=stride, stride=stride)

        # Process evaluation descriptors one by one
        with torch.no_grad():
            for eval_idx, eval_image_features in enumerate(dense_eval_features_pca):
                # eval_image_features: (V, H'W', pca_components)
                # Combine descriptors across the V renders
                best_scores_per_train = []
                for v_idx in range(eval_image_features.shape[0]):
                    eval_descriptors = eval_image_features[
                        v_idx
                    ]  # Shape: (H'W', pca_components)

                    # Normalize descriptors for cosine similarity
                    normalized_train_descriptors = (
                        train_descriptors / train_descriptors.norm(dim=-1, keepdim=True)
                    )  # Shape: (N_train, H' * W', pca_components)
                    normalized_eval_descriptors = (
                        eval_descriptors / eval_descriptors.norm(dim=-1, keepdim=True)
                    )  # Shape: (H' * W', pca_components)

                    # Compute pairwise cosine similarity
                    cos_sim = torch.sum(
                        normalized_eval_descriptors[None]
                        * normalized_train_descriptors,
                        dim=-1,
                    )  # Shape: (N_train, H' * W')
                    if masks is not None:
                        cos_sim = (
                            torch.sum(cos_sim * patch_masks[eval_idx].reshape(1, -1), dim=-1) / torch.sum(patch_masks[eval_idx])
                        )
                    else:
                        cos_sim = (
                            torch.sum(cos_sim, dim=-1) / cos_sim.shape[-1]
                        )  # Shape: (N_train,)
                    best_scores_per_train.append(
                        cos_sim
                    )  # Store similarity for this V render

                # Stack and select the best (highest similarity) for each training image
                best_scores_per_train = torch.stack(
                    best_scores_per_train, dim=0
                )  # Shape: (V, N_train)
                best_scores_per_train = torch.max(best_scores_per_train, dim=0)[
                    0
                ]  # Shape: (N_train,)

                # Compute top-k similarity scores for the training images
                topk_sims, topk_indices = torch.topk(
                    best_scores_per_train, self.k, largest=True
                )  # Top-k highest similarities
                mean_score = (
                    topk_sims.mean()
                )  # Mean cosine similarity of k nearest neighbors

                scores.append(
                    torch.nn.functional.relu(mean_score)
                )  # Clip negative values

                # If return_top_k is True, store the corresponding poses and images
                if return_top_k:
                    top_k_scores.append(topk_sims.detach().cpu().numpy())
                    top_k_images.append(
                        self.initial_images[topk_indices.detach().cpu().numpy()]
                    )

            # Convert scores to tensor
            scores_tensor = torch.tensor(scores).to(edited_rendered_images.device)

        # Cleanup
        del train_descriptors

        if return_top_k:
            # Concatenate top-k poses and images for all evaluations
            top_k_scores = np.stack(top_k_scores)
            top_k_images = np.stack(top_k_images)
            return scores_tensor, top_k_scores, top_k_images

        return scores_tensor


class MiniCPMV2ObjectDetectionDistribution:
    def __init__(self, model_name="openbmb/MiniCPM-V-2", device="cuda",
                 view_robot=False):
        # Transformers only allows SDPA attention with torch >= 2.1.1; fall back to
        # eager attention (same outputs, slower) on older torch
        from transformers.utils import is_torch_sdpa_available
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            attn_implementation="sdpa" if is_torch_sdpa_available() else "eager",
            torch_dtype=torch.bfloat16,
        )
        self.model = self.model.to(device, dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model.eval()
        self.to_pil = ToPILImage()
        self.time_printed = False
        self.view_robot = view_robot

    def set_object(self, object_name):
        self.object_description = object_name

    def compute_score(self, rendered_images, edited_rendered_images, poses):
        assert self.object_description is not None, "Object description must be set."
        image_handle = rendered_images if not self.view_robot else edited_rendered_images
        if image_handle.max() > 1.0 or image_handle.min() < 0.0:
            raise ValueError(
                "Input images must have pixel values normalized to [0, 1]."
            )

        pil_images = [self.to_pil(img_tensor) for img_tensor in image_handle[:, 0]]
        scores = []
        if not self.time_printed:
            start_time = time.time()
        for image in pil_images:
            msg = {
                "role": "user",
                "content": f"Is {self.object_description} in the image? Answer exactly 'yes' or 'no'.",
            }
            res, context, _ = self.model.chat(
                image=image,
                msgs=[msg],
                tokenizer=self.tokenizer,
                max_new_tokens=10,
                context=None,
                sampling=True,
                temperature=0.7,
            )
            scores.append("yes" in res.lower())
        if not self.time_printed:
            print(f"Elapsed time: {time.time() - start_time:.2f}s")
            self.time_printed = True
        return torch.tensor(scores, device=image_handle.device), None


class PCVacancyDistribution:
    def __init__(
        self,
        ply_path,
        floor_fixture_bounds_2d,
        robot_size,
        max_z=1.0,
        buffer_workspace_boundary=True,
    ):
        # Load the point cloud and convert to numpy array
        point_cloud = o3d.io.read_point_cloud(ply_path)
        self.pc = np.asarray(point_cloud.points)
        self.floor_fixture_bounds_2d = floor_fixture_bounds_2d
        self.robot_size = robot_size
        # Build a KDTree for fast neighbor queries
        self.kdtree = cKDTree(self.pc[:, :2])  # Use only xy coordinates for 2D queries
        self.max_z = max_z
        self.buffer_workspace_boundary = buffer_workspace_boundary

    def compute_score(
        self, rendered_images, edited_rendered_images, poses, numpy=False,
        robot_size=None, collision_threshold_multiplier=0.5
    ):
        if robot_size is None:
            collision_threshold = self.robot_size * collision_threshold_multiplier
        else:
            collision_threshold = self.robot_size * collision_threshold_multiplier
        if numpy:
            # Extract xy positions from poses
            poses_np = poses[:, :2]  # (num_queries, 2)
        else:
            poses_np = poses[:, :2].detach().cpu().numpy()
        # Query points within collision threshold radius for all poses
        indices = self.kdtree.query_ball_point(poses_np, collision_threshold)

        # Prepare scores for each pose
        scores = np.ones(len(poses_np), dtype=np.bool_)
        for i, idx_list in enumerate(indices):
            # Extract the points within the query radius
            points_in_radius = self.pc[idx_list]
            # Check if any point in the radius has z in the range [0.05, 1.0]
            if np.any(
                (points_in_radius[:, 2] >= 0.05)
                & (points_in_radius[:, 2] <= self.max_z)
            ):
                scores[i] = 0  # Mark as not vacant

        # Compound with room boundary
        if self.buffer_workspace_boundary:
            floor_min, floor_max = (
                self.floor_fixture_bounds_2d[3],
                self.floor_fixture_bounds_2d[1],
            )
            in_bounds = np.all(
                np.logical_and(
                    poses_np > floor_min[None] + self.robot_size * 2,
                    poses_np < floor_max[None] - self.robot_size * 2,
                ),
                axis=1,
            )
            scores = np.logical_and(scores, in_bounds)

        # Convert scores to a torch tensor and return
        if numpy:
            return scores
        return torch.tensor(scores, device=poses.device)


class HybridDistribution(Distribution):
    def __init__(
        self,
        config,
        image_encoder,
        k=None,
        object_detection_model_name="google/owlvit-base-patch32",
        base_fixture_bounds_2d=None,
        floor_fixture_bounds_2d=None,
        ply_path=None,
        robot_size=None,
        max_z=1.0,
        skip_detection=False,
        skip_collision=False,
        image_size=224,
        buffer_workspace_boundary=True,
        detector_view_robot=False,
        collision_threshold_multiplier=0.5,
    ):
        print(f"[score_utils.py] Creating hybrid score function.")
        if not skip_detection:
            if "openbmb" in object_detection_model_name:
                self.k_obj = MiniCPMV2ObjectDetectionDistribution(
                    model_name=object_detection_model_name, view_robot=detector_view_robot
                )
            else:
                raise ValueError(f"Object detection model name [{object_detection_model_name}] not recognized.")
        print(f"[score_utils.py] k_obj.")
        if k is not None:
            self.k_id = KNNDenseDescriptorDistribution(
                config, image_encoder, None, k=k
            )
        else:
            raise ValueError("k shouldn't be empty")
        print(f"[score_utils.py] k_id.")
        self.skip_collision = skip_collision
        # Scoring only: navigation and init pose sampling query k_col with their own radii
        self.collision_threshold_multiplier = collision_threshold_multiplier
        self.k_col = PCVacancyDistribution(
            ply_path,
            floor_fixture_bounds_2d,
            robot_size,
            max_z,
            buffer_workspace_boundary,
        )
        print(f"[score_utils.py] k_col.")

        self.image_size = (
            (image_size, image_size) if type(image_size) is int else image_size
        )
        self.obs_keys = []

    def set_obs_keys(self, obs_keys):
        self.obs_keys = obs_keys

    def set_object(self, object_name):
        if hasattr(self, "k_obj"):
            self.k_obj.set_object(object_name)

    def set_task_name(self, task_name):
        if hasattr(self, "k_id"):
            self.k_id.set_task(task_name)

    def compute_score(
        self,
        lazy_renders,
        poses,
        env=None,
        base_pos=None,
        base_heading=None,
        check_fov=None,
    ):
        num_samples = len(lazy_renders)
        device = poses.device
        padded_rendered_images = np.zeros(
            [num_samples, len(self.obs_keys), self.image_size[0], self.image_size[1], 3]
        )
        final_scores = torch.ones(num_samples, dtype=torch.float32, device=device) * (
            -3
        )

        # Compute scores from k_col (without rendering)
        if not self.skip_collision:
            score_col = self.k_col.compute_score(
                None, None, poses, collision_threshold_multiplier=self.collision_threshold_multiplier
            )
        else:
            score_col = torch.ones_like(final_scores)

        # Compute fov check if needed
        if check_fov is not None:
            assert base_pos is not None and base_heading is not None
            base_pos_torch = torch.tensor(
                base_pos, dtype=torch.float32, device=poses.device
            )
            curr_to_target_vec = poses[:, :2] - base_pos_torch[:2][None]
            curr_to_target_dir = torch.atan2(
                curr_to_target_vec[:, 1], curr_to_target_vec[:, 0]
            )
            score_fov = (
                torch.abs(
                    (curr_to_target_dir - base_heading + torch.pi) % (2 * torch.pi)
                    - torch.pi
                )
                < check_fov / 2
            )
            failed_fov_indices = torch.nonzero(score_fov == 0, as_tuple=False).squeeze(
                1
            )
            final_scores[failed_fov_indices] = -3

        # Mark scores as -2 where k_col failed
        failed_col_indices = torch.nonzero(score_col == 0, as_tuple=False).squeeze(1)
        final_scores[failed_col_indices] = -2
        if check_fov is not None:
            score_col *= score_fov

        # Filter samples where score_col == 1
        col_indices = torch.nonzero(score_col == 1, as_tuple=False).squeeze(1)
        if len(col_indices) == 0:
            return (
                final_scores,
                [final_scores for obs_key in self.obs_keys],
                padded_rendered_images,
            )

        filtered_lazy_renders = [lazy_renders[i] for i in col_indices]
        filtered_poses = poses[col_indices]

        # Evaluate score_k_obj for filtered samples
        rendered_images_raw, edited_rendered_images_raw = zip(
            *[func() for func in filtered_lazy_renders]
        )
        rendered_images_raw, edited_rendered_images_raw = torch.stack(
            rendered_images_raw
        ), torch.stack(edited_rendered_images_raw)
        rendered_images = rendered_images_raw.permute(0, 1, 4, 2, 3)
        edited_rendered_images = edited_rendered_images_raw.permute(
            0, 1, 4, 2, 3
        )  # (N, V, C, H, W)

        combined_scores_per_view = []
        for obs_key_idx, obs_key in enumerate(self.obs_keys):
            combined_scores = torch.zeros(
                len(rendered_images), dtype=torch.float32, device=device
            )

            obj_masks_torch = None
            if hasattr(self, "k_obj"):
                score_k_obj, obj_masks = self.k_obj.compute_score(
                    rendered_images[:, [obs_key_idx]],
                    edited_rendered_images[:, [obs_key_idx]],
                    filtered_poses,
                )
            else:
                score_k_obj = torch.ones(
                    [len(rendered_images)], device=rendered_images.device
                )

            # Mark scores as -1 where k_obj failed
            failed_obj_indices = torch.nonzero(
                score_k_obj == 0, as_tuple=False
            ).squeeze(1)
            if len(failed_obj_indices) > 0:
                combined_scores[failed_obj_indices] = -1

            # Filter samples where score_k_obj == 1
            obj_indices = torch.nonzero(score_k_obj == 1, as_tuple=False).squeeze(1)
            if len(obj_indices) > 0:
                # Compute scores from k_id
                self.k_id.set_obs_key(obs_key)
                score_id = self.k_id.compute_score(
                    rendered_images[obj_indices],
                    edited_rendered_images[obj_indices],
                    filtered_poses[obj_indices],
                    masks=obj_masks_torch[obj_indices] if obj_masks_torch is not None else None
                )
                combined_scores[obj_indices] = score_id

            combined_scores_for_view = final_scores.clone()
            combined_scores_for_view[col_indices] = combined_scores
            combined_scores_per_view.append(combined_scores_for_view)

            final_scores[col_indices] = torch.max(
                final_scores[col_indices], combined_scores
            )

        padded_rendered_images[col_indices.detach().cpu().numpy()] = (
            rendered_images_raw.detach().cpu().numpy()
        )

        return final_scores, combined_scores_per_view, padded_rendered_images
