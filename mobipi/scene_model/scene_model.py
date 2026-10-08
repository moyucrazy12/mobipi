"""
Utilities for running inference using a trained 3DGS model.

@yjy0625
"""
import os
import json
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup  # Assume this is imported correctly


class SceneModel:
    def __init__(self, ckpt_dir, camera_intrinsics):
        self.ckpt_dir = ckpt_dir

        config_path = os.path.join(ckpt_dir, "config.yml")
        config, pipeline, checkpoint_path, step = eval_setup(
            Path(config_path), update_config_callback=self._relocate_config
        )
        pipeline.train()
        self.model = pipeline.model
        self.model.training = False
        self.config = config

        with open(os.path.join(ckpt_dir, "dataparser_transforms.json")) as f:
            dataparser_transforms = json.load(f)
        self.data_transforms = dataparser_transforms

        self.camera_intrinsics = camera_intrinsics

    def _relocate_config(self, config):
        # Downloaded checkpoints store absolute paths from the machine they were
        # trained on; point them at the scene folder the checkpoint was loaded from,
        # i.e. [scene_dir]/model/[method]/[timestamp].
        run_dir = Path(self.ckpt_dir).resolve()
        scene_dir = run_dir.parents[2]
        config.output_dir = run_dir.parents[1]
        config.data = scene_dir
        config.pipeline.datamanager.data = scene_dir
        return config

    def render(self, extrinsics, image_size=128):
        device = extrinsics.device
        data_mat = torch.tensor(self.data_transforms["transform"], requires_grad=True).to(extrinsics.dtype).to(device)
        data_scale = torch.tensor(self.data_transforms["scale"], requires_grad=True).to(extrinsics.dtype).to(device)
        model_extrinsics = data_mat @ extrinsics * data_scale

        num_cameras = len(model_extrinsics)

        scale_factor = image_size / self.camera_intrinsics["w"]

        width = int(self.camera_intrinsics["w"] * scale_factor)
        height = int(self.camera_intrinsics["h"] * scale_factor)

        camera = Cameras(
            camera_to_worlds=model_extrinsics[:, :3],
            fx=torch.tensor([self.camera_intrinsics["fl_x"] * scale_factor] * num_cameras).to(device),
            fy=torch.tensor([self.camera_intrinsics["fl_y"] * scale_factor] * num_cameras).to(device),
            cx=torch.tensor([self.camera_intrinsics["cx"] * scale_factor] * num_cameras).to(device),
            cy=torch.tensor([self.camera_intrinsics["cy"] * scale_factor] * num_cameras).to(device),
            width=torch.tensor([width] * num_cameras).to(device),
            height=torch.tensor([height] * num_cameras).to(device),
            camera_type=CameraType.PERSPECTIVE
        )
        camera.to(extrinsics.device)

        model_outputs = self.model.get_outputs(camera)
        return model_outputs["rgb"]

    def render_topdown(self, center, extent, resolution):
        """
        Renders a top-down orthographic view of the scene.

        Args:
            center (tuple): (x, y) center of the view.
            extent (tuple): (x_size, y_size) extent of the view.
            resolution (tuple): (height, width) resolution of the rendered image.

        Returns:
            torch.Tensor: Rendered top-down view as an RGB image.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        height, width = resolution
        x_size, y_size = extent
        x_center, y_center = center

        # Create orthographic camera extrinsics
        camera_to_world = torch.eye(4).to(device)
        camera_to_world[0, 3] = x_center
        camera_to_world[1, 3] = y_center
        camera_to_world[2, 3] = 10.0  # Height above the scene (adjust if needed)
        camera_to_world[2, 2] = -1.0  # Looking down along -z axis

        # Orthographic camera parameters
        fx = width / x_size  # Scale factor for x
        fy = height / y_size  # Scale factor for y
        cx = width / 2
        cy = height / 2

        camera = Cameras(
            camera_to_worlds=camera_to_world[None, :3],
            fx=torch.tensor([fx]).to(device),
            fy=torch.tensor([fy]).to(device),
            cx=torch.tensor([cx]).to(device),
            cy=torch.tensor([cy]).to(device),
            width=torch.tensor([width]).to(device),
            height=torch.tensor([height]).to(device),
            camera_type=CameraType.ORTHOGRAPHIC  # Use orthographic projection
        )
        camera.to(device)

        model_outputs = self.model.get_outputs(camera)
        return model_outputs["rgb"]


class BatchSceneModel:
    def __init__(self, ckpt_dir, camera_intrinsics):
        self.ckpt_dir = ckpt_dir
        self.camera_intrinsics = camera_intrinsics
        self.num_threads = len(camera_intrinsics)

        # Pre-initialize models
        self.models = [SceneModel(ckpt_dir, camera_intrinsics[i]) for i in range(self.num_threads)]

    def render(self, extrinsics_list, image_size=128):
        results = []

        with torch.no_grad():
            for i, extrinsics in enumerate(extrinsics_list):
                results.append(self.models[i].render(extrinsics[None], image_size))

        return torch.stack(results)

    def _render_single(self, thread_id, extrinsics, image_size):
        # Use the pre-initialized model for the thread
        model = self.models[thread_id % self.num_threads]
        return model.render(extrinsics[None], image_size)
