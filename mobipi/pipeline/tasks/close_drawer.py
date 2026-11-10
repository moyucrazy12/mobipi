"""CloseDrawerLeft (mobipi/envs/kitchen_drawer.py); progress is minus the drawer opening."""

import numpy as np

from mobipi.pipeline.tasks.base import ROOT, PipelineTask

# RoboCasa's CloseDrawer succeeds when every drawer joint is at most 5 % open
SUCCESS_MAX_OPENING = 0.05


def find_scene_open_drawer(env, ply_path, min_points=200):
    """
    The drawer that is open in a scene model's capture, and its opening (fraction
    of RoboCasa's fully open drawer, 0.55 x drawer depth), from the scene's point
    cloud: an open drawer leaves points in front of its cabinet face between the
    drawer's bottom and top, a closed one only its handle (~5 cm).
    """
    import open3d as o3d
    from robocasa.models.fixtures.cabinets import Drawer

    pc = np.asarray(o3d.io.read_point_cloud(str(ply_path)).points)
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
        raise RuntimeError(f"Found no open drawer in {ply_path}; pass --no-scene-overrides to sample drawers instead.")
    if len(candidates) > 1:
        print(f"Several open drawer candidates in {ply_path}: {sorted(candidates, reverse=True)}; using the densest.")
    _, name, opening = max(candidates)
    return name, opening


class CloseDrawer(PipelineTask):
    name = "close_drawer"
    env_name = "CloseDrawerLeft"
    default_checkpoint = (ROOT / "external/robomimic/experiment_results/rby1_close_drawer_cartesian_cmd_look300"
                          / "20261020114821/models/model_epoch_10.pth")
    default_dataset = ROOT / "datasets/close_drawer_look/close_drawer_look_300_im128_cmd.hdf5"
    progress_label = "-opening"
    success_definition = "Navigation, stable drawer closure, and arm return all succeed"
    success_threshold = -SUCCESS_MAX_OPENING
    # The demos start 0.23-0.33 m from the nearest scene point, at the edge of
    # the default 0.5 x 0.44 m scoring radius.
    collision_threshold_multiplier = 0.4

    def scene_overrides(self, env, scene_dir):
        # The scene models are static captures with the drawer that RoboCasa happened
        # to open baked in. Make that drawer the target, opened as far as in the capture.
        drawer_name, opening = find_scene_open_drawer(env, scene_dir / "pc.ply")
        print(f"Open drawer in the scene model: {drawer_name} (opening {opening:.2f}); using it as the target",
              flush=True)
        return dict(drawer_id=drawer_name, drawer_open_range=(opening, opening))

    def progress_fn(self, env):
        drawer = getattr(env, "drawer", None)
        if drawer is None or env.behavior != "close":
            raise ValueError("This checkpoint expects the drawer-closing task")
        return lambda: -float(max(drawer.get_door_state(env=env).values()))
