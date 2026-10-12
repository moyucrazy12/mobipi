"""
RoboCasa drawer tasks restricted to one robot placement.

RoboCasa's ManipulateDrawer places the robot beside the drawer and picks the
side at random. The RB-Y1 demos only use the left arm, which can only reach a
drawer on the robot's left, so these variants always place the robot to the
right of the drawer, and reset into the task's squat / head pose
(rby1_utils.TORSO_POSES, HEAD_POSES). How far the drawer starts open is
configurable (drawer_open_range); RoboCasa's randomize_base_init_pose adds
noise to the robot placement. Importing this module registers the
environments.
"""
import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.environments.kitchen.single_stage.kitchen_drawer import CloseDrawer

from mobipi.utils import rby1_utils


class CloseDrawerLeft(CloseDrawer):
    """
    CloseDrawer with the drawer always on the robot's left ("close the left drawer").

    Args:
        drawer_open_range (2-tuple): range the initial drawer opening is sampled
            from, as fractions of RoboCasa's fully open drawer (1.0 is 55% of the
            drawer depth). RoboCasa's CloseDrawer uses (0.9, 1.0).
        randomize_base_init_pose (float or 3-sequence): std of the Gaussian noise
            added to the robot base x [m], y [m] and yaw [rad] (RoboCasa kwarg).
    """

    def __init__(self, *args, drawer_open_range=(0.9, 1.0), **kwargs):
        lo, hi = drawer_open_range
        assert 0.0 <= lo <= hi <= 1.0, drawer_open_range
        self.drawer_open_range = (float(lo), float(hi))
        super().__init__(*args, **kwargs)

    # Scenes sampled per reset before giving up when none has room for the robot
    MAX_SCENE_SAMPLES = 20

    def _load_model(self):
        # Same as ManipulateDrawer._load_model with only the placement to the
        # right of the drawer. When that side is blocked, sample another scene
        # (as Kitchen._load_model does when objects cannot be placed), so a
        # plain reset() always succeeds. A scene restored from ep_meta is
        # fixed, so it raises instead.
        for _ in range(self.MAX_SCENE_SAMPLES):
            Kitchen._load_model(self)
            x_ofs = (self.drawer.width / 2) + 0.3
            robot_base_pos, robot_base_ori = self.compute_robot_base_placement_pose(
                ref_fixture=self.drawer, offset=(x_ofs, -0.23)
            )
            test_pos, _ = self.compute_robot_base_placement_pose(
                ref_fixture=self.drawer, offset=(x_ofs + 0.3, -0.23)
            )
            if not (self.check_fxtr_contact(test_pos) or self.check_sidewall_contact(test_pos)):
                break
            if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
                raise RuntimeError(
                    f"No room for the robot to the right of drawer {self.drawer.name}; "
                    "this scene cannot be used for CloseDrawerLeft."
                )
        else:
            raise RuntimeError(
                f"No room for the robot to the right of the drawer in {self.MAX_SCENE_SAMPLES} scenes "
                f"sampled from (layout, style) {self.layout_and_style_ids}."
            )
        robot_model = self.robots[0].robot_model
        self.drawer_side = "left"

        if self.place_robot:
            if self.force_robot_placement:
                robot_base_pos, robot_base_ori = self.force_robot_placement
            elif self.randomize_base_init_pose is not None:
                robot_base_vec = [robot_base_pos[0], robot_base_pos[1], robot_base_ori[2]]
                noise = self.randomized_robot_base_pose_rng.normal(
                    loc=0, scale=self.randomize_base_init_pose, size=(3,)
                )
                robot_base_pos = [robot_base_vec[0] + noise[0], robot_base_vec[1] + noise[1], 0]
                robot_base_ori = [0, 0, robot_base_vec[2] + noise[2]]
            robot_base_pos, robot_base_ori = np.asarray(robot_base_pos), np.asarray(robot_base_ori)
            robot_model.set_base_xpos(robot_base_pos)
            robot_model.set_base_ori(robot_base_ori)
            self._init_robot_pos, self._init_robot_ori = robot_base_pos, robot_base_ori

    def _reset_internal(self):
        # ManipulateDrawer._reset_internal with the opening from drawer_open_range.
        # The drawer is opened before Kitchen places the object inside it.
        lo, hi = self.drawer_open_range
        self.drawer.set_door_state(min=lo, max=hi, env=self, rng=self.rng)
        Kitchen._reset_internal(self)
        # Part of the reset so the initial state already has it: recorded demos
        # start in this pose and reset_to() / playback restore it with matching
        # controller goals.
        rby1_utils.set_task_pose(self, "CloseDrawerLeft")
