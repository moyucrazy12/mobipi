"""
RoboCasa drawer tasks restricted to one robot placement.

RoboCasa's ManipulateDrawer places the robot beside the drawer and picks the
side at random. The RB-Y1 demos only use the left arm, which can only reach a
drawer on the robot's left, so these variants always place the robot to the
right of the drawer, and reset into the task's squat / head pose
(rby1_utils.TORSO_POSES, HEAD_POSES). Importing this module registers the
environments.
"""
import numpy as np

from robocasa.environments.kitchen.kitchen import Kitchen
from robocasa.environments.kitchen.single_stage.kitchen_drawer import CloseDrawer

from mobipi.utils import rby1_utils


class CloseDrawerLeft(CloseDrawer):
    """CloseDrawer with the drawer always on the robot's left ("close the left drawer")."""

    def _load_model(self):
        # Same as ManipulateDrawer._load_model with only the placement to the
        # right of the drawer; scenes where it is blocked raise instead of
        # falling back to the other side.
        Kitchen._load_model(self)
        robot_model = self.robots[0].robot_model
        x_ofs = (self.drawer.width / 2) + 0.3
        robot_base_pos, robot_base_ori = self.compute_robot_base_placement_pose(
            ref_fixture=self.drawer, offset=(x_ofs, -0.23)
        )
        test_pos, _ = self.compute_robot_base_placement_pose(
            ref_fixture=self.drawer, offset=(x_ofs + 0.3, -0.23)
        )
        if self.check_fxtr_contact(test_pos) or self.check_sidewall_contact(test_pos):
            raise RuntimeError(
                f"No room for the robot to the right of drawer {self.drawer.name}; "
                "this scene cannot be used for CloseDrawerLeft."
            )
        self.drawer_side = "left"

        if self.place_robot:
            if self.force_robot_placement:
                robot_base_pos, robot_base_ori = self.force_robot_placement
            elif self.randomize_base_init_pose:
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
        super()._reset_internal()
        # Part of the reset so the initial state already has it: recorded demos
        # start in this pose and reset_to() / playback restore it with matching
        # controller goals.
        rby1_utils.set_task_pose(self, "CloseDrawerLeft")
