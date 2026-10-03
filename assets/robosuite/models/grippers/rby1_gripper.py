import numpy as np

from robosuite.models.grippers.gripper_model import GripperModel
from robosuite.utils.mjcf_utils import xml_path_completion


class RBY1Gripper(GripperModel):
    """Logical gripper adapter for the finger actuators embedded in RBY1a."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("grippers/rby1_gripper.xml"), idn=idn)
        self.arm = str(idn).rsplit("_", 1)[-1]
        self.robot_id = str(idn).split("_", 1)[0]

    @property
    def dof(self):
        return 1

    @property
    def joints(self):
        finger = "r1" if self.arm == "right" else "l1"
        return [f"robot{self.robot_id}_gripper_finger_{finger}"]

    @property
    def actuators(self):
        finger = "right" if self.arm == "right" else "left"
        return [f"robot{self.robot_id}_{finger}_finger_act"]

    @property
    def init_qpos(self):
        return np.zeros(1)

    def format_action(self, action):
        action = np.asarray(action)
        assert len(action) == self.dof
        return action
