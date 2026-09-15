import numpy as np

# We inherit from ManipulatorModel as done in baxter_robot.py
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.models.bases.mount_model import MountModel
from robosuite.models.bases import register_base
from robosuite.utils.mjcf_utils import xml_path_completion


@register_base
class RBY1Mount(MountModel):
    """Reference mount for the base site embedded in the RBY1a MJCF."""

    def __init__(self, idn=0):
        super().__init__(xml_path_completion("bases/rby1_mount.xml"), idn=idn)

    @property
    def top_offset(self):
        return np.zeros(3)

    @property
    def horizontal_radius(self):
        # Used by RoboCasa to keep the complete mobile base clear of fixtures.
        return 0.6


class RBY1(ManipulatorModel):
    """
    RBY-1 is a bimanual mobile manipulator.

    Args:
        idn (int or str): Number or some other unique identification string for this robot instance
    """

    # Explicitly declare it has two arms like Baxter
    arms = ["right", "left"]

    def __init__(self, idn=0):
        # Ensure you put your rby1.xml in the robosuite/models/assets/robots/rby1/ directory
        super().__init__(xml_path_completion("robots/rby1a/rby1a_1.2.xml"), idn=idn)
        self._convert_torque_controlled_actuators()

    def _convert_torque_controlled_actuators(self):
        """Convert arm, torso, and head position targets to torque actuators.

        robosuite's joint and operational-space controllers output torques. The
        source RBY1a MJCF uses position actuators for these joints, so passing
        controller output directly to the original actuators interprets torque
        commands as joint-position targets.
        """
        for actuator in self._elements["actuators"]:
            name = actuator.get("name", "")
            raw_name = name.removeprefix(f"robot{self.idn}_")
            if not raw_name.startswith(("link", "right_arm_", "left_arm_", "head_")):
                continue
            force_range = actuator.get("forcerange")
            if force_range is None:
                continue
            actuator.tag = "motor"
            actuator.set("ctrlrange", force_range)
            for attribute in ("forcelimited", "forcerange", "kp", "kv"):
                actuator.attrib.pop(attribute, None)

    def update_joints(self):
        """Partition RBY1a joints by the components used by robosuite controllers."""
        self._base_joints = []
        self._torso_joints = []
        self._head_joints = []
        self._legs_joints = []
        self._arms_joints = []

        for joint in self.all_joints:
            raw_name = joint.removeprefix(f"robot{self.idn}_")
            if raw_name in {"left_wheel", "right_wheel"}:
                self._base_joints.append(joint)
            elif raw_name.startswith("torso_"):
                self._torso_joints.append(joint)
            elif raw_name.startswith("head_"):
                self._head_joints.append(joint)
            elif raw_name.startswith(("right_arm_", "left_arm_")):
                self._arms_joints.append(joint)

    def update_actuators(self):
        """Partition RBY1a actuators without treating embedded fingers as arm actuators."""
        self._base_actuators = []
        self._torso_actuators = []
        self._head_actuators = []
        self._legs_actuators = []
        self._arms_actuators = []

        for actuator in self.all_actuators:
            raw_name = actuator.removeprefix(f"robot{self.idn}_")
            if raw_name in {"left_wheel_act", "right_wheel_act"}:
                self._base_actuators.append(actuator)
            elif raw_name.startswith("link") and raw_name.endswith("_act"):
                self._torso_actuators.append(actuator)
            elif raw_name.startswith(("head_", "head")) and raw_name.endswith("_act"):
                self._head_actuators.append(actuator)
            elif raw_name.startswith(("right_arm_", "left_arm_")):
                self._arms_actuators.append(actuator)

    @property
    def default_base(self):
        # The MJCF already contains the robot's mobile base and wheel joints.
        return "RBY1Mount"

    @property
    def default_gripper(self):
        """
        Returns dict with 'right', 'left' keywords for the arm-specific gripper names.
        """
        return {"right": "RBY1Gripper", "left": "RBY1Gripper"}

    @property
    def default_controller_config(self):
        """
        Returns dict with 'right', 'left' default controller configs.
        """
        return {"right": "osc_pose", "left": "osc_pose"}

    @property
    def init_qpos(self):
        # Keep both arms folded near the torso for a gravity-stable reset pose.
        return np.zeros(28)

    @property
    def base_xpos_offset(self):
        # Spawning offsets relative to environments
        return {
            "bins": (-0.5, -0.1, 0.0),
            "empty": (-0.29, 0, 0.0),
            "table": lambda table_length: (-0.5 - table_length / 2, 0, 0.0),
        }

    @property
    def top_offset(self):
        # Z-axis offset for calculating IK targeting
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.5

    @property
    def arm_type(self):
        # Explicitly define as bimanual like Baxter
        return "bimanual"

    @property
    def _eef_name(self):
        """
        Returns dict with 'right', 'left' keywords mapped to the dummy body names we added in the XML.
        """
        return {"right": "right_hand", "left": "left_hand"}