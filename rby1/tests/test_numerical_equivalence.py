"""Fixed seeded references captured from the merged release before migration."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from rby1.paths import REPOSITORY_ROOT
from rby1.robot import controllers, navigation, sim_utils
from rby1.demos import auto_close_door_general, auto_close_microwave_door


FIXTURES = Path(__file__).with_name("fixtures")


class NumericalEquivalenceTests(unittest.TestCase):
    def test_seeded_pose_helpers_match_release(self):
        cases = json.loads((FIXTURES / "numerical_reference.json").read_text())["geometry"]
        for case in cases:
            a, b, p = (np.asarray(case[key]) for key in ("a", "b", "p"))
            np.testing.assert_allclose(sim_utils.make_T(p, a), case["transform"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(sim_utils.interpolate_rotation(a, b, case["t"]), case["interpolate"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(sim_utils.normalize(p), case["normal"], atol=1e-12, rtol=0)
            self.assertAlmostEqual(sim_utils.smoothstep(case["t"]), case["smooth"], places=14)

    def test_seeded_navigation_paths_match_release(self):
        cases = json.loads((FIXTURES / "numerical_reference.json").read_text())["navigation"]
        for case in cases:
            actual = navigation._diff_drive_poses(case["start"], case["goal"], case["heading"])
            np.testing.assert_allclose(actual, case["poses"], atol=1e-12, rtol=0)

    def test_configuration_bytes_match_release(self):
        hashes = json.loads((FIXTURES / "configuration_hashes.json").read_text())
        for path, expected in hashes.items():
            with self.subTest(config=path):
                self.assertEqual(hashlib.sha256((REPOSITORY_ROOT / path).read_bytes()).hexdigest(), expected)

    def test_planners_reexport_one_implementation(self):
        names = ("raw_model", "raw_data", "forward", "normalize", "smoothstep", "site_pose",
                 "make_T", "interpolate_rotation", "arm_joint_names", "force_tangent_direction")
        for name in names:
            for module in (auto_close_door_general, auto_close_microwave_door):
                self.assertIs(getattr(module, name), getattr(sim_utils, name))
        for module in (auto_close_door_general, auto_close_microwave_door):
            self.assertIs(module.set_ik_q0_from_current_pose, controllers.set_ik_q0_from_current_pose)
        for name in ("build_controller_interface", "create_robot_action", "pose_target_for_ref", "execute_target"):
            self.assertIs(getattr(auto_close_microwave_door, name), getattr(controllers, name))

    def test_simulator_wrappers_and_forward_fallback(self):
        model, data = object(), object()
        wrapped = SimpleNamespace(model=SimpleNamespace(_model=model), data=SimpleNamespace(_data=data))
        self.assertIs(sim_utils.raw_model(wrapped), model)
        self.assertIs(sim_utils.raw_data(wrapped), data)
        wrapped.forward = lambda: (_ for _ in ()).throw(RuntimeError("wrapper unavailable"))
        with patch.object(sim_utils.mujoco, "mj_forward") as forward:
            sim_utils.forward(wrapped)
            forward.assert_called_once_with(model, data)
        self.assertIs(sim_utils.raw_model(SimpleNamespace(model=model)), model)
        with self.assertRaises(RuntimeError):
            sim_utils.normalize([0, 0, 0])


if __name__ == "__main__":
    unittest.main()
