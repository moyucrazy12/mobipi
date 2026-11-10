"""Simulator control guards using an explicitly supplied external task config."""
import json
import os
from pathlib import Path
import unittest

import numpy as np


@unittest.skipUnless(os.environ.get("RBY1_TEST_CONFIG"), "Set RBY1_TEST_CONFIG to the external faucet task JSON")
class FaucetControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import robosuite
        from robosuite.controllers import load_composite_controller_config
        from rby1.robot.faucet_runtime import CONTROLLER, MG_RBY1TurnOnSinkFaucet
        from rby1.robot.faucet_bounded_ik import bind_bounded_ik
        kwargs = dict(json.loads(Path(os.environ["RBY1_TEST_CONFIG"]).expanduser().read_text())["environment"])
        for key in ("camera_names", "camera_heights", "camera_widths"):
            kwargs.pop(key, None)
        cls.env = robosuite.make("TurnOnSinkFaucetRBY1", **kwargs,
            controller_configs=load_composite_controller_config(controller=str(CONTROLLER), robot="RBY1"),
            seed=0, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        cls.addClassCleanup(cls.env.close)
        cls.env.reset()
        cls.interface = MG_RBY1TurnOnSinkFaucet(cls.env)
        cls.solver = bind_bounded_ik(cls.interface)

    def test_ik_respects_joint_limits_without_mutating_live_state(self):
        from rby1.robot.faucet_runtime import raw_data, UnsafeTrajectory
        before = raw_data(self.env.sim).qpos.copy()
        target = self.interface.get_robot_eef_pose()
        target[2, 3] += .01
        q = self.solver.optimize(target)
        self.assertTrue(np.all(q >= self.solver.low) and np.all(q <= self.solver.high))
        np.testing.assert_array_equal(raw_data(self.env.sim).qpos, before)
        with self.assertRaises(UnsafeTrajectory):
            self.solver.check(self.solver.high + .1)
        with self.assertRaises(UnsafeTrajectory):
            self.solver.check(np.full(7, np.nan))

    def test_tcp_command_and_gripper_survive_whole_body_packing(self):
        pose = self.interface.get_robot_eef_pose()
        for gripper in (-1., 1.):
            action = self.interface.full_action(pose, gripper)
            self.assertEqual(action.shape, (self.env.action_dim,))
            np.testing.assert_allclose(self.interface.action_to_target_pose(action), pose, atol=1e-10)
            np.testing.assert_array_equal(self.interface.action_to_gripper_action(action), [gripper])
