"""Bounded IK, reset preservation and command contracts on the real robot model."""
import json
from pathlib import Path
import unittest

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

from mobipi.collect_demos.faucet_runtime import CONTROLLER, MG_RBY1TurnOnSinkFaucet, raw_data, UnsafeTrajectory
from mobipi.collect_demos.faucet_bounded_ik import bind_bounded_ik


class BoundedIKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path(__file__).parents[1].joinpath('faucet_config.json').read_text())
        kwargs = dict(cls.config['environment'])
        for key in ('camera_names', 'camera_heights', 'camera_widths'):
            kwargs.pop(key, None)
        cls.env = robosuite.make('TurnOnSinkFaucetRBY1', **kwargs,
            controller_configs=load_composite_controller_config(controller=str(CONTROLLER), robot='RBY1'),
            seed=0, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        cls.env.reset()
        cls.interface = MG_RBY1TurnOnSinkFaucet(cls.env)
        cls.solver = bind_bounded_ik(cls.interface)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_reset_preserves_bent_arm(self):
        actual = raw_data(self.env.sim).qpos[self.solver.addresses]
        np.testing.assert_allclose(actual, self.config['environment']['faucet_initial_arm_qpos'], atol=.04)
        self.assertLess(actual[3], -.5)
        self.solver.check_scene(raw_data(self.env.sim))
        self.assertFalse(self.env._check_success())

    def test_joint_limits_are_optimizer_constraints(self):
        target = self.interface.get_robot_eef_pose()
        target[:3, 3] += [-.003, -.003, .006]
        q = self.solver.optimize(target)
        self.assertTrue(np.all(q >= self.solver.low))
        self.assertTrue(np.all(q <= self.solver.high))
        self.assertLess(self.solver.last['position_error'], .003)
        self.assertLess(self.solver.last['orientation_error'], .03)
        self.solver.check(q)

    def test_optimization_never_changes_live_simulator_state(self):
        before = raw_data(self.env.sim).qpos.copy()
        target = self.interface.get_robot_eef_pose()
        target[2, 3] += .01
        self.solver.optimize(target)
        np.testing.assert_array_equal(raw_data(self.env.sim).qpos, before)

    def test_joint_trajectory_command_uses_existing_pose_action(self):
        q = raw_data(self.env.sim).qpos[self.solver.addresses].copy()
        q[2] += .005
        pose = self.solver.command_joints(q)
        action = self.interface.full_action(pose, -1.)
        decoded = self.interface.action_to_target_pose(action)
        np.testing.assert_allclose(decoded, pose, atol=1e-10)
        solution = self.solver.solve(action[:self.solver.policy.control_dim])
        np.testing.assert_array_equal(solution[self.solver.indices], q)
        self.assertEqual(action.shape, (self.env.action_dim,))

    def test_out_of_limit_or_nonfinite_joint_command_rejected(self):
        with self.assertRaises(UnsafeTrajectory): self.solver.check(self.solver.high+.1)
        with self.assertRaises(UnsafeTrajectory): self.solver.check(np.full(7, np.nan))


if __name__ == '__main__':
    unittest.main()
