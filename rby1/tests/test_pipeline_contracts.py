"""Pipeline control-flow checks; fake servos do not establish physical success.

Legacy imports deliberately allow this same test file to run on release too.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from mobipi.scripts import run_rby1_pipeline as pipeline
from mobipi.scripts import evaluate_rby1_pipeline as evaluation
from mobipi.utils import rby1_utils as navigation


class PipelineContracts(unittest.TestCase):
    def arm_fixture(self, failure=None):
        names = [f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]
        ik = SimpleNamespace(joint_names=names, solve=Mock())
        left = SimpleNamespace(joint_names=names[:7], qpos_index=np.arange(7), input_type="absolute")
        right = SimpleNamespace(joint_names=names[7:], input_type="absolute")
        controller = SimpleNamespace(part_controllers={"left": left, "right": right}, joint_action_policy=ik)
        sim = SimpleNamespace(data=SimpleNamespace(qpos=np.arange(14) * .01),
                              model=SimpleNamespace(get_joint_qpos_addr=names.index))
        env = SimpleNamespace(robots=[SimpleNamespace(composite_controller=controller)], sim=sim,
                              control_freq=2, control_timestep=.5, done=True)

        def step(action):
            if failure == "exception":
                raise RuntimeError("synthetic step failure")
            sim.data.qpos[:] = ik.solve(action)
            if failure == "task_undone":
                env.done = False

        env.step = step
        task = SimpleNamespace(progress_fn=lambda env: lambda: float(env.done), success_threshold=.5)
        return env, task, ik

    def test_arm_return_restores_solver_and_holds_other_arm(self):
        env, task, ik = self.arm_fixture()
        original = ik.solve
        right = env.sim.data.qpos[7:].copy()
        with patch.object(navigation, "hold_action", return_value=np.zeros(1)), \
                patch.object(navigation, "get_base_vec", return_value=np.zeros(3)), \
                patch.object(navigation, "robot_env_contacts", return_value=[]):
            result = pipeline.return_arm_to_start(env, task, np.zeros(7), {})
        self.assertIs(ik.solve, original)
        self.assertTrue(result["success"])
        self.assertEqual(result["steps_executed"], 8)
        np.testing.assert_array_equal(env.sim.data.qpos[7:], right)

    def test_arm_return_stops_when_task_undone_and_restores_after_error(self):
        for failure in ("task_undone", "exception"):
            env, task, ik = self.arm_fixture(failure)
            original = ik.solve
            with patch.object(navigation, "hold_action", return_value=np.zeros(1)), \
                    patch.object(navigation, "get_base_vec", return_value=np.zeros(3)), \
                    patch.object(navigation, "robot_env_contacts", return_value=[]):
                if failure == "exception":
                    with self.assertRaisesRegex(RuntimeError, "synthetic"):
                        pipeline.return_arm_to_start(env, task, np.zeros(7), {})
                else:
                    result = pipeline.return_arm_to_start(env, task, np.zeros(7), {})
                    self.assertFalse(result["success"])
                    self.assertEqual(result["steps_executed"], 1)
            self.assertIs(ik.solve, original)

    def test_navigation_retains_collision_checker_and_approach_mode(self):
        env, target, collision, bounds, grippers = (object() for _ in range(5))
        with patch.object(navigation, "move_to_pose", return_value="arrived") as move:
            self.assertEqual(pipeline.navigate_to_pose(env, target, collision, bounds, grippers, "velocity"), "arrived")
            move.assert_called_once_with(env, target, collision, bounds, render_camera=None,
                                         gripper_action=grippers, mode="velocity", use_approach=True)

    def test_result_export_counts_finished_trials_including_timeouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = {0: dict(seed=0, status="success", success=True),
                    1: dict(seed=1, status="timeout", success=False),
                    2: dict(seed=2, status="interrupted", success=False)}
            summary = evaluation.save_summary(output, [0, 1, 2, 3], rows, "synthetic")
            self.assertEqual(summary["success_rate"], .5)
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["pending_seeds"], [3])
            self.assertEqual(json.loads((output / "summary.json").read_text()), summary)
            self.assertEqual(len((output / "results.csv").read_text().splitlines()), 4)


class VirtualBaseContracts(unittest.TestCase):
    def test_base_pose_hold_posture_and_camera_calibration(self):
        import robosuite
        import robocasa  # noqa: F401
        from robosuite.controllers import load_composite_controller_config
        from robosuite.models.bases import BASE_MAPPING
        from mobipi.collect_demos.faucet_runtime import CONTROLLER

        if "RBY1MobileBase" not in BASE_MAPPING:
            self.skipTest("Installed RoboSuite lacks RBY1MobileBase; use the documented mobile-model environment")

        env = robosuite.make(
            "CloseSingleDoor", robots="RBY1", base_types="RBY1MobileBase",
            controller_configs=load_composite_controller_config(controller=str(CONTROLLER), robot="RBY1"),
            layout_ids=[1], style_ids=[10], initialization_noise=None, seed=0,
            randomize_base_init_pose=None, randomize_cameras=False,
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        )
        try:
            env.reset()
            target = navigation.get_base_vec(env.sim) + [.01, -.01, .01]
            target[2] = navigation.angle_wrap(target[2])
            navigation.set_base_vec(env.sim, target)
            np.testing.assert_allclose(navigation.get_base_vec(env.sim), target, atol=1e-9)
            navigation.set_task_pose(env, "CloseDrawerLeft")
            for name, value in zip(navigation.HEAD_JOINTS, navigation.HEAD_POSES["CloseDrawerLeft"]):
                self.assertAlmostEqual(float(env.sim.data.get_joint_qpos(name)), value)
            action = navigation.hold_action(env)
            self.assertEqual(action.shape, (env.action_dim,))
            self.assertTrue(np.isfinite(action).all())
            position, rotation = navigation.compute_relative_cam_pose(env.sim, "robot0_head_camera")
            self.assertTrue(np.isfinite(position).all())
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-9)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
