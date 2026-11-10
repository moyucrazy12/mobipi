"""Real MuJoCo interface checks; no successful faucet rollout is implied."""
import json
from pathlib import Path
import unittest

import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config
from mobipi.collect_demos.faucet_runtime import (
    CONTROLLER, MG_RBY1TurnOnSinkFaucet, UnsafeTrajectory, raw_data,
)


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = robosuite.make(
            "TurnOnSinkFaucetRBY1", robots="RBY1", gripper_types="RBY1FaucetGripper",
            controller_configs=load_composite_controller_config(controller=str(CONTROLLER), robot="RBY1"),
            layout_ids=[1], style_ids=[10], initialization_noise=None,
            randomize_base_init_pose=None, randomize_cameras=False, seed=0,
            has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        cls.env.reset()
        cls.interface = MG_RBY1TurnOnSinkFaucet(cls.env)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_absolute_command_roundtrip(self):
        pose = self.interface.get_robot_eef_pose()
        pose[:3, 3] += [.01, -.01, .01]
        action = self.interface.full_action(pose, -1.)
        self.assertEqual(action.shape, (self.env.action_dim,))
        np.testing.assert_allclose(self.interface.action_to_target_pose(action), pose, atol=1e-12)
        np.testing.assert_array_equal(self.interface.action_to_gripper_action(action), [-1.])

    def test_ik_holds_pose_with_free_root(self):
        policy = self.interface._policy
        self.assertFalse(np.array_equal(policy.joint_ids, policy.qpos_addresses))
        self.assertFalse(np.array_equal(policy.joint_ids, policy.velocity_addresses))
        target = self.interface.get_robot_eef_pose()
        action = self.interface.full_action(target, 1.)
        desired = policy.solve(action[:policy.control_dim])
        np.testing.assert_allclose(desired, raw_data(self.env.sim).qpos[policy.qpos_addresses], atol=1e-9)

    def test_correct_faucet_reader_and_schema(self):
        sink = self.env.sink
        self.assertEqual(float(self.env.sim.data.get_joint_qpos(sink.naming_prefix+"handle_joint")), 0.)
        self.assertEqual(sink.get_handle_state(self.env)["handle_joint"], 0.)
        self.assertFalse(self.env._check_success())
        action = self.interface.full_action(self.interface.get_robot_eef_pose(), 1.)
        info = self.interface.get_datagen_info(action).to_dict()
        self.assertTrue({"eef_pose", "target_pose", "gripper_action", "object_poses", "subtask_term_signals"} <= set(info))
        self.assertEqual(set(info["subtask_term_signals"]), {"contact", "water_on"})

    def test_safety_covers_arm_not_wheels(self):
        geometry = self.interface.geom
        bodies = [self.env.sim.model.body_id2name(int(geometry.m.geom_bodyid[g])) for g in geometry.arm_geoms]
        self.assertTrue(any("link_right_arm_3" in b for b in bodies))
        self.assertFalse(any("wheel" in b for b in bodies))
        self.assertTrue(geometry.basins)
        geometry.check()

    def test_nonfinite_command_rejected(self):
        pose = np.eye(4)
        pose[0, 3] = np.nan
        with self.assertRaises(UnsafeTrajectory):
            self.interface.full_action(pose, 1.)

    def test_task_spec_matches_installed_mimicgen(self):
        from mimicgen.configs.task_spec import MG_TaskSpec
        config = json.loads(Path(__file__).parents[1].joinpath("faucet_config.json").read_text())
        spec = MG_TaskSpec.from_json(json_dict=config["mimicgen"]["task_spec"])
        self.assertEqual(len(spec), 3)
        self.assertTrue(all(s["object_ref"] == "faucet" for s in spec))


class ExecutorAdapterTests(unittest.TestCase):
    def test_absolute_targets_not_clipped_and_executor_restored(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from mimicgen.datagen.waypoint import Waypoint, WaypointSequence, WaypointTrajectory
        from mobipi.collect_demos.faucet_pipeline import rby1_waypoint_executor

        class FakeRecorder:
            def __init__(self, env, interface, settings, video, log, tracking):
                self.rows = []
                self.datagen_infos = []
                self.tracking = SimpleNamespace(write=lambda info: None)
                self.bounded = SimpleNamespace(
                    command=interface.previous, addresses=np.arange(7),
                    optimize=lambda pose, seed=None: np.full(7, interface.delta),
                    command_joints=lambda q: pose,
                    last={"position_error": interface.residual, "orientation_error": 0.})

            def step(self, pose, gripper, phase):
                self.rows.append(dict(states=np.zeros(3), obs={}, actions=pose[:3, 3].copy()))
                self.datagen_infos.append(gripper.copy())

            def transition(self, q, gripper, phase):
                self.step(pose, gripper, phase)
                self.step(pose, gripper, phase)

        env = SimpleNamespace(is_success=lambda: {"task": True})
        pose = np.eye(4)
        pose[:3, 3] = [2.75, -2.4, 1.1]
        trajectory = WaypointTrajectory()
        trajectory.add_waypoint_sequence(WaypointSequence([
            Waypoint(pose, np.array([1.]), noise=0.),
            Waypoint(pose, np.array([-1.]), noise=0.)]))
        original = WaypointTrajectory.execute
        for delta, previous, expected_steps, residual, enforce in ((0., None, 2, 0., True), (.1, None, 4, 0., True), (.1, np.full(7, .1), 2, 0., True), (0., None, 2, .01, False)):
            interface = SimpleNamespace(refresh=lambda: None, delta=delta, previous=previous, residual=residual,
                env=SimpleNamespace(sim=None, control_timestep=.05))
            with patch("mobipi.collect_demos.faucet_pipeline.JointSpaceRecorder", FakeRecorder), \
                 patch("mobipi.collect_demos.faucet_pipeline.raw_data", return_value=SimpleNamespace(qpos=np.zeros(7))):
                with rby1_waypoint_executor({"contact_position_tolerance": .005, "joint_velocity": .4, "require_cartesian_tolerance": enforce}, None, None):
                    result = trajectory.execute(env, interface)
            self.assertIs(WaypointTrajectory.execute, original)
            np.testing.assert_array_equal(result["actions"][0], [2.75, -2.4, 1.1])
            self.assertEqual(len(result["actions"]), expected_steps)
            self.assertEqual(len(result["datagen_infos"]), expected_steps)
            # Early task success must not skip the later release command.
            np.testing.assert_array_equal(result["datagen_infos"][-1], [-1.])

    def test_executor_restored_after_exception(self):
        from mimicgen.datagen.waypoint import WaypointTrajectory
        from mobipi.collect_demos.faucet_pipeline import rby1_waypoint_executor
        original = WaypointTrajectory.execute
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with rby1_waypoint_executor({}, None, None):
                raise RuntimeError("synthetic")
        self.assertIs(WaypointTrajectory.execute, original)


if __name__ == "__main__":
    unittest.main()
