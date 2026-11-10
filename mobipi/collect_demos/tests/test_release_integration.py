"""Small release-integration checks; these do not measure policy success rates."""
import importlib
import json
from pathlib import Path
import unittest

import numpy as np
import robosuite
import robocasa  # noqa: F401: register tasks
import mobipi.envs  # noqa: F401: register CloseDrawerLeft
from robosuite.controllers import load_composite_controller_config

from mobipi.collect_demos.faucet_runtime import CONTROLLER
from mobipi.collect_demos.pipeline_tasks import TASKS as DEMO_TASKS
from mobipi.pipeline.tasks import get_task
from mobipi.scripts.run_rby1_pipeline import parser, apply_task_defaults


class ConfigurationTests(unittest.TestCase):
    def test_collection_callbacks_and_configurations_resolve(self):
        from mimicgen.configs.task_spec import MG_TaskSpec
        task = DEMO_TASKS["TurnOnSinkFaucet"]
        self.assertEqual(task.active_arm, "right")
        for operation in ("collect", "prepare", "augment", "validate", "attempt_config", "convert"):
            module, name = getattr(task, operation).split(":")
            self.assertTrue(callable(getattr(importlib.import_module(module), name)))
        for path in (task.seed_config, task.augmentation_config):
            config = json.loads(Path(path).read_text())
            self.assertEqual(config["environment"]["camera_names"],
                             ["faucet_overview", "robot0_right_eye_in_hand"])
            self.assertEqual(len(MG_TaskSpec.from_json(json_dict=config["mimicgen"]["task_spec"])), 3)

    def test_mobilization_task_defaults_are_retained(self):
        for name, env_name in (("close_microwave", "CloseSingleDoor"), ("close_drawer", "CloseDrawerLeft")):
            with self.subTest(task=name):
                args = parser().parse_args(["--task", name, "--scene-dir", "/tmp/unused-scene"])
                task = apply_task_defaults(args)
                self.assertIs(task, get_task(name))
                self.assertEqual(task.env_name, env_name)
                self.assertEqual(args.dataset, task.default_dataset)
                self.assertEqual(args.checkpoint, task.default_checkpoint)

    def test_native_mimicgen_task_adapters_are_available(self):
        from mimicgen.env_interfaces.robocasa.single_stage.mg_rby1_close_single_door import MG_RBY1_CloseSingleDoor
        from mimicgen.env_interfaces.robocasa.single_stage.mg_rby1_close_drawer import MG_RBY1_CloseDrawerLeft
        self.assertTrue(issubclass(MG_RBY1_CloseDrawerLeft, MG_RBY1_CloseSingleDoor))


class DoorDrawerEnvironmentTests(unittest.TestCase):
    def test_left_arm_controller_action_on_door_and_drawer(self):
        from mobipi.collect_demos.auto_close_microwave_door import (
            build_controller_interface, create_robot_action, pose_target_for_ref,
        )
        controller = load_composite_controller_config(controller=str(CONTROLLER), robot="RBY1")
        for task in ("CloseSingleDoor", "CloseDrawerLeft"):
            with self.subTest(task=task):
                env = robosuite.make(
                    task, robots="RBY1", controller_configs=controller,
                    layout_ids=[1], style_ids=[10], initialization_noise=None,
                    randomize_base_init_pose=None, randomize_cameras=False, seed=0,
                    has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
                )
                try:
                    env.reset()
                    robot = env.robots[0]
                    arm, tcp, ref, pos, rot, tcp_to_ref, holds = build_controller_interface(env, robot, controller, "left")
                    self.assertEqual(arm, "left")
                    self.assertIn("left", tcp)
                    self.assertIn("left", ref)
                    target = pose_target_for_ref(pos, rot, tcp_to_ref)
                    action = create_robot_action(robot, arm, target, holds)
                    self.assertEqual(action.shape, (env.action_dim,))
                    self.assertTrue(np.isfinite(action).all())
                    env.step(action)
                    self.assertTrue(np.isfinite(env.sim.data.qpos).all())
                    self.assertIsInstance(bool(env._check_success()), bool)
                finally:
                    env.close()


if __name__ == "__main__":
    unittest.main()
