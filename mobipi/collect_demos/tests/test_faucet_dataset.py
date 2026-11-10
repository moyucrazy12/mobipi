"""Synthetic structural fixtures: these tests do not demonstrate task success."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from mobipi.collect_demos.faucet_dataset import validate


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name)/"synthetic.hdf5"
        n = 8
        with h5py.File(self.path, "x") as f:
            data = f.create_group("data")
            data.attrs.update(total=n, env_args=json.dumps({"env_name": "TurnOnSinkFaucetRBY1"}))
            ep = data.create_group("demo_0")
            ep.attrs.update(num_samples=n, success=True, model_file="synthetic", ep_meta="{}", origin="test")
            for key, shape in (("actions", (n, 24)), ("states", (n, 40)), ("rewards", (n,)), ("dones", (n,)),
                               ("obs/robot0_right_joint_pos", (n, 7)),
                               ("obs/faucet_overview_image", (n, 8, 8, 3))):
                ep.create_dataset(key, data=np.zeros(shape))
            poses = np.repeat(np.eye(4)[None], n, axis=0)
            ep.create_dataset("diagnostics/commanded_tcp", data=poses)
            ep.create_dataset("diagnostics/water_on", data=[0, 0, 0, 0, 0, 1, 1, 1])
            for key in ("eef_pose", "target_pose", "object_poses/faucet", "object_poses/handle"):
                ep.create_dataset("datagen_info/"+key, data=poses)
            ep.create_dataset("datagen_info/gripper_action", data=np.ones((n, 1)))
            ep.create_dataset("datagen_info/subtask_term_signals/contact", data=[0, 0, 1, 1, 1, 1, 0, 0])
            ep.create_dataset("datagen_info/subtask_term_signals/water_on", data=[0, 0, 0, 0, 0, 1, 1, 1])

    def tearDown(self):
        self.temporary.cleanup()

    def test_writer_preserves_commands_and_training_keys(self):
        from types import SimpleNamespace
        from mobipi.collect_demos.faucet_dataset import write_episode
        target = np.eye(4)
        target[0, 3] = 2.75
        rows = [dict(actions=np.arange(24, dtype=float), states=np.zeros(40),
                     rewards=0., dones=False,
                     obs={"robot0_right_joint_pos": np.zeros(7), "camera_image": np.zeros((4, 4, 3), dtype=np.uint8)},
                     diagnostics={"commanded_tcp": target.copy(), "actual_tcp": np.eye(4),
                                  "water_on": bool(t), "phase": "synthetic"}) for t in range(3)]
        env = SimpleNamespace(serialize=lambda: {"env_name": "TurnOnSinkFaucetRBY1"})
        path = Path(self.temporary.name)/"writer_fixture.hdf5"
        with h5py.File(path, "x") as f:
            ep = f.create_group("data/demo_0")
            write_episode(ep, env, dict(model="synthetic", ep_meta="{}"), rows, "test")
            np.testing.assert_array_equal(ep["actions"][0], np.arange(24, dtype=float))
            np.testing.assert_array_equal(ep["dones"], [False, False, True])
            self.assertEqual(ep["obs/camera_image"].dtype, np.uint8)
            self.assertEqual(ep["diagnostics/commanded_tcp"][0, 0, 3], 2.75)
            self.assertEqual(ep["diagnostics/actual_tcp"][0, 0, 3], 0.)
        validate(path)

    def test_native_combined_robot_proprioception_is_valid(self):
        with h5py.File(self.path, "a") as f:
            obs = f["data/demo_0/obs"]
            del obs["robot0_right_joint_pos"]
            obs.create_dataset("robot0_joint_pos_cos", data=np.ones((8, 28)))
            obs.create_dataset("robot0_joint_pos_sin", data=np.zeros((8, 28)))
            obs.create_dataset("robot0_joint_vel", data=np.zeros((8, 28)))
            obs.create_dataset("robot0_right_eef_pos", data=np.zeros((8, 3)))
        validate(self.path)

    def test_valid_ordered_segmentation(self):
        self.assertEqual(validate(self.path, prepared=True)[0]["boundaries"], [2, 5])

    def test_reject_nan_actions(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/actions"][3, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            validate(self.path)

    def test_reject_missing_contact(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/datagen_info/subtask_term_signals/contact"][:] = 0
        with self.assertRaisesRegex(ValueError, "no rising contact"):
            validate(self.path, prepared=True)

    def test_reject_articulating_fixture_frame(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/datagen_info/object_poses/faucet"][3, 0, 3] = .1
        with self.assertRaisesRegex(ValueError, "articulated frame"):
            validate(self.path, prepared=True)

    def test_reject_relabelled_achieved_commands(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/datagen_info/target_pose"][3, 0, 3] = .1
        with self.assertRaisesRegex(ValueError, "executed commands"):
            validate(self.path, prepared=True)

    def test_reject_unordered_boundaries(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/datagen_info/subtask_term_signals/contact"][:] = [0, 0, 0, 0, 0, 0, 1, 1]
        with self.assertRaisesRegex(ValueError, "ordered subtask"):
            validate(self.path, prepared=True)

    def test_reject_length_mismatch(self):
        with h5py.File(self.path, "a") as f:
            del f["data/demo_0/states"]
            f["data/demo_0"].create_dataset("states", data=np.zeros((7, 40)))
        with self.assertRaisesRegex(ValueError, "length"):
            validate(self.path)


if __name__ == "__main__":
    unittest.main()
