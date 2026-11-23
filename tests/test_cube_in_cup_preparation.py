"""Protect recorded right-arm labels, RGB alignment and reversible organization."""
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from rby1_mobipi.policies.prepare_cube_in_cup import CAMERAS, LOW_DIM, convert, organize


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw"
        self.source.mkdir()
        self.output = self.root / "train.hdf5"
        self.actions = np.arange(60, dtype=np.float32).reshape(3, 20)
        self.names = (["right_wheel", "left_wheel"] + [f"torso_{j}" for j in range(6)]
                      + [f"right_arm_{j}" for j in range(7)]
                      + [f"left_arm_{j}" for j in range(7)] + ["head_0", "head_1"])
        self.first = self.fixture("session_002/sequence_010")
        self.fixture("session_003/sequence_001")
        self.fixture("session_002/sequence_009")
        (self.source / "session_002/sequence_019").mkdir()

    def fixture(self, key):
        directory = self.source / key
        directory.mkdir(parents=True)
        rows = []
        for t in range(3):
            obs = dict(joint_pos=np.arange(24, dtype=np.float32),
                       joint_vel=np.zeros(24, dtype=np.float32),
                       right_ee_pos=np.zeros(3, dtype=np.float32),
                       right_ee_quat=np.array([1, 0, 0, 0], dtype=np.float32),
                       grippers_width=(.01, .07))
            for camera in CAMERAS.values():
                # Descending file numbers deliberately differ from row order.
                path = directory / camera / f"frame_{2-t}.npz"
                path.parent.mkdir(exist_ok=True)
                image = np.full((4, 5, 3), [10+t, 20+t, 30+t], dtype=np.uint8)
                np.savez_compressed(path, frame=image)
                obs[camera] = str(path.relative_to(directory))
            rows.append(dict(obs=obs, action=self.actions[t], timestamp=t * .1))
        path = directory / (directory.name + ".npz")
        np.savez_compressed(path, trajectory_metadata=np.array(dict(num_actions=3, fps=10.,
                            joint_names=self.names), dtype=object), arr_0=np.array(rows, dtype=object))
        return path

    def organize(self):
        return organize(self.source, ["session_002/sequence_009"], skip_incomplete=True)

    def test_organization_and_exact_right_arm_projection(self):
        original = self.first.read_bytes()
        demos = self.organize()
        self.assertEqual([p.name for p in demos], ["demo_1", "demo_2"])
        self.assertEqual((demos[0] / "sequence_010.npz").read_bytes(), original)
        self.assertTrue((self.source / "excluded/session_002/sequence_009").is_dir())
        self.assertTrue((self.source / "excluded/session_002/sequence_019").is_dir())
        report = convert(demos, self.output, "bgr", self.source / "recordings_manifest.csv")
        self.assertEqual(report["split"], {"train": 1, "valid": 1})
        with h5py.File(self.output, "r") as f:
            ep = f["data/demo_0"]
            np.testing.assert_array_equal(ep["actions"], self.actions[:, 7:14])
            np.testing.assert_array_equal(ep["obs/right_joint_pos"][0], np.arange(8, 15))
            np.testing.assert_array_equal(ep["obs/head_rgb"][0, 0, 0], [30, 20, 10])
            self.assertEqual(set(ep["obs"]), set(CAMERAS) | set(LOW_DIM))
            self.assertEqual(ep.attrs["original_sequence"], "session_002/sequence_010")
            np.testing.assert_array_equal(ep["dones"], [0, 0, 1])
            self.assertFalse(set(f["mask/train"][:]) & set(f["mask/valid"][:]))
        self.assertEqual(organize(self.source), demos)
        with self.assertRaises(FileExistsError):
            convert(demos, self.output, "bgr")

    def test_incomplete_source_fails_before_moving_anything(self):
        with self.assertRaises(ValueError):
            organize(self.source)
        self.assertTrue(self.first.is_file())
        self.assertFalse((self.source / "recordings").exists())

    def test_sessions_can_be_inside_a_directory_already_named_recordings(self):
        target = self.root / "recordings"
        self.source.rename(target)
        self.source = target
        demos = self.organize()
        self.assertEqual(demos[0], target / "demo_1")
        self.assertEqual(organize(target), demos)

    def test_corruption_does_not_publish_a_training_dataset(self):
        demos = self.organize()
        path = demos[0] / "head_rgb/frame_1.npz"
        np.savez_compressed(path, frame=np.zeros((4, 5), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "uint8 HWC"):
            convert(demos, self.output, "rgb")
        self.assertFalse(self.output.exists())
        self.assertTrue(demos[0].is_dir())


if __name__ == "__main__":
    unittest.main()
