"""Small ongoing guards for saved commands and bounded control termination."""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from rby1.demos.faucet_dataset import validate, write_episode
from rby1.demos.faucet_progress import ProgressGate
from rby1.pipeline.collect import run_lock


class RecordingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "recording.hdf5"
        target = np.eye(4)
        target[0, 3] = 2.75
        rows = [dict(actions=np.arange(24, dtype=float), states=np.zeros(40),
                     rewards=0., dones=False,
                     obs={"robot0_right_joint_pos": np.zeros(7),
                          "camera_image": np.zeros((4, 4, 3), dtype=np.uint8)},
                     diagnostics={"commanded_tcp": target.copy(), "actual_tcp": np.eye(4),
                                  "water_on": bool(t), "phase": "test"}) for t in range(3)]
        env = SimpleNamespace(serialize=lambda: {"env_name": "TurnOnSinkFaucetRBY1"})
        with h5py.File(self.path, "x") as f:
            write_episode(f.create_group("data/demo_0"), env, {"model": "test"}, rows, "test")

    def test_recorded_commands_are_unclipped_and_distinct_from_observations(self):
        with h5py.File(self.path, "r") as f:
            ep = f["data/demo_0"]
            np.testing.assert_array_equal(ep["actions"][0], np.arange(24, dtype=float))
            np.testing.assert_array_equal(ep["dones"], [False, False, True])
            self.assertEqual(ep["obs/camera_image"].dtype, np.uint8)
            self.assertEqual(ep["diagnostics/commanded_tcp"][0, 0, 3], 2.75)
            self.assertEqual(ep["diagnostics/actual_tcp"][0, 0, 3], 0.)
        self.assertEqual(validate(self.path)[0]["samples"], 3)

    def test_corrupt_commands_or_alignment_are_rejected(self):
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/actions"][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN / Inf"):
            validate(self.path)
        with h5py.File(self.path, "a") as f:
            f["data/demo_0/actions"][0, 0] = 0.
            del f["data/demo_0/states"]
            f.create_dataset("data/demo_0/states", data=np.zeros((2, 40)))
        with self.assertRaisesRegex(ValueError, "length != 3"):
            validate(self.path)

    def test_parallel_pipeline_writer_is_rejected_and_lock_released(self):
        root = self.path.parent
        with run_lock(root):
            with self.assertRaisesRegex(RuntimeError, "Another pipeline process"):
                with run_lock(root):
                    self.fail("Concurrent writers acquired the same output")
        with run_lock(root):
            pass


class ProgressTests(unittest.TestCase):
    def gate(self):
        # Synthetic control bounds, independent of private experiment settings.
        return ProgressGate(dict(position_tolerance=.01, orientation_tolerance=.02,
                                 min_position_progress=.001, min_orientation_progress=.01,
                                 max_seconds=1., max_steps=20, stall_window_steps=2,
                                 transient_position_error=.1), control_dt=.1)

    def test_convergence_requires_both_errors_and_finite_values(self):
        gate = self.gate()
        self.assertNotEqual(gate.update(0., .1), "converged")
        self.assertEqual(gate.update(.001, .001), "converged")
        self.assertEqual(self.gate().update(np.nan, 0.), "ik_failure")

    def test_unmoving_waypoint_stops(self):
        gate = self.gate()
        for _ in range(3):
            result = gate.update(.2, .1)
        self.assertEqual(result, "persistent_tracking_error")

    def test_improving_but_unreached_waypoint_still_times_out(self):
        gate = self.gate()
        for step in range(10):
            result = gate.update(1. - step * .01, .1)
        self.assertEqual(result, "waypoint_timeout")


if __name__ == "__main__":
    unittest.main()
