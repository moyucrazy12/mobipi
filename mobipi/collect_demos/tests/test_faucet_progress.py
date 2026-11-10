"""Regression tests for held-target convergence and independent failure conditions."""
import json
from pathlib import Path
import unittest

from mobipi.collect_demos.faucet_progress import ProgressGate


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.settings = json.loads(Path(__file__).parents[1].joinpath('faucet_config.json').read_text())["script"]["waypoint"]

    def test_large_error_keeps_running_while_converging(self):
        gate = ProgressGate(self.settings, .05)
        for i in range(150):
            condition = gate.update(.10-i*.0002, .1-i*.0001)
            self.assertNotIn(condition, ("persistent_tracking_error", "stalled_convergence", "waypoint_timeout"))
        self.assertEqual(condition, "converging")

    def test_persistent_large_error_stalls_after_full_window(self):
        gate = ProgressGate(self.settings, .05)
        for _ in range(self.settings["stall_window_steps"]):
            self.assertNotEqual(gate.update(.075, .1), "persistent_tracking_error")
        self.assertEqual(gate.update(.075, .1), "persistent_tracking_error")

    def test_small_unresolved_error_stalls(self):
        gate = ProgressGate(self.settings, .05)
        for _ in range(self.settings["stall_window_steps"]+1):
            condition = gate.update(.01, .04)
        self.assertEqual(condition, "stalled_convergence")

    def test_position_alone_does_not_advance_waypoint(self):
        gate = ProgressGate(self.settings, .05)
        self.assertNotEqual(gate.update(.001, .2), "converged")
        self.assertEqual(gate.update(.001, .02), "converged")

    def test_finite_timeout_even_with_progress(self):
        settings = dict(self.settings, max_steps=5)
        gate = ProgressGate(settings, .05)
        for i in range(5):
            condition = gate.update(.1-i*.001, .1)
        self.assertEqual(condition, "waypoint_timeout")

    def test_nonfinite_is_ik_failure(self):
        self.assertEqual(ProgressGate(self.settings, .05).update(float('nan'), .1), "ik_failure")

    def test_new_target_has_independent_history(self):
        first = ProgressGate(self.settings, .05)
        for _ in range(30): first.update(.08, .1)
        second = ProgressGate(self.settings, .05)
        self.assertEqual(second.update(.08, .1), "transient_tracking_error")
        self.assertEqual(second.steps, 1)

    def test_waypoint_holds_the_same_target_until_converged(self):
        from types import SimpleNamespace
        import numpy as np
        from mobipi.collect_demos.collect_faucet_demos import ScriptedRecorder
        recorder = ScriptedRecorder.__new__(ScriptedRecorder)
        recorder.settings = {"waypoint": self.settings}
        recorder.interface = SimpleNamespace(env=SimpleNamespace(control_timestep=.05))
        recorder.waypoint_index = -1
        recorder.last_condition = "running"
        commands = []
        errors = iter([.025, .020, .015, .010, .004])
        def step(pose, gripper, phase):
            commands.append(pose.copy())
            recorder.last_condition = recorder.gate.update(next(errors), .01)
        recorder.step = step
        target = np.eye(4)
        target[0, 3] = 2.75
        recorder.waypoint(target, -1., "safe_preapproach")
        self.assertEqual(len(commands), 5)
        self.assertTrue(all(np.array_equal(command, target) for command in commands))
        self.assertIsNone(recorder.gate)

    def test_shared_recorder_keeps_legacy_tracking_behavior(self):
        from mobipi.collect_demos.collect_faucet_demos import Recorder
        from mobipi.collect_demos.faucet_runtime import UnsafeTrajectory
        recorder = Recorder.__new__(Recorder)
        recorder.settings = {"max_tracking_error": .06, "tracking_patience": 12}
        recorder.bad_tracking = 0
        for _ in range(11): recorder.check_tracking(.075, .1)
        with self.assertRaises(UnsafeTrajectory): recorder.check_tracking(.075, .1)


if __name__ == '__main__':
    unittest.main()
