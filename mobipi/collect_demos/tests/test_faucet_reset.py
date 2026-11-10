"""Regressions for recursive model loading and failed initialization cleanup."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np
from robocasa.environments.kitchen.single_stage.kitchen_sink import TurnOnSinkFaucet

from mobipi.collect_demos.faucet_runtime import TurnOnSinkFaucetRBY1
from mobipi.collect_demos.collect_faucet_demos import collect


class CameraRegistrationTests(unittest.TestCase):
    def test_recursive_placement_retry_registers_one_camera(self):
        env = object.__new__(TurnOnSinkFaucetRBY1)
        env.faucet_fixture_offset = np.zeros(4)
        env._faucet_offset_model = None
        env.faucet_initial_arm_qpos = None
        env.faucet_initial_torso_qpos = None
        env.model = SimpleNamespace(worldbody=ET.Element("worldbody"))
        env.sink = SimpleNamespace(pos=np.array([1., 2., 3.]))
        calls = 0

        def load(instance):
            nonlocal calls
            calls += 1
            if calls == 1:
                # Kitchen retries placement via self._load_model(), then returns
                # the final model through both nested faucet override calls.
                instance._load_model()

        with patch.object(TurnOnSinkFaucet, "_load_model", load):
            env._load_model()
        cameras = env.model.worldbody.findall("camera[@name='faucet_overview']")
        self.assertEqual(calls, 2)
        self.assertEqual(len(cameras), 1)
        np.testing.assert_allclose(np.fromstring(cameras[0].get("pos"), sep=" "), [-.2, 3.4, 4.15])
        self.assertEqual(cameras[0].get("fovy"), "55")

    def test_fixture_offset_is_applied_once_during_recursive_load(self):
        env = object.__new__(TurnOnSinkFaucetRBY1)
        env.faucet_initial_arm_qpos = None
        env.faucet_initial_torso_qpos = None
        env.faucet_fixture_offset = np.array([.02, -.02, 0., .03])
        env._faucet_offset_model = None
        env.model = SimpleNamespace(worldbody=ET.Element("worldbody"))
        sink = SimpleNamespace(pos=np.array([1., 2., 3.]), euler=np.zeros(3))
        sink.set_pos = lambda value: setattr(sink, "pos", value)
        sink.set_euler = lambda value: setattr(sink, "euler", value)
        env.sink = sink
        calls = 0

        def load(instance):
            nonlocal calls
            calls += 1
            if calls == 1:
                instance._load_model()

        with patch.object(TurnOnSinkFaucet, "_load_model", load):
            env._load_model()
        np.testing.assert_allclose(sink.pos, [1.02, 1.98, 3.])
        np.testing.assert_allclose(sink.euler, [0., 0., .03])
        self.assertEqual(len(env.model.worldbody.findall("camera[@name='faucet_overview']")), 1)
        camera = env.model.worldbody.find("camera[@name='faucet_overview']")
        np.testing.assert_allclose(np.fromstring(camera.get("pos"), sep=" "), [-.2, 3.4, 4.15])

    def test_constructor_failure_closes_partial_environment(self):
        original = ValueError("original compilation failure")
        with patch.object(TurnOnSinkFaucet, "__init__", side_effect=original), \
             patch.object(TurnOnSinkFaucetRBY1, "close") as close:
            with self.assertRaises(ValueError) as caught:
                TurnOnSinkFaucetRBY1()
        self.assertIs(caught.exception, original)
        close.assert_called_once()


class CollectionFailureTests(unittest.TestCase):
    def test_failed_reset_keeps_original_exception_and_closes_without_sim(self):
        original = ValueError("Error: repeated name 'faucet_overview' in camera")
        sink = Mock()
        base = SimpleNamespace(sim=None, sink=sink)
        env = SimpleNamespace(base_env=base, env=Mock(), reset=Mock(side_effect=original))
        # Even a cleanup error must not replace the original reset exception.
        env.env.close.side_effect = RuntimeError("cleanup failure")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"attempt"
            with patch("mobipi.collect_demos.collect_faucet_demos.make_env", return_value=env):
                with self.assertRaises(ValueError) as caught:
                    collect({"seed": 7}, output, 1, 1)
            report = json.loads((output/"summary.json").read_text())
        self.assertIs(caught.exception, original)
        sink.get_handle_state.assert_not_called()
        env.env.close.assert_called_once()
        self.assertEqual(report["failures"][0]["reason"], str(original))
        self.assertEqual(report["failures"][0]["stage"], "reset")
        self.assertEqual(report["failures"][0]["seed"], 7)
        self.assertIn("ValueError", report["failures"][0]["traceback"])
        self.assertNotIn("last_task_state", report)
        self.assertEqual(report["cleanup_errors"][0]["operation"], "close environment")

    def test_failed_construction_saves_original_exception(self):
        original = ValueError("original construction failure")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/"attempt"
            with patch("mobipi.collect_demos.collect_faucet_demos.make_env", side_effect=original):
                with self.assertRaises(ValueError) as caught:
                    collect({"seed": 0}, output, 1, 1)
            report = json.loads((output/"summary.json").read_text())
        self.assertIs(caught.exception, original)
        self.assertEqual(report["failures"][0]["stage"], "initialize")
        self.assertEqual(report["failures"][0]["reason"], str(original))


if __name__ == "__main__":
    unittest.main()
