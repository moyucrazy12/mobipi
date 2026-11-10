"""Offline grid planning, resume and restoration guards; no simulator construction."""
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from mobipi.collect_demos import evaluate_faucet_dp as evaluation


PROJECT = Path(__file__).resolve().parents[4]
DATASET = PROJECT / "experiment_results/TurnOnFaucet/demos/faucet_relative_cartesian_100/train.hdf5"
CHECKPOINT = PROJECT / "experiment_results/TurnOnFaucet/train/rby1_faucet_relative_cartesian_4000steps/20261028145350/models/model_epoch_14.pth"


class FaucetGridTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        evaluation.O.initialize_obs_utils_with_obs_specs(
            {"obs": {"low_dim": evaluation.LOWDIM, "rgb": evaluation.RGB}})

    @staticmethod
    def restored_fixture(group):
        """A saved-data-backed fake; no MuJoCo imports, compilation or stepping."""
        state = group["states"][0]
        xml = group.attrs["model_file"]
        joints, offset = {}, 1
        for joint in ET.fromstring(xml).find("worldbody").iter("joint"):
            width = {"free": 7, "ball": 4}.get(joint.get("type", "hinge"), 1)
            joints[joint.get("name")] = state[offset:offset+width].copy()
            offset += width
        meta = json.loads(group.attrs["ep_meta"])
        objects = {o["name"]: SimpleNamespace(root_body=o["name"] + "_main") for o in meta["object_cfgs"]}
        env = SimpleNamespace(state=state.copy(), xml=xml, joints=joints)
        env.get_state = lambda: {"model": env.xml}
        env.base_env = SimpleNamespace(
            sim=SimpleNamespace(get_state=lambda: SimpleNamespace(flatten=lambda: env.state),
                                data=SimpleNamespace(get_joint_qpos=lambda name: env.joints[name])), objects=objects)
        poses = {}
        for name, obj in objects.items():
            q = joints[name + "_joint0"]
            poses[obj.root_body] = evaluation.tcp_pose(q[:3], np.roll(q[3:], -1))
        interface = SimpleNamespace(tcp=group["diagnostics/actual_tcp"][0],
                                    fixtures={n: group[f"datagen_info/object_poses/{n}"][0]
                                              for n in ("faucet", "handle")}, object_poses=poses)
        interface.get_robot_eef_pose = lambda: interface.tcp
        interface.get_object_poses = lambda: interface.fixtures
        interface.get_object_pose = lambda body, kind: interface.object_poses[body]
        initial = {key: group[f"obs/{key}"][0] for key in evaluation.LOWDIM}
        initial.update({key: evaluation.O.process_obs(group[f"obs/{key}"][0], obs_key=key)
                        for key in evaluation.RGB})
        base = evaluation.tcp_pose(group["obs/robot0_base_pos"][0], group["obs/robot0_base_quat"][0])
        return env, interface, initial, base

    def test_actual_held_out_starts_and_close_single_door_seeds(self):
        configurations = evaluation.validation_configurations(DATASET, evaluation.VALIDATION_DEMOS)
        close = json.loads((PROJECT / "experiment_results/CloseSingleDoor/rollouts/rby1_relative_final_125_eval/metadata.json").read_text())
        self.assertEqual(evaluation.GRID_SEEDS, close["seeds"])
        self.assertEqual(len(configurations) * len(evaluation.GRID_SEEDS), 125)
        self.assertEqual(len({c["state_sha256"] for c in configurations}), 5)
        self.assertEqual(len({tuple(c["fixture_position"]) for c in configurations}), 5)
        self.assertEqual([c["generation_seed"] for c in configurations], [1005, 1006, 1012, 2007, 2018])
        self.assertTrue(CHECKPOINT.is_file())
        with self.assertRaises(ValueError):
            evaluation.validation_configurations(DATASET, ["demo_0"])
        with self.assertRaises(ValueError):
            evaluation.validation_configurations(DATASET, ["demo_5", "demo_5"])
        with h5py.File(DATASET, "r") as f:
            for name in evaluation.VALIDATION_DEMOS:
                group = f["data/" + name]
                joints = ET.fromstring(group.attrs["model_file"]).find("worldbody").iter("joint")
                sizes = [{"free": (7, 6), "ball": (4, 3)}.get(j.get("type", "hinge"), (1, 1))
                         for j in joints]
                self.assertEqual(1 + sum(q + v for q, v in sizes), len(group["states"][0]))

    def test_resume_skips_completed_and_archives_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "grid"
            args = SimpleNamespace(dataset=DATASET, output=output, check_only=False,
                                   fixture_offset=[0., 0., 0.], demo_keys=["demo_5"],
                                   seeds=[0, 1], env_seed=0, max_steps=2, device="cpu",
                                   task_config=Path(evaluation.__file__).with_name("faucet_config.json"),
                                   plan_only=False, resume=False)
            calls = []

            def rollout(policy, checkpoint, args, seed, directory, configuration):
                calls.append(seed)
                directory.mkdir(parents=True, exist_ok=False)
                return dict(initialization_verified=True, configuration_id=configuration["id"],
                            diffusion_seed=seed, full_success=seed == 0,
                            activation=True, physical_contact=True)

            with patch.object(evaluation, "load_policy", return_value=(None, {})), \
                    patch.object(evaluation, "rollout", side_effect=rollout):
                evaluation.validation_grid(args, [CHECKPOINT])
                self.assertEqual(calls, [0, 1])
                completed = output / "config_01_demo_5/seed_0/result.json"
                original = completed.read_bytes()
                args.resume = True
                evaluation.validation_grid(args, [CHECKPOINT])
                self.assertEqual(calls, [0, 1])
                self.assertEqual(completed.read_bytes(), original)
                partial = output / "config_01_demo_5/seed_1"
                (partial / "result.json").write_text('{"completed": false}')
                (partial / "control.jsonl").write_text("preserve interrupted log")
                evaluation.validation_grid(args, [CHECKPOINT])
                self.assertEqual(calls, [0, 1, 1])
                archives = list(partial.parent.glob("seed_1_interrupted_*"))
                self.assertEqual(len(archives), 1)
                self.assertEqual((archives[0] / "control.jsonl").read_text(), "preserve interrupted log")
                args.max_steps = 3
                with self.assertRaises(ValueError):
                    evaluation.validation_grid(args, [CHECKPOINT])

    def test_state_restoration_guard_rejects_changed_state(self):
        with h5py.File(DATASET, "r") as f:
            group = f["data/demo_5"]
            env = SimpleNamespace(base_env=SimpleNamespace(sim=SimpleNamespace(
                get_state=lambda: SimpleNamespace(flatten=lambda: group["states"][0] + 1))))
            with self.assertRaises(AssertionError):
                evaluation.verify_physical_start(env, None, {}, group, None)

    def test_restore_preserves_compiled_model_and_never_settles_after_state(self):
        with h5py.File(DATASET, "r") as f:
            group = f["data/demo_5"]
            events = []

            def set_state(state):
                np.testing.assert_array_equal(state, group["states"][0])
                events.append("set_state")

            # A model object is deliberately unnecessary: this must not compile,
            # replace, or step a simulator when applying the saved state.
            sim = SimpleNamespace(reset=lambda: events.append("reset"),
                                  set_state_from_flattened=set_state,
                                  forward=lambda: events.append("forward"))
            env = SimpleNamespace(get_state=lambda: {"model": group.attrs["model_file"]},
                                  base_env=SimpleNamespace(sim=sim, update_state=lambda: events.append("update_state")))
            evaluation.restore_validation_state(env, group)
            self.assertEqual(events, ["reset", "set_state", "forward", "update_state"])

    def test_restore_rejects_different_model_before_touching_state(self):
        with h5py.File(DATASET, "r") as f:
            group = f["data/demo_5"]
            # Simulate a different camera or rounded model being reconstructed.
            changed = group.attrs["model_file"].replace('fovy="55"', 'fovy="54"')
            self.assertNotEqual(changed, group.attrs["model_file"])
            env = SimpleNamespace(get_state=lambda: {"model": changed})
            with self.assertRaisesRegex(ValueError, "Rebuilt demonstration model differs"):
                evaluation.restore_validation_state(env, group)

    def test_image_diagnostics_are_calculated_saved_and_do_not_change_inputs(self):
        with h5py.File(DATASET, "r") as f, tempfile.TemporaryDirectory() as temporary:
            group = f["data/demo_5"]
            env, interface, initial, base = self.restored_fixture(group)
            initial[evaluation.RGB[0]][0, 0, 0] -= 2./255
            initial[evaluation.RGB[0]][0, 0, 1] -= 1./255
            original = {key: value.copy() for key, value in initial.items()}
            with patch("builtins.print") as warning:
                result = evaluation.verify_restored_start(env, interface, initial, group, base, Path(temporary))
            warning.assert_called_once()
            self.assertIn("physical/geometry checks passed", warning.call_args.args[0])
            self.assertEqual(json.loads((Path(temporary) / "image_diagnostics.json").read_text()), result)
            self.assertTrue(result["physical_and_geometric_restoration_verified"])
            self.assertEqual(result["historical_collection_renderer_configuration"], "unknown")
            metrics = result["cameras"][evaluation.RGB[0]]
            self.assertAlmostEqual(metrics["mae"], 3. / (49152 * 255))
            self.assertAlmostEqual(metrics["maximum_absolute_error"], 2. / 255)
            self.assertEqual(metrics["fraction_exceeding_1_over_255"], 1. / 49152)
            self.assertEqual(metrics["channels_exceeding_1_over_255"], 1)
            self.assertFalse(metrics["original_comparison_passed"])
            self.assertEqual(result["cameras"][evaluation.RGB[1]]["mae"], 0.)
            for key in initial:
                np.testing.assert_array_equal(initial[key], original[key])

    def test_physical_failures_are_fatal_before_rendering_is_classified(self):
        with h5py.File(DATASET, "r") as f, tempfile.TemporaryDirectory() as temporary:
            group = f["data/demo_5"]
            for failure in ("state", "xml", "camera", "joint", "base", "lowdim", "tcp", "faucet", "handle", "object"):
                with self.subTest(failure=failure):
                    env, interface, initial, base = self.restored_fixture(group)
                    if failure == "state":
                        env.state[0] += .01
                    elif failure == "xml":
                        env.xml += "\n"  # Even a byte-only difference is fatal.
                    elif failure == "camera":
                        env.xml = env.xml.replace('fovy="55"', 'fovy="54"')
                    elif failure == "joint":
                        env.joints["robot0_right_arm_0"][0] += .01
                    elif failure == "base":
                        base[0, 3] += .01
                    elif failure == "lowdim":
                        initial[evaluation.LOWDIM[0]][0] += .01
                    elif failure == "tcp":
                        interface.tcp[0, 3] += .01
                    elif failure in ("faucet", "handle"):
                        interface.fixtures[failure][0, 3] += .01
                    else:
                        interface.object_poses["distr_sink_main"][0, 3] += .01
                    with patch.object(evaluation, "image_diagnostics") as diagnostics, patch("builtins.print") as warning:
                        with self.assertRaises((AssertionError, ValueError)):
                            evaluation.verify_restored_start(env, interface, initial, group, base, Path(temporary))
                    diagnostics.assert_not_called()
                    warning.assert_not_called()
                    self.assertFalse((Path(temporary) / "image_diagnostics.json").exists())

    def test_plan_does_not_create_output_or_load_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = SimpleNamespace(dataset=DATASET, output=Path(temporary) / "uncreated",
                                   check_only=False, fixture_offset=[0., 0., 0.], demo_keys=None,
                                   seeds=evaluation.GRID_SEEDS, env_seed=0, max_steps=2500,
                                   device="cuda", task_config=Path(evaluation.__file__).with_name("faucet_config.json"),
                                   plan_only=True)
            with patch.object(evaluation, "load_policy", side_effect=AssertionError("Policy must not load")), \
                    patch("builtins.print"):
                evaluation.validation_grid(args, [CHECKPOINT])
            self.assertFalse(args.output.exists())


if __name__ == "__main__":
    unittest.main()
