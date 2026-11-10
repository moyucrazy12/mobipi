"""CPU-only recovery tests with a synthetic episode; no task-success claim."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from mobipi.collect_demos.faucet_dataset import validate
from mobipi.collect_demos.pipeline import Pipeline, main, merge, run_lock
from mobipi.collect_demos.pipeline_tasks import faucet_attempt_config


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.fixture = Path(cls.temporary.name) / "episode.hdf5"
        with h5py.File(cls.fixture, "x") as dest:
            data = dest.create_group("data")
            n = 8
            data.attrs.update(total=n, env_args=json.dumps({"env_name": "TurnOnSinkFaucetRBY1"}),
                              source_demo_keys=json.dumps(["seed_0", "seed_1"]))
            ep = data.create_group("demo_0")
            ep.attrs.update(num_samples=n, success=True, model_file="synthetic", ep_meta="{}", origin="test")
            for name, shape in (("actions", (n, 24)), ("states", (n, 40)),
                                ("rewards", (n,)), ("dones", (n,)),
                                ("obs/robot0_right_joint_pos", (n, 7)),
                                ("obs/faucet_overview_image", (n, 8, 8, 3))):
                ep.create_dataset(name, data=np.zeros(shape))
            poses = np.repeat(np.eye(4)[None], n, axis=0)
            for name in ("diagnostics/commanded_tcp", "datagen_info/eef_pose", "datagen_info/target_pose",
                         "datagen_info/object_poses/faucet", "datagen_info/object_poses/handle"):
                ep.create_dataset(name, data=poses)
            ep.create_dataset("diagnostics/water_on", data=[0, 0, 0, 0, 0, 1, 1, 1])
            ep.create_dataset("diagnostics/phase", data=np.asarray(["synthetic"] * n, dtype="S"))
            ep.create_dataset("datagen_info/gripper_action", data=np.ones((n, 1)))
            ep.create_dataset("datagen_info/subtask_term_signals/contact", data=[0, 0, 1, 1, 1, 1, 0, 0])
            ep.create_dataset("datagen_info/subtask_term_signals/water_on", data=[0, 0, 0, 0, 0, 1, 1, 1])
            ep.create_dataset("src_demo_inds", data=np.ones(n, dtype=int))
        validate(cls.fixture, prepared=True)

    def test_resume_recovers_flushed_episode_and_partial_preparation(self):
        fixture = self.fixture
        class Adapter:
            active_arm = "left"  # The runner must not assume the faucet arm.
            convert = "fake"
            collected = generated = prepared = converted = 0

            def call(self, operation, *args, **kwargs):
                if operation == "validate":
                    return validate(*args, **kwargs)
                if operation == "attempt_config":
                    return args[0], {"seed": 42 + args[2]}
                if operation == "collect":
                    self.collected += 1
                    output = args[1]; output.mkdir()
                    shutil.copyfile(fixture, output / "seeds.hdf5")
                    if self.collected == 1:
                        raise KeyboardInterrupt()
                elif operation == "prepare":
                    self.prepared += 1
                    shutil.copyfile(args[0], args[1])
                    if self.prepared == 1:
                        raise KeyboardInterrupt()
                elif operation == "augment":
                    self.generated += 1
                    output = args[1]; output.mkdir()
                    if self.generated == 1:
                        with h5py.File(output / "generated.hdf5", "x") as f:
                            f.create_group("data")  # Rejected attempt stays preserved.
                        return False
                    shutil.copyfile(fixture, output / "generated.hdf5")
                elif operation == "convert":
                    self.converted += 1
                    shutil.copyfile(args[0], args[1])
                    return {"dataset": str(args[1])}
                return True
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); task = Adapter()
            settings = dict(task="synthetic", seeds=2, augment=2, seed_attempts=4, augment_attempts=4,
                            translation=0., rotation_deg=0., seed_config={}, augmentation_config={})
            run = Pipeline(root, task, settings)
            with self.assertRaises(KeyboardInterrupt):
                run.run("all", convert=True)
            with self.assertRaises(KeyboardInterrupt):
                Pipeline(root, task, settings, resume=True).run("all", convert=True)
            run = Pipeline(root, task, settings, resume=True)
            summary = run.run("all", convert=True)
            self.assertEqual(summary["active_arm"], "left")
            self.assertEqual(summary["dataset"]["episodes"], 2)
            self.assertEqual(summary["stages"]["augment"], dict(attempts=3, successful=2, success_rate=2/3))
            self.assertEqual((task.collected, task.prepared, task.generated, task.converted), (2, 2, 3, 1))
            before = (root / "generated.hdf5").stat().st_mtime_ns
            Pipeline(root, task, settings, resume=True).run("all", convert=True)
            self.assertEqual(before, (root / "generated.hdf5").stat().st_mtime_ns)
            self.assertEqual(task.generated, 3)
            self.assertEqual(len(list((root / "artifacts/prepare").glob("build_*"))), 2)
            with self.assertRaises(ValueError):
                Pipeline(root, task, dict(settings, augment=3), resume=True)
            with run_lock(root), self.assertRaises(RuntimeError):
                with run_lock(root):
                    pass

    def test_real_merge_remaps_source_ids_and_preserves_recordings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); second = root / "second.hdf5"
            shutil.copyfile(self.fixture, second)
            with h5py.File(second, "r+") as f:
                keys = json.loads(f["data"].attrs["source_demo_keys"])
                selected = [keys[int(i)] for i in f["data/demo_0/src_demo_inds"][:]]
                keys.reverse()
                f["data"].attrs["source_demo_keys"] = json.dumps(keys)
                f["data/demo_0/src_demo_inds"][...] = [keys.index(n) for n in selected]
            output = root / "merged.hdf5"
            merge([self.fixture, second], output, validate, prepared=True)
            with h5py.File(output, "r") as merged, h5py.File(self.fixture, "r") as original:
                self.assertEqual(len(merged["data"]), 2)
                for name in ("demo_0", "demo_1"):
                    for key in ("actions", "datagen_info/target_pose", "src_demo_inds", "diagnostics/phase"):
                        np.testing.assert_array_equal(merged[f"data/{name}/{key}"][:], original[f"data/demo_0/{key}"][:])
            config = json.loads(Path("mobipi/collect_demos/faucet_mimicgen_diverse_config.json").read_text())
            effective, metadata = faucet_attempt_config(config, "augment", 0, self.fixture, .02, 2)
            self.assertEqual(len(effective["mimicgen"]["fixture_offsets"]), 1)
            self.assertLessEqual(max(abs(v) for v in metadata["fixture_offset"][:3]), .02)

    def test_cli_individual_validation_and_resume_without_rollouts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            argv = ["pipeline", "--task", "TurnOnSinkFaucet", "--stage", "validate", "--augment", "1",
                    "--augmented-dataset", str(self.fixture), "--output", str(root)]
            with patch("sys.argv", argv):
                main()
            with patch("sys.argv", ["pipeline", "--task", "TurnOnSinkFaucet", "--stage", "validate",
                                    "--output", str(root), "--resume"]):
                main()
            self.assertTrue((root / "validation.json").exists())
            self.assertEqual(json.loads((root / "manifest.json").read_text())["attempts"], {"collect": [], "augment": []})
            missing = Path(directory) / "missing.hdf5"
            with patch("sys.argv", ["pipeline", "--task", "TurnOnSinkFaucet", "--stage", "prepare",
                                    "--prepared-dataset", str(missing), "--output", str(Path(directory) / "bad")]):
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(missing.exists())


if __name__ == "__main__":
    unittest.main()
