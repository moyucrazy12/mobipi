"""A legacy import must share globals and classes with its canonical module."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rby1.paths import REPOSITORY_ROOT


class CompatibilityTests(unittest.TestCase):
    def test_task_registry_and_child_modules_keep_identity(self):
        legacy = importlib.import_module("mobipi.pipeline.tasks")
        canonical = importlib.import_module("rby1.pipeline.tasks")
        self.assertIs(legacy.TASKS, canonical.TASKS)
        for name in ("base", "close_drawer", "close_microwave"):
            self.assertIs(importlib.import_module("mobipi.pipeline.tasks." + name),
                          importlib.import_module("rby1.pipeline.tasks." + name))

    def test_legacy_direct_scripts_work_from_another_directory(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(str(REPOSITORY_ROOT / "external" / name)
                                          for name in ("robomimic", "mimicgen", "robocasa"))
        with tempfile.TemporaryDirectory() as temporary:
            for script in ("mobipi/collect_demos/collect_faucet_demos.py",
                           "mobipi/collect_demos/collect_close_drawer_demos.py",
                           "mobipi/scripts/run_rby1_pipeline.py"):
                result = subprocess.run([sys.executable, str(REPOSITORY_ROOT / script), "--help"],
                                        cwd=temporary, env=env, capture_output=True, text=True, timeout=45)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_legacy_modules_share_identity_and_patch_targets(self):
        migration = json.loads((REPOSITORY_ROOT / "docs/codebase_cleanup/migration.json").read_text())
        for old, new in migration.items():
            if not old.endswith(".py") or "/tests/" in old or old.endswith(("test_mobilization.py", "/__init__.py")):
                continue
            with self.subTest(module=old):
                legacy = importlib.import_module(old[:-3].replace("/", "."))
                canonical = importlib.import_module(new[:-3].replace("/", "."))
                self.assertIs(legacy, canonical)
                with patch.object(legacy, "_compatibility_probe", 42, create=True):
                    self.assertEqual(canonical._compatibility_probe, 42)

    def test_legacy_config_paths_resolve_to_one_source(self):
        migration = json.loads((REPOSITORY_ROOT / "docs/codebase_cleanup/migration.json").read_text())
        for old, new in migration.items():
            if old.endswith(".json"):
                with self.subTest(config=old):
                    path = REPOSITORY_ROOT / old
                    self.assertTrue(path.is_symlink())
                    self.assertEqual(path.resolve(), REPOSITORY_ROOT / new)
                    self.assertEqual(json.loads(path.read_text()), json.loads((REPOSITORY_ROOT / new).read_text()))


if __name__ == "__main__":
    unittest.main()
