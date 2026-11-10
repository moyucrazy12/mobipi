"""Paths for custom faucet workflows; no simulator or RoboMimic imports."""
import copy
from datetime import datetime
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TASK = "TurnOnFaucet"


def experiment_root():
    value = os.environ.get("RBY1_EXPERIMENT_ROOT")
    path = Path(os.path.expandvars(value)).expanduser() if value else PROJECT_ROOT / "experiment_results"
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def artifact_path(category, *parts):
    return experiment_root() / TASK / category / Path(*parts)


def new_output(category, name, suffix=""):
    return artifact_path(category, f"{name}_{datetime.now():%Y%m%d_%H%M%S_%f}{suffix}")


def historical_path(category, *parts):
    """Keep migrated evaluation inputs fixed even when output root is overridden."""
    return PROJECT_ROOT / "experiment_results" / TASK / category / Path(*parts)


def resolve_training_paths(settings, dataset=None):
    """Resolve only paths for native faucet dataset validation, not model settings."""
    settings = copy.deepcopy(settings)
    for item in settings["train"]["data"]:
        value = item["path"].replace("${RBY1_EXPERIMENT_ROOT}", str(experiment_root()))
        item["path"] = str(Path(os.path.expandvars(value)).expanduser().resolve())
    if dataset is not None:
        settings["train"]["data"][0]["path"] = str(Path(dataset).expanduser().resolve())
    settings["train"]["output_dir"] = str(artifact_path("train"))
    return settings
