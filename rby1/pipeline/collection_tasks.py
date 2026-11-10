"""Small registry of task adapters; simulator imports occur only during rollouts."""
from rby1.paths import CONFIG_ROOT
from dataclasses import dataclass
import importlib
import itertools
import json
from pathlib import Path
import random


@dataclass(frozen=True)
class Task:
    active_arm: str
    seed_config: str
    augmentation_config: str
    collect: str
    prepare: str
    augment: str
    validate: str
    attempt_config: str
    convert: str = ""

    def call(self, operation, *args, **kwargs):
        module, function = getattr(self, operation).split(":")
        return getattr(importlib.import_module(module), function)(*args, **kwargs)


def faucet_attempt_config(config, stage, index, source, translation, rotation):
    """Keep the existing planner/controller and vary only seeds and fixture pose."""
    config = json.loads(json.dumps(config))
    if stage == "collect":
        config["seed"] = config.get("seed", 0) + index
        return config, {"seed": config["seed"]}
    import h5py
    with h5py.File(source, "r") as f:
        keys = sorted(f["data"], key=lambda key: int(key.split("_")[-1]))
    if not 0 <= translation <= .03 or not 0 <= rotation <= 2:
        raise ValueError("Faucet limits: translation 0..0.03 m XY, yaw 0..2 degrees")
    mg = config["mimicgen"]
    base_seed = mg.get("seed", 1000)
    mg["seed"] = base_seed + index
    mg["source_demo_keys"] = keys
    for subtask in mg["task_spec"].values():
        if "nn_k" in subtask.get("selection_strategy_kwargs", {}):
            subtask["selection_strategy_kwargs"]["nn_k"] = min(len(keys), subtask["selection_strategy_kwargs"]["nn_k"])
    yaw = rotation * 3.141592653589793 / 180
    offsets = [[x, y, 0., angle] for x, y, angle in itertools.product(
        [-translation, -translation/2, 0., translation/2, translation],
        [-translation, -translation/2, 0., translation/2, translation], [-yaw, 0., yaw])]
    random.Random(base_seed).shuffle(offsets)
    offset = offsets[index % len(offsets)]
    mg["fixture_offsets"] = [offset]
    return config, {"seed": mg["seed"], "fixture_offset": offset, "source_demo_keys": keys}


_HERE = CONFIG_ROOT
TASKS = {
    "TurnOnSinkFaucet": Task(
        active_arm="right",
        seed_config=str(_HERE / "faucet_config.json"),
        augmentation_config=str(_HERE / "faucet_mimicgen_diverse_config.json"),
        collect="rby1.demos.collect_faucet_demos:collect",
        prepare="rby1.demos.faucet_pipeline:prepare",
        augment="rby1.demos.faucet_pipeline:generate",
        validate="rby1.demos.faucet_dataset:validate",
        attempt_config="rby1.pipeline.collection_tasks:faucet_attempt_config",
        convert="rby1.policies.prepare_faucet_dp:prepare",
    ),
}
