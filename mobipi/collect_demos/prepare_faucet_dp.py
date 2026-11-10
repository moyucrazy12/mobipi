"""Offline faucet conversion for the existing common-anchor relative Diffusion Policy.

Store absolute base-center TCP commands; the native dataset creates relative
labels for each prediction window. Preserve scalar gripper commands separately.
No simulator, IK, resampling, or observation/action time shift is used.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from mobipi.collect_demos.experiment_paths import new_output, resolve_training_paths
import shutil

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from robomimic.utils.rby1_relative_cartesian import (
    tcp_pose, transform_to_pose9, pose9_to_transform,
    decode_relative_target, tcp_observation_keys,
)

POSITION, QUATERNION = tcp_observation_keys("right")
CAMERAS = ["faucet_overview_image", "robot0_right_eye_in_hand_image"]


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if source == output or output.exists():
        raise ValueError("Output must be a new file; source remains read-only")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, output)
    report = dict(source=str(source), dataset=str(output), episodes=[],
                  alignment="obs[t] before command[t], no shift or resampling",
                  pose="base-center grip-site xyz + concatenated rotation columns 0,1",
                  relative="one measured TCP anchor at t for all 16 targets t-1..t+14",
                  gripper="actions_gripper scalar: -1 open, +1 closed; not transformed",
                  observation_fix="Reconstruct right base-relative TCP; upstream loop closures record left arm under both base-relative arm keys")
    # Group siblings by their ORIGINAL source, not by generated episode number.
    with h5py.File(output, "r+") as f:
        data = f["data"]
        canonical = json.loads(data.attrs["source_demo_keys"])
        held_out = set(np.random.default_rng(1).choice(canonical, 2, replace=False))
        masks = {"train": [], "valid": []}
        hashes = {"train": set(), "valid": set()}
        for name in sorted(data, key=lambda n: int(n.split("_")[-1])):
            ep = data[name]; info = ep["datagen_info"]; n = ep.attrs["num_samples"]
            sources = {canonical[int(i)] for i in ep["src_demo_inds"][:]}
            if len(sources) != 1:
                raise ValueError("Source-disjoint split requires single-source episodes")
            split = "valid" if sources & held_out else "train"
            masks[split].append(name)
            # Base sensors are correct; the original base-to-arm closures are not.
            base = tcp_pose(ep["obs/robot0_base_pos"][:], ep["obs/robot0_base_quat"][:])
            inv_base = np.linalg.inv(base)
            measured = info["eef_pose"][:]
            target = info["target_pose"][:]
            np.testing.assert_allclose(measured, ep["diagnostics/actual_tcp"][:], atol=1e-8)
            np.testing.assert_allclose(target, ep["diagnostics/commanded_tcp"][:], atol=1e-8)
            np.testing.assert_allclose(measured[:, :3, 3], ep["obs/robot0_right_eef_pos"][:], atol=1e-8)
            world_rot = Rotation.from_quat(ep["obs/robot0_right_eef_quat_site"][:]).as_matrix()
            np.testing.assert_allclose(measured[:, :3, :3], world_rot, atol=3e-6)
            # Raw actions are whole-body commands. Their right wrist pose differs
            # from the TCP by a rigid ~25 cm offset, not a Cartesian TCP label.
            raw = ep["actions"][:]
            wrist = tcp_pose(raw[:, :3], Rotation.from_rotvec(raw[:, 3:6]).as_quat())
            tcp_to_wrist = np.linalg.inv(target) @ wrist
            np.testing.assert_allclose(tcp_to_wrist, np.broadcast_to(tcp_to_wrist[0], tcp_to_wrist.shape), atol=1e-8)
            assert abs(np.linalg.norm(tcp_to_wrist[0, :3, 3]) - .2501) < .002
            observation = inv_base @ measured
            commands = transform_to_pose9(inv_base @ target).astype(np.float32)
            gripper = info["gripper_action"][:].astype(np.float32)
            assert commands.shape == (n, 9) and gripper.shape == (n, 1)
            assert np.isfinite(commands).all() and set(np.unique(gripper)) == {-1., 1.}
            np.testing.assert_allclose(pose9_to_transform(commands), inv_base @ target, atol=2e-6)
            for key, values in ((POSITION, observation[:, :3, 3]),
                                (QUATERNION, Rotation.from_matrix(observation[:, :3, :3]).as_quat())):
                ep["obs/" + key][...] = values
                ep["obs/" + key].attrs.update(frame="robot_base_center", corrected_right_tcp=True)
            stored = ep.create_dataset("actions_eef_cmd", data=commands)
            stored.attrs.update(frame="robot_base_center", tcp_site="gripper0_right_grip_site",
                                rotation6d="concatenate(R[:,0], R[:,1])", source="datagen_info/target_pose[t]",
                                temporal_alignment="obs[t] -> command[t]; no shift")
            ep.create_dataset("actions_gripper", data=gripper)
            for camera in CAMERAS:
                assert ep["obs/" + camera].shape == (n, 128, 128, 3)
                assert ep["obs/" + camera].dtype == np.uint8
            phases = Counter(ep["diagnostics/phase"].asstr()[:])
            assert set(phases) == {f"mimicgen_subtask_{i}" for i in (1, 2, 3)}
            hashes[split].add(hashlib.sha256(np.c_[commands, gripper].tobytes()).hexdigest())
            report["episodes"].append(dict(name=name, source=next(iter(sources)), split=split,
                                           steps=int(n), phase_steps=dict(phases)))
        assert not hashes["train"] & hashes["valid"]
        assert all(masks.values()) and len(data) == 100
        mask = f.require_group("mask")
        assert "train" not in mask and "valid" not in mask
        for split, names in masks.items():
            mask.create_dataset(split, data=np.asarray(names, dtype="S"))
        report["split"] = dict(seed=1, held_out_sources=sorted(held_out),
                               episodes={k: len(v) for k, v in masks.items()},
                               strategy="source-disjoint whole episodes; all siblings stay together",
                               limitation="Near-identical scripted sources and overlapping local fixture poses; not unseen-layout or geometric-generalization validation")
        data.attrs["faucet_dp_conversion"] = json.dumps({k: v for k, v in report.items() if k != "episodes"})
    return report


def validate_training(config_path, report):
    """One native dataset check: train-only statistics and three phase-spanning samples."""
    from robomimic.config import config_factory
    from robomimic.utils import obs_utils, train_utils
    from robomimic.utils import tensor_utils
    settings = resolve_training_paths(json.loads(config_path.read_text()), dataset=report["dataset"])
    config = config_factory(settings["algo_name"])
    with config.values_unlocked():
        config.update(settings)
    assert config.algo.relative_actions.arm == "right" and config.train.num_epochs == 15
    assert dict(config.algo.horizon) == dict(observation_horizon=2, prediction_horizon=16, action_horizon=8)
    assert config.algo.ddim.num_inference_timesteps == 10
    assert not config.experiment.rollout.enabled and not config.train.data[0]["do_eval"]
    assert Path(config.train.data[0]["path"]).expanduser().resolve() == Path(report["dataset"])
    obs_utils.initialize_obs_utils_with_config(config)
    train, valid = train_utils.load_data_for_training(config, obs_keys=config.all_obs_keys)
    try:
        stats = train.get_action_normalization_stats()
        valid.set_action_normalization_stats(stats)
        assert set(train.demos).isdisjoint(valid.demos)
        samples = []
        with h5py.File(report["dataset"], "r") as f:
            chosen = [(train, train.demos[0]), (train, train.demos[-1]), (valid, valid.demos[0])]
            for dataset, name in chosen:
                ep = f["data/" + name]
                phases = ep["diagnostics/phase"].asstr()[:]
                for phase in dict.fromkeys(phases):
                    indices = np.flatnonzero(phases == phase)
                    t = int(indices[len(indices) // 2])
                    sample = dataset[dataset._demo_id_to_start_indices[name] + t]
                    assert sample["actions"].shape == (16, 10) and np.isfinite(sample["actions"]).all()
                    if dataset is train:
                        assert np.abs(sample["actions"]).max() <= 1.00001
                    physical = obs_utils.unnormalize_dict(
                        {"actions_eef_cmd": sample["actions"][:, :9].copy(),
                         "actions_gripper": sample["actions"][:, 9:].copy()}, stats)
                    anchor = tcp_pose(sample["obs"][POSITION][1], sample["obs"][QUATERNION][1])
                    window = np.clip(t + np.arange(-1, 15), 0, len(phases) - 1)
                    targets = pose9_to_transform(ep["actions_eef_cmd"][:][window])
                    np.testing.assert_allclose(decode_relative_target(anchor, physical["actions_eef_cmd"]), targets, atol=3e-6)
                    np.testing.assert_allclose(physical["actions_gripper"], ep["actions_gripper"][:][window], atol=2e-6)
                    samples.append(dict(demo=name, phase=phase, t=t, normalized_abs_max=float(np.abs(sample["actions"]).max())))
        report["validation"] = dict(samples=samples, action_dim=10,
                                     normalization="native train-only min/max over fixed-anchor padded 16-target windows; reused for validation",
                                     normalization_stats=tensor_utils.to_list(stats))
    finally:
        train.close_and_delete_hdf5_handle()
        valid.close_and_delete_hdf5_handle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=new_output("demos", "faucet_relative_cartesian") / "train.hdf5")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("faucet_relative_cartesian_dp.json"))
    parser.add_argument("--report", type=Path, default=new_output("diagnostics", "faucet_preparation", ".json"))
    parser.add_argument("--validate-only", action="store_true", help="Check an already converted file without reconverting")
    args = parser.parse_args()
    path = args.report
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.validate_only:
        with h5py.File(args.output, "r") as f:
            report = json.loads(f["data"].attrs["faucet_dp_conversion"])
            train_names = set(f["mask/train"].asstr()[:])
            report["episodes"] = [dict(name=name, steps=int(ep.attrs["num_samples"]),
                source=json.loads(ep.attrs["source_demo_names"])[0],
                split="train" if name in train_names else "valid",
                phase_steps=dict(Counter(ep["diagnostics/phase"].asstr()[:]))) for name, ep in f["data"].items()]
    else:
        report = prepare(args.source, args.output)
    path.write_text(json.dumps(report, indent=2))
    validate_training(args.config, report)
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps({"dataset": report["dataset"], "split": report["split"],
                      "steps": sum(e["steps"] for e in report["episodes"]), "report": str(path)}, indent=2))


if __name__ == "__main__":
    main()
