"""Organize NPZ recordings and export right-arm, head/wrist RGB RoboMimic data.

Keep recorded action[7:14]: translation deltas, XYZ Euler rotation deltas,
and future right-gripper width. No pose9 conversion, resampling or time shift.
"""
import argparse
import csv
from io import BytesIO
import json
from pathlib import Path
import pickle
import re
from zipfile import ZipFile

import h5py
import numpy as np


CAMERAS = {"head_rgb": "head_rgb", "right_wrist_rgb": "wrist_rgb"}
LOW_DIM = ("right_joint_pos", "right_joint_vel", "right_ee_pos",
           "right_ee_quat", "right_gripper_width")
ACTION_SPEC = dict(arm="right", source="recorded action[t][7:14]",
                   layout=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_width"],
                   translation="metres, unchanged recording reference axes",
                   rotation="extrinsic xyz Euler radians; R_next = R_delta @ R_current",
                   gripper="absolute opening in metres at t+1; not an open/close sign",
                   alignment="obs[t] -> recorded transition t to t+1; final pose delta is zero")


class _NumpyObjects(pickle.Unpickler):
    """Read recorder object arrays across NumPy 1/2 without arbitrary classes."""
    def find_class(self, module, name):
        allowed = {("numpy", "ndarray"): np.ndarray, ("numpy", "dtype"): np.dtype}
        for prefix in ("numpy.core.multiarray", "numpy._core.multiarray"):
            allowed[(prefix, "_reconstruct")] = np.core.multiarray._reconstruct
            allowed[(prefix, "scalar")] = np.core.multiarray.scalar
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"Unsupported recording object: {module}.{name}")
        return allowed[(module, name)]


def load_recording(directory):
    files = list(directory.glob("*.npz"))
    if len(files) != 1:
        raise ValueError(f"{directory}: expected exactly one trajectory NPZ")
    values = {}
    with ZipFile(files[0]) as archive:
        for key in ("trajectory_metadata", "arr_0"):
            stream = BytesIO(archive.read(key + ".npy"))
            version = np.lib.format.read_magic(stream)
            reader = {(1, 0): np.lib.format.read_array_header_1_0,
                      (2, 0): np.lib.format.read_array_header_2_0}.get(version)
            if reader is None:
                raise ValueError(f"Unsupported NPY header: {version}")
            shape, _, dtype = reader(stream)
            value = _NumpyObjects(stream).load()
            if not isinstance(value, np.ndarray) or value.shape != shape or value.dtype != dtype:
                raise ValueError(f"{files[0]}: inconsistent {key} array")
            values[key] = value
    metadata, rows = values["trajectory_metadata"].item(), values["arr_0"]
    if rows.ndim != 1 or len(rows) < 2 or len(rows) != metadata["num_actions"]:
        raise ValueError(f"{files[0]}: inconsistent trajectory length")
    return metadata, rows


def _number(path):
    return int(path.name.rsplit("_", 1)[1])


def organize(source, exclude=(), skip_incomplete=False):
    """Move session sequences once; existing recordings remain in place."""
    source = source.expanduser().resolve()
    recordings = source if source.name == "recordings" else source / "recordings"
    sessions = sorted((p for p in source.iterdir() if p.is_dir()
                       and re.fullmatch(r"session_\d+", p.name)), key=_number)
    existing = list(recordings.glob("demo_*")) if recordings.exists() else []
    if recordings.exists() and (not sessions or recordings != source or existing):
        if sessions or exclude:
            raise ValueError("Existing recordings found: do not mix with sessions or new exclusions")
        demos = sorted((p for p in recordings.iterdir() if p.is_dir()
                        and re.fullmatch(r"demo_\d+", p.name)), key=_number)
        if [p.name for p in demos] != [f"demo_{i}" for i in range(1, len(demos) + 1)]:
            raise ValueError("Recording directory numbers must be contiguous, starting at demo_1")
        if not demos:
            raise ValueError("No recordings found")
        return demos
    sequences = [p for session in sessions for p in sorted(session.iterdir(), key=lambda p: p.name)
                 if p.is_dir() and re.fullmatch(r"sequence_\d+", p.name)]
    sequences.sort(key=lambda p: (_number(p.parent), _number(p)))
    keys = {p.relative_to(source).as_posix() for p in sequences}
    if not sequences or set(exclude) - keys:
        raise ValueError("No sequences found, or an exclusion does not identify an existing sequence")
    plan = []
    demos = []
    for path in sequences:
        key = path.relative_to(source).as_posix()
        reason = "explicitly excluded" if key in exclude else ""
        if not reason:
            if len(list(path.glob("*.npz"))) != 1 and skip_incomplete:
                reason = "incomplete: missing or ambiguous trajectory NPZ"
            else:
                load_recording(path)  # Validate before moving any directories.
        target = source / "excluded" / key if reason else recordings / f"demo_{len(demos) + 1}"
        if target.exists():
            raise FileExistsError(target)
        if not reason:
            demos.append(target)
        plan.append((path, target, key, reason))
    if len(demos) < 2:
        raise ValueError("At least two complete recordings are needed for train/valid masks")
    manifest = source / "recordings_manifest.csv"
    with manifest.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["recording", "original_sequence", "status", "reason"])
        writer.writeheader()
        for _, target, key, reason in plan:
            writer.writerow(dict(recording=target.relative_to(source), original_sequence=key,
                                 status="excluded" if reason else "included", reason=reason))
    for path, target, _, _ in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)
    for session in sessions:
        if not any(session.iterdir()):
            session.rmdir()
    return demos


def _finite(values, name):
    result = np.asarray(values)
    if not np.isfinite(result).all():
        raise ValueError(f"{name}: nonfinite values")
    return result


def convert(demos, output, source_color_order, manifest=None, valid_ratio=.1, seed=2026):
    output = output.expanduser().resolve()
    partial = output.with_name(output.name + ".partial")
    if output.exists() or partial.exists():
        raise FileExistsError(f"Output or unfinished conversion already exists: {output}")
    if len(demos) < 2 or not 0 < valid_ratio < 1 or source_color_order not in ("rgb", "bgr"):
        raise ValueError("Need >=2 demos, validation ratio in (0,1), and rgb/bgr source order")
    origins = {}
    if manifest:
        with manifest.open(newline="") as stream:
            origins = {Path(r["recording"]).name: r["original_sequence"]
                       for r in csv.DictReader(stream) if r["status"] == "included"}
    count = min(len(demos) - 1, max(1, int(np.ceil(len(demos) * valid_ratio))))
    valid = set(np.random.default_rng(seed).choice(len(demos), count, replace=False).tolist())
    masks = {"train": [], "valid": []}
    report = dict(dataset=str(output), episodes=len(demos), samples=0, action_dim=7,
                  cameras=list(CAMERAS), source_color_order=source_color_order, split_seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(partial, "x") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps(dict(env_name="CubeInCup", type=None,
            env_kwargs={}, real_robot=True, offline_only=True))
        data.attrs.update(action_spec=json.dumps(ACTION_SPEC), quaternion_order="wxyz",
                          color_order="rgb", source_color_order=source_color_order,
                          split="whole episodes; overlapping sessions, not unseen-session validation",
                          rewards="zero placeholders; dones mark recording boundaries, not verified success")
        for index, directory in enumerate(demos):
            metadata, rows = load_recording(directory)
            n = len(rows)
            raw = _finite(np.stack([r["action"] for r in rows]), directory.name)
            if raw.shape != (n, 20) or raw.dtype != np.float32:
                raise ValueError(f"{directory}: expected recorded float32 (N,20) actions")
            times = _finite([r["timestamp"] for r in rows], "timestamps")
            if not np.all(np.diff(times) > 0):
                raise ValueError(f"{directory}: timestamps must increase; no resampling is performed")
            joints = [metadata["joint_names"].index(f"right_arm_{j}") for j in range(7)]
            observations = {key: _finite(np.stack([r["obs"][key] for r in rows]), key)
                            for key in ("joint_pos", "joint_vel", "right_ee_pos", "right_ee_quat", "grippers_width")}
            if not np.allclose(np.linalg.norm(observations["right_ee_quat"], axis=1), 1, atol=1e-3):
                raise ValueError(f"{directory}: right-arm quaternions are not normalized")
            name = f"demo_{index}"
            ep = data.create_group(name)
            ep.attrs.update(num_samples=n, source_recording=str(directory),
                            original_sequence=origins.get(directory.name, directory.name),
                            recording_metadata=json.dumps(metadata), nominal_fps=metadata["fps"])
            ep.create_dataset("actions", data=raw[:, 7:14], compression="lzf")
            ep.create_dataset("timestamps", data=times)
            ep.create_dataset("rewards", data=np.zeros(n, dtype=np.float32))
            done = np.zeros(n, dtype=np.uint8); done[-1] = 1
            ep.create_dataset("dones", data=done)
            obs = ep.create_group("obs")
            for key, values in dict(right_joint_pos=observations["joint_pos"][:, joints],
                    right_joint_vel=observations["joint_vel"][:, joints],
                    right_ee_pos=observations["right_ee_pos"], right_ee_quat=observations["right_ee_quat"],
                    right_gripper_width=observations["grippers_width"][:, 1:2]).items():
                obs.create_dataset(key, data=values, compression="lzf")
            for target, camera in CAMERAS.items():
                images = None
                for t, row in enumerate(rows):
                    path = (directory / row["obs"][camera]).resolve()
                    if not path.is_relative_to(directory.resolve()):
                        raise ValueError(f"Camera path escapes recording: {path}")
                    with np.load(path, allow_pickle=False) as archive:
                        image = archive["frame"]
                    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
                        raise ValueError(f"{path}: expected uint8 HWC color image")
                    if source_color_order == "bgr":
                        image = image[..., ::-1]
                    if images is None:
                        images = obs.create_dataset(target, shape=(n, *image.shape), dtype=np.uint8,
                                                    chunks=(1, *image.shape), compression="lzf")
                    if image.shape != images.shape[1:]:
                        raise ValueError(f"{path}: changing image dimensions")
                    images[t] = image
                images.attrs.update(color_order="rgb", source_camera=camera)
            masks["valid" if index in valid else "train"].append(name)
            report["samples"] += n
            print(f"Converted {directory.name}: {n} samples", flush=True)
        for split, names in masks.items():
            f.create_dataset("mask/" + split, data=np.asarray(names, dtype="S"))
        data.attrs.update(total=report["samples"], conversion_complete=True)
    if output.exists():
        raise FileExistsError(output)
    partial.rename(output)
    report["split"] = {k: len(v) for k, v in masks.items()}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Sessions parent, or recordings directory")
    parser.add_argument("--output", type=Path, required=True, help="New HDF5 file outside Git")
    parser.add_argument("--source-color-order", choices=("rgb", "bgr"), required=True)
    parser.add_argument("--exclude", action="append", default=[], help="session_004/sequence_009, for example")
    parser.add_argument("--skip-incomplete", action="store_true", help="Quarantine incomplete sequences")
    parser.add_argument("--manifest", type=Path, help="Existing source mapping CSV; otherwise use organizer's manifest")
    parser.add_argument("--valid-ratio", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.source = args.source.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.manifest:
        args.manifest = args.manifest.expanduser().resolve()
    if not 0 < args.valid_ratio < 1 or args.seed < 0:
        parser.error("--valid-ratio must be in (0,1) and --seed must be nonnegative")
    if args.output.exists() or args.output.with_name(args.output.name + ".partial").exists():
        parser.error("Output or .partial already exists")
    demos = organize(args.source, args.exclude, args.skip_incomplete)
    candidates = [demos[0].parent / "recordings_manifest.csv",
                  demos[0].parent.parent / "recordings_manifest.csv"]
    manifest = args.manifest or next((p for p in candidates if p.exists()), None)
    report = convert(demos, args.output, args.source_color_order,
                     manifest=manifest, valid_ratio=args.valid_ratio, seed=args.seed)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
