"""Exclusive-create RoboMimic recording and structural / segmentation validation."""
import json
from pathlib import Path

import h5py
import numpy as np


def write_episode(group, env, initial, rows, origin, source_indices=None, datagen_infos=None):
    """Rows contain pre-action state, observation, metadata and exact executed command."""
    if not rows:
        raise ValueError("Cannot save an empty trajectory")
    for key in ("actions", "states", "rewards", "dones"):
        group.create_dataset(key, data=np.asarray([r[key] for r in rows]), compression="gzip")
    group["dones"][-1] = True
    group.parent.attrs["action_representation"] = "absolute_whole_body_ik_world_ft_frame"
    for key in rows[0]["obs"]:
        values = np.asarray([r["obs"][key] for r in rows])
        group.create_dataset("obs/"+key, data=values, compression="gzip")
    for key, value in rows[0]["diagnostics"].items():
        values = [r["diagnostics"][key] for r in rows]
        if isinstance(value, str):
            values = np.asarray(values, dtype=h5py.string_dtype())
        group.create_dataset("diagnostics/"+key, data=values)
    group.attrs.update(num_samples=len(rows), success=True, origin=origin,
                       termination_reason="success_after_safe_retreat", model_file=initial["model"],
                       ep_meta=initial.get("ep_meta", "{}"))
    if datagen_infos is not None:
        infos = [info.to_dict() for info in datagen_infos]
        for key, value in infos[0].items():
            if isinstance(value, dict):
                for subkey in value:
                    group.create_dataset(f"datagen_info/{key}/{subkey}", data=np.asarray([i[key][subkey] for i in infos]))
            else:
                group.create_dataset(f"datagen_info/{key}", data=np.asarray([i[key] for i in infos]))
        group["datagen_info"].attrs.update(env_interface_name="MG_RBY1TurnOnSinkFaucet", env_interface_type="robosuite")
    if source_indices is not None:
        group.create_dataset("src_demo_inds", data=source_indices)
    group.parent.attrs["env_args"] = json.dumps(env.serialize())
    group.parent.attrs["total"] = sum(int(g.attrs["num_samples"]) for g in group.parent.values())


def validate(path, prepared=False):
    """Reject invalid command/state alignment, poses and unusable MimicGen boundaries."""
    summary = []
    with h5py.File(path, "r") as f:
        data = f["data"]
        if not len(data):
            raise ValueError("Dataset contains no successful demonstrations")
        meta = json.loads(data.attrs["env_args"])
        if meta["env_name"] != "TurnOnSinkFaucetRBY1":
            raise ValueError("Wrong faucet environment")
        total = 0
        for name, ep in data.items():
            n = int(ep.attrs["num_samples"])
            if n < 3 or not ep.attrs.get("success", False):
                raise ValueError(f"{name}: empty or unsuccessful episode")
            for k in ("model_file", "ep_meta", "origin"):
                if k not in ep.attrs:
                    raise ValueError(f"{name}: missing {k}")
            for k in ("actions", "states", "rewards", "dones", "obs", "diagnostics"):
                if k not in ep:
                    raise ValueError(f"{name}: missing {k}")
            # Native RB-Y1 observables expose all 28 joints together, often as
            # sin/cos rather than a separately named right_joint_pos sensor.
            # Preserve that standard schema instead of inventing arm-only keys.
            obs = ep["obs"]
            named_arm = any("right" in k and "joint" in k for k in obs)
            combined = (all(k in obs for k in ("robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_right_eef_pos"))
                        and obs["robot0_joint_pos_cos"].shape == obs["robot0_joint_pos_sin"].shape == obs["robot0_joint_vel"].shape
                        and len(obs["robot0_joint_pos_cos"].shape) == 2
                        and obs["robot0_joint_pos_cos"].shape[1] >= 14)
            if not (named_arm or combined):
                raise ValueError(f"{name}: missing right-arm proprioception")
            if not any(k.endswith("_image") for k in ep["obs"]):
                raise ValueError(f"{name}: missing camera observations")
            def check_array(key, item):
                if not isinstance(item, h5py.Dataset) or key == "src_demo_inds":
                    return
                if item.shape[0] != n:
                    raise ValueError(f"{name}/{key}: length != {n}")
                a = item[()]
                if np.issubdtype(a.dtype, np.number) and not np.isfinite(a).all():
                    raise ValueError(f"{name}/{key}: NaN / Inf")
            ep.visititems(check_array)
            diag = ep["diagnostics"]
            if bool(diag["water_on"][0]) or not bool(diag["water_on"][-1]):
                raise ValueError(f"{name}: faucet must start off and finish on")
            boundaries = []
            if prepared:
                info = ep["datagen_info"]
                for key in ("eef_pose", "target_pose", "gripper_action", "object_poses", "subtask_term_signals"):
                    if key not in info:
                        raise ValueError(f"{name}: missing datagen_info/{key}")
                for key in ("eef_pose", "target_pose", "object_poses/faucet", "object_poses/handle"):
                    poses = info[key][()]
                    if poses.shape != (n, 4, 4) or not np.allclose(poses[:, 3], [0, 0, 0, 1]):
                        raise ValueError(f"{name}/{key}: invalid SE(3) shape")
                    rot = poses[:, :3, :3]
                    if not np.allclose(rot.transpose(0, 2, 1) @ rot, np.eye(3), atol=1e-5) or not np.allclose(np.linalg.det(rot), 1, atol=1e-5):
                        raise ValueError(f"{name}/{key}: invalid rotation")
                if not np.allclose(info["target_pose"], diag["commanded_tcp"], atol=1e-5):
                    raise ValueError(f"{name}: prepared targets differ from executed commands")
                stable = info["object_poses/faucet"][()]
                if not np.allclose(stable, stable[0], atol=1e-5):
                    raise ValueError(f"{name}: articulated frame used as fixture reference")
                for key in ("contact", "water_on"):
                    signal = info["subtask_term_signals/"+key][()].reshape(-1)
                    if not np.isin(signal, [0, 1]).all() or signal[0] != 0:
                        raise ValueError(f"{name}: invalid initial {key} signal")
                    edges = np.flatnonzero(np.diff(signal.astype(int)) == 1)
                    if not len(edges):
                        raise ValueError(f"{name}: no rising {key} boundary")
                    boundaries.append(int(edges[0])+1)
                if not 0 < boundaries[0] < boundaries[1] < n-1:
                    raise ValueError(f"{name}: invalid ordered subtask boundaries {boundaries}")
            total += n
            summary.append(dict(demo=name, samples=n, boundaries=boundaries))
        if total != int(data.attrs["total"]):
            raise ValueError("Global sample total mismatch")
    return summary
