#!/usr/bin/env python3
"""
Gather RB-Y1 drawer demos from collect_close_drawer_demos.py into one robomimic HDF5.

Reads the successful demos listed in each input directory's manifest.csv (the
raw DataCollectionWrapper episodes: model.xml, ep_meta.json, state_*.npz) and
writes the format that MimicGen's prepare_src_dataset.py, robomimic's
dataset_states_to_obs.py and the RB-Y1 diffusion policy pipeline read:

    data/                 attrs: env_args (robomimic env metadata), total
      demo_1/             attrs: model_file, ep_meta, num_samples, source
        states            (T, state_dim)  state in which actions[t] was taken
        actions           (T, action_dim) absolute whole-body IK commands
        actions_abs       (T, action_dim) same as actions: they are already absolute
      ...

env_args recreates the collector's environment from the env_kwargs.json that
auto_close_drawer.py saves with each recording (same controller, base noise and
drawer opening range; all demos must agree), with layout_ids / style_ids
covering all gathered demos. Recordings without env_kwargs.json (older batches)
get the default collection settings. Train /
validation masks are not written; add them with robomimic's split_train_val.py.

Example:
    python gather_demos_hdf5.py --input rby1_drawer_demos_10 --output datasets/close_drawer/demo.hdf5
"""
import argparse
import csv
import datetime
import json
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import robosuite
from robosuite.controllers import load_composite_controller_config

from auto_close_microwave_door import DEFAULT_CONTROLLER

ROBOSUITE_ENV_TYPE = 1  # robomimic.envs.env_base.EnvType.ROBOSUITE_TYPE


def load_episode(ep_dir):
    """States and actions of one raw episode, with states[t] the state before actions[t]."""
    states, actions, successful, env_names = [], [], False, set()
    for state_file in sorted(glob(str(ep_dir / "state_*.npz"))):
        dic = np.load(state_file, allow_pickle=True)
        env_names.add(str(dic["env"]))
        states.extend(dic["states"])
        actions.extend(ai["actions"] for ai in dic["action_infos"])
        successful = successful or bool(dic["successful"])
    # The wrapper stores the initial state and the state after every action
    if len(states) != len(actions) + 1:
        raise RuntimeError(f"{ep_dir}: {len(states)} states for {len(actions)} actions")
    return np.array(states[:-1]), np.array(actions), successful, env_names


def episode_dirs(input_dir):
    """(demo directory, manifest row) of every successful demo, in manifest order."""
    with (input_dir / "manifest.csv").open() as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        demo_dir = Path(row["directory"])
        if not demo_dir.is_dir():
            # the batch directory may have been moved since collection
            demo_dir = input_dir / demo_dir.name
        eps = sorted(demo_dir.glob("ep_*"))
        if len(eps) != 1:
            raise RuntimeError(f"{demo_dir}: expected one ep_* directory, found {len(eps)}")
        yield eps[0], row


# Per-episode or on-screen settings, not part of the environment definition
EPISODE_KWARGS = ("env_name", "layout_ids", "style_ids", "seed", "has_renderer", "has_offscreen_renderer",
                  "render_camera", "renderer", "use_camera_obs")


def recorded_env_kwargs(demo_dir):
    """Environment kwargs saved by auto_close_drawer.py, without the per-episode ones (None if absent)."""
    path = demo_dir / "env_kwargs.json"
    if not path.exists():
        return None
    kwargs = json.loads(path.read_text())
    return {k: v for k, v in kwargs.items() if k not in EPISODE_KWARGS}


def env_args(env_name, recorded, controller, control_freq, layouts, styles):
    """robomimic env metadata for the environment auto_close_drawer.py records in."""
    if recorded is not None:
        env_kwargs = dict(recorded, layout_ids=layouts, style_ids=styles, has_renderer=False,
                          has_offscreen_renderer=False, use_camera_obs=False)
        return dict(env_name=env_name, env_version=robosuite.__version__, type=ROBOSUITE_ENV_TYPE, env_kwargs=env_kwargs)
    env_kwargs = {
        "robots": "RBY1",
        "controller_configs": load_composite_controller_config(controller=controller, robot="RBY1"),
        "layout_ids": layouts,
        "style_ids": styles,
        "has_renderer": False,
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_camera_obs": False,
        "control_freq": control_freq,
        "translucent_robot": False,
        "initialization_noise": None,
        "randomize_cameras": False,
    }
    return dict(env_name=env_name, env_version=robosuite.__version__, type=ROBOSUITE_ENV_TYPE, env_kwargs=env_kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", required=True, help="Batch output directories (with manifest.csv)")
    parser.add_argument("--output", required=True, help="HDF5 file to write")
    parser.add_argument("--environment", default="CloseDrawerLeft")
    parser.add_argument("--controller", default=DEFAULT_CONTROLLER, help="Controller JSON the demos were collected with (only used without env_kwargs.json)")
    parser.add_argument("--control-freq", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)

    layouts, styles, total, num_demos = set(), set(), 0, 0
    recorded, recorded_from = None, None
    with h5py.File(output, "w") as f:
        data = f.create_group("data")
        for input_dir in args.input:
            input_dir = Path(input_dir).expanduser().resolve()
            for ep_dir, row in episode_dirs(input_dir):
                states, actions, successful, env_names = load_episode(ep_dir)
                if env_names != {args.environment}:
                    raise RuntimeError(f"{ep_dir}: recorded in {env_names}, expected {args.environment}")
                if not successful:
                    raise RuntimeError(f"{ep_dir}: listed in the manifest but not marked successful")
                kwargs = recorded_env_kwargs(ep_dir.parent)
                if num_demos == 0:
                    recorded, recorded_from = kwargs, ep_dir.parent
                elif kwargs != recorded:
                    raise RuntimeError(f"{ep_dir.parent} was collected with different environment settings "
                                       f"than {recorded_from}; gather them into separate datasets")
                ep_meta = json.loads((ep_dir / "ep_meta.json").read_text())
                layouts.add(int(ep_meta["layout_id"]))
                styles.add(int(ep_meta["style_id"]))

                num_demos += 1
                grp = data.create_group(f"demo_{num_demos}")
                grp.attrs["model_file"] = (ep_dir / "model.xml").read_text()
                grp.attrs["ep_meta"] = json.dumps(ep_meta, indent=4)
                grp.attrs["num_samples"] = len(actions)
                grp.attrs["source"] = str(ep_dir)
                grp.create_dataset("states", data=states)
                grp.create_dataset("actions", data=actions)
                grp.create_dataset("actions_abs", data=actions)
                total += len(actions)
                print(f"demo_{num_demos}: {len(actions)} steps, layout {row['layout']} style {row['style']} "
                      f"seed {row['seed']}  <- {ep_dir}")

        if num_demos == 0:
            raise RuntimeError("No demos found")
        if recorded is None:
            print("\nNo env_kwargs.json in the recordings; using the default collection settings")
        data.attrs["env_args"] = json.dumps(env_args(
            args.environment, recorded, args.controller, args.control_freq, sorted(layouts), sorted(styles)), indent=4)
        data.attrs["total"] = total
        data.attrs["date"] = datetime.datetime.now().isoformat(timespec="seconds")

    print(f"\nWrote {num_demos} demos ({total} steps) to {output}")
    print(f"layouts {sorted(layouts)}, styles {sorted(styles)}")


if __name__ == "__main__":
    main()
