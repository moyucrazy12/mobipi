"""Portable CubeInCup comparisons using native RoboMimic training and loaders."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from .cube_actions import (EXPERIMENTS, WIDTH_THRESHOLD, binary_gripper,
                          measured_poses, rotation6d_from_euler, decode_chunk, GripperHysteresis)
from robomimic.utils.rby1_relative_cartesian import transform_to_pose9, encode_relative_target

VERSION = 1
CONFIG_DIR = Path(__file__).with_name('configs')
OBS_KEYS = ('right_joint_pos', 'right_joint_vel', 'right_ee_pos', 'right_ee_quat',
            'right_gripper_width', 'head_rgb', 'right_wrist_rgb')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def split_names(f, split):
    return [s.decode() for s in f['mask'][split][:]]


def contract(experiment):
    return dict(version=VERSION, experiment=experiment, frame='WBC world / end_effector_r',
                quaternion_order='wxyz', resampled=False, source='achieved measured motion',
                target_alignment='action row i targets measured pose min(i+1,N-1)',
                gripper='width_m' if experiment == 'baseline_7d' else '-1=open,+1=closed',
                width_threshold_m=WIDTH_THRESHOLD,
                anchor='fixed final observation' if experiment == 'relative_10d' else 'incremental world')


def prepare(experiment, source, output):
    if experiment == 'baseline_7d':
        raise ValueError('Baseline is immutable: use its original HDF5; no conversion')
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or output.exists():
        raise FileExistsError(output)
    validate('baseline_7d', source)
    source_hash = sha256(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=output.name + '.', suffix='.partial', dir=output.parent)
    os.close(fd)
    try:
        # Standalone copies are portable; external links/VDS require companion paths.
        shutil.copyfile(source, temp)
        with h5py.File(temp, 'r+') as f:
            f.attrs['cube_contract'] = json.dumps(contract(experiment), sort_keys=True)
            f.attrs['source_sha256'] = source_hash
            # Baseline env metadata describes 7D actions; replace stale action spec.
            f['data'].attrs['action_spec'] = json.dumps(contract(experiment), sort_keys=True)
            env = json.loads(f['data'].attrs['env_args'])
            env['env_kwargs'].update(cube_experiment=experiment, cube_source_sha256=source_hash)
            f['data'].attrs['env_args'] = json.dumps(env, sort_keys=True)
            for ep in f['data'].values():
                a = ep['actions'][:]
                grip = binary_gripper(a[:, -1:])
                if experiment == 'binary_7d':
                    actions = np.concatenate((a[:, :6], grip), axis=-1)
                elif experiment == 'rot6d_10d':
                    actions = np.concatenate((a[:, :3], rotation6d_from_euler(a[:, 3:6]), grip), axis=-1)
                else:
                    poses = measured_poses(ep['obs'])
                    future = poses[np.minimum(np.arange(len(a)) + 1, len(a) - 1)]
                    pose9 = transform_to_pose9(future).astype(np.float32)
                    ep.create_dataset('actions_eef_cmd', data=pose9)
                    ep.create_dataset('actions_gripper', data=grip)
                    actions = np.concatenate((pose9, grip), axis=-1)
                del ep['actions']
                ep.create_dataset('actions', data=actions.astype(np.float32), compression='lzf')
        validate(experiment, temp, source=source)
        # No-clobber publication on same filesystem, including concurrent prepares.
        os.link(temp, output)
    finally:
        Path(temp).unlink(missing_ok=True)
    return dict(path=str(output), bytes=output.stat().st_size, sha256=sha256(output))


def validate(experiment, dataset, source=None):
    counts = {}; worst_pos = 0.; worst_angle = 0.
    with h5py.File(dataset, 'r') as f:
        if experiment != 'baseline_7d':
            if json.loads(f.attrs.get('cube_contract', '{}')) != contract(experiment):
                raise ValueError('Dataset action contract mismatch')
        elif 'cube_contract' in f.attrs:
            raise ValueError('Converted data cannot be used as the baseline')
        train, valid = split_names(f, 'train'), split_names(f, 'valid')
        if len(train) != len(set(train)) or len(valid) != len(set(valid)) or set(train) & set(valid):
            raise ValueError('Duplicate episodes or train/validation leakage')
        if set(train + valid) != set(f['data']):
            raise ValueError('Split does not cover all demonstrations')
        for split, names in [('train', train), ('valid', valid)]:
            counts[split] = dict(demos=len(names), windows=sum(len(f['data'][n]['actions']) for n in names))
        for name, ep in f['data'].items():
            a = ep['actions'][:]; n = len(a)
            dim = 7 if experiment in EXPERIMENTS[:2] else 10
            if a.shape != (n, dim) or not np.isfinite(a).all():
                raise ValueError(f'{name}: invalid actions')
            if n < 2 or not np.all(np.diff(ep['timestamps'][:]) > 0):
                raise ValueError(f'{name}: invalid timestamps')
            if set(ep['obs']) != set(OBS_KEYS):
                raise ValueError(f'{name}: unexpected observations')
            for key in OBS_KEYS:
                ds = ep['obs'][key]
                if len(ds) != n:
                    raise ValueError('Observation length mismatch')
                for start in range(0, n, 128):
                    if not np.isfinite(ds[start:start+128]).all():
                        raise ValueError('Non-finite observations')
            poses = measured_poses(ep['obs']); future = np.minimum(np.arange(n)+1, n-1)
            # One-step checks for incremental labels; exhaustive fixed-anchor windows for D.
            if experiment == 'relative_10d':
                if not np.array_equal(a[:, :9], ep['actions_eef_cmd'][:]) or not np.array_equal(a[:, 9:], ep['actions_gripper'][:]):
                    raise ValueError('D action fields disagree')
                from robomimic.utils.rby1_relative_cartesian import pose9_to_transform, decode_relative_target
                targets = pose9_to_transform(a[:, :9])
                indices = np.clip(np.arange(n)[:, None] + np.arange(-1, 15), 0, n-1)
                encoded = encode_relative_target(poses[:, None], targets[indices]).astype(np.float32)
                decoded = decode_relative_target(poses[:, None], encoded)
                expected = poses[future][indices]
            else:
                decoded = np.stack([decode_chunk(experiment, row[None], pose)[0] for row, pose in zip(a, poses)])
                expected = poses[future]
            pos = float(np.max(np.linalg.norm(decoded[..., :3, 3]-expected[..., :3, 3], axis=-1)))
            relative = decoded[..., :3, :3] @ expected[..., :3, :3].swapaxes(-1, -2)
            angle = float(np.max(Rotation.from_matrix(relative.reshape(-1, 3, 3)).magnitude()))
            worst_pos, worst_angle = max(worst_pos, pos), max(worst_angle, angle)
            if pos > 1e-5 or angle > 1e-5:
                raise ValueError(f'{name}: pose reconstruction error {pos} m / {angle} rad')
            if experiment != 'baseline_7d' and not np.isin(a[:, -1], [-1, 1]).all():
                raise ValueError('Gripper is not binary')
        if source is not None:
            with h5py.File(source, 'r') as original:
                for split in ('train','valid'):
                    np.testing.assert_array_equal(f['mask'][split], original['mask'][split])
                for name, ep in f['data'].items():
                    base = original['data'][name]
                    np.testing.assert_array_equal(ep['timestamps'], base['timestamps'])
                    np.testing.assert_array_equal(ep['actions'][:, -1], binary_gripper(base['actions'][:, -1]))
                    if experiment == 'binary_7d':
                        np.testing.assert_array_equal(ep['actions'][:, :6], base['actions'][:, :6])
                    elif experiment == 'rot6d_10d':
                        np.testing.assert_array_equal(ep['actions'][:, :3], base['actions'][:, :3])
                    for key in OBS_KEYS:
                        for start in range(0, len(ep['actions']), 128):
                            np.testing.assert_array_equal(ep['obs'][key][start:start+128], base['obs'][key][start:start+128])
        return dict(experiment=experiment, splits=counts, max_position_error_m=worst_pos,
                    max_orientation_error_deg=float(np.rad2deg(worst_angle)))


def configuration(experiment, dataset, output):
    from robomimic.config import config_factory
    raw = json.loads((CONFIG_DIR / f'cube_{experiment}.json').read_text())
    raw['train']['data'] = [dict(path=str(Path(dataset).resolve()), do_eval=False)]
    raw['train']['output_dir'] = str(Path(output).resolve())
    return config_factory('diffusion_policy', dic=raw)


def load_training_data(config):
    from robomimic.utils import obs_utils, train_utils
    obs_utils.initialize_obs_utils_with_config(config)
    keys = config.observation.modalities.obs.low_dim + config.observation.modalities.obs.rgb
    return train_utils.load_data_for_training(config, obs_keys=keys)


def train(experiment, dataset, output, device):
    if experiment == 'baseline_7d':
        raise ValueError('Launch the existing baseline with its original command; this CLI never changes it')
    validate(experiment, dataset)
    output = Path(output).resolve()
    # Native get_exp_dir can prompt to delete prior runs: refuse preexisting root.
    output.mkdir(parents=True, exist_ok=False)
    config = configuration(experiment, dataset, output)
    config.experiment.env_meta_update_dict.cube_dataset_sha256 = sha256(dataset)
    from robomimic.scripts.train import train as native_train
    import torch
    native_train(config, torch.device(device))


def evaluate_offline(experiment, dataset, checkpoint, output, stride=8, max_windows=None):
    """Teacher-forced replans, open-loop predicted chunks; never robot rollouts."""
    import torch
    from torch.utils.data import DataLoader, Subset
    from robomimic.utils import file_utils
    validate(experiment, dataset)
    if stride < 1 or (max_windows is not None and max_windows < 1):
        raise ValueError('stride and max-windows must be positive')
    rollout, ckpt = file_utils.policy_from_checkpoint(ckpt_path=str(checkpoint), device=torch.device('cpu'))
    from robomimic.config import config_factory
    saved = config_factory('diffusion_policy', dic=json.loads(ckpt['config']))
    expected = configuration(experiment, dataset, '/tmp/unused')
    # Strict semantics: dimension alone cannot distinguish C and D.
    for key in ('action_keys', 'action_config', 'frame_stack', 'seq_length', 'pad_frame_stack', 'pad_seq_length'):
        if saved['train'][key] != expected.train[key]:
            raise ValueError(f'Checkpoint incompatible: {key}')
    if saved['observation'] != expected.observation or saved['algo'] != expected.algo:
        raise ValueError('Checkpoint observation/algorithm contract mismatch')
    metadata = saved['experiment']['env_meta_update_dict']
    if experiment != 'baseline_7d' and metadata.get('cube_dataset_sha256') != sha256(dataset):
        raise ValueError('Checkpoint dataset checksum mismatch')
    marker = metadata.get('cube_experiment')
    if marker != experiment and experiment != 'baseline_7d':
        raise ValueError('Checkpoint experiment marker mismatch')
    trainset, validset = load_training_data(expected)
    stats = trainset.get_action_normalization_stats()
    actual = ckpt.get('action_normalization_stats')
    if not actual or set(actual) != set(stats):
        raise ValueError('Missing/incompatible normalization keys')
    for key in stats:
        for field in ('scale','offset'):
            if np.asarray(actual[key][field]).shape != stats[key][field].shape or not np.allclose(actual[key][field], stats[key][field], rtol=1e-6, atol=1e-8):
                raise ValueError('Checkpoint normalization differs from this training split')
    validset.set_action_normalization_stats(stats)
    if rollout.policy.ac_dim != (7 if experiment in EXPERIMENTS[:2] else 10):
        raise ValueError('Checkpoint dimension mismatch')
    episodes = validset.datasets[0] if hasattr(validset, 'datasets') else validset
    indices = [start + t for demo, start in episodes._demo_id_to_start_indices.items()
               for t in range(0, episodes._demo_id_to_demo_length[demo]-1, stride)]
    if max_windows:
        indices = indices[:max_windows]
    loader = DataLoader(Subset(validset, indices), batch_size=1, shuffle=False, num_workers=0)
    rng = torch.Generator(device='cpu').manual_seed(2026)
    model = rollout.policy; model.inference_noise_generator = rng; model.set_eval()
    positions=[]; angles=[]; accuracy=[]; transitions=[]; trajectories=[]
    order = list(expected.train.action_keys)
    def unnormalize(a):
        chunks=[]; start=0
        for key in order:
            scale=np.asarray(stats[key]['scale']); offset=np.asarray(stats[key]['offset']); dim=scale.shape[-1]
            chunks.append(a[..., start:start+dim]*scale+offset);start+=dim
        return np.concatenate(chunks,axis=-1)
    # Resolve native flat validation indices back to episode timestamps.
    episodes = validset.datasets[0] if hasattr(validset, 'datasets') else validset
    with h5py.File(dataset, 'r') as datafile, torch.no_grad():
        for index, batch in zip(indices, loader):
            processed = model.process_batch_for_training(batch)
            processed = model.postprocess_batch_for_training(processed, obs_normalization_stats=None)
            pred = unnormalize(model._get_action_trajectory(processed['obs'])[0].cpu().numpy())
            demo = episodes._index_to_demo_id[index]
            t = index - episodes._demo_id_to_start_indices[demo]
            ep = datafile['data'][demo]
            future = np.arange(t + 1, min(t + 9, len(ep['actions'])))
            pred = pred[:len(future)]
            # Ground truth stays raw: validation clipping must not change metrics.
            truth = ep['actions'][t:t + len(future)]
            raw_obs=batch['obs']
            anchor=measured_poses({key:raw_obs[key][0,1].numpy() for key in ('right_ee_pos','right_ee_quat')})
            ppose=decode_chunk(experiment,pred,anchor); tpose=measured_poses(ep['obs'])[future]
            pe=np.linalg.norm(ppose[:,:3,3]-tpose[:,:3,3],axis=-1)
            re=np.rad2deg(Rotation.from_matrix(ppose[:,:3,:3] @ tpose[:,:3,:3].swapaxes(-1,-2)).magnitude())
            initial=bool(raw_obs['right_gripper_width'][0,1,0] < WIDTH_THRESHOLD)
            gripper=GripperHysteresis(closed=initial)
            pg=(pred[:,-1] < WIDTH_THRESHOLD) if experiment=='baseline_7d' else np.array([gripper.decode(value)==0 for value in pred[:,-1]])
            tg=(truth[:,-1] < WIDTH_THRESHOLD) if experiment=='baseline_7d' else (truth[:,-1]>=0)
            positions.append(np.pad(pe,(0,8-len(pe)),constant_values=np.nan))
            angles.append(np.pad(re,(0,8-len(re)),constant_values=np.nan))
            accuracy.append(np.pad((pg==tg).astype(float),(0,8-len(pg)),constant_values=np.nan))
            pt=np.flatnonzero(np.diff(np.r_[initial,pg]));tt=np.flatnonzero(np.diff(np.r_[initial,tg]))
            times = ep['timestamps'][:]
            slots = times[future] - times[t]
            # Actual recorded durations, not an assumed .1 s per action.
            transitions.append(dict(validation_index=index,predicted_slots=pt.tolist(),demonstrated_slots=tt.tolist(),
                                    first_transition_error_slots=int(pt[0]-tt[0]) if len(pt) and len(tt) else None,
                                    first_transition_error_s=float(slots[pt[0]]-slots[tt[0]]) if len(pt) and len(tt) else None))
            trajectories.append(dict(validation_index=index,demo=demo,anchor_index=t,time_offsets_s=slots.tolist(),predicted_xyz=ppose[:,:3,3].tolist(),
                                     demonstrated_xyz=tpose[:,:3,3].tolist(),translation_error_m=pe.tolist(),orientation_error_deg=re.tolist()))
    def by_slot(values):
        columns = np.asarray(values).T
        return [float(np.mean(col[np.isfinite(col)])) if np.isfinite(col).any() else None
                for col in columns]
    report=dict(experiment=experiment,checkpoint=str(checkpoint),dataset_sha256=sha256(dataset),windows=len(indices),
                mode='teacher-forced observation anchors; open-loop 8-action chunks; NOT closed-loop/task success',
                translation_mean_m=float(np.nanmean(positions)),orientation_mean_deg=float(np.nanmean(angles)),
                gripper_accuracy=float(np.nanmean(accuracy)),translation_by_slot_m=by_slot(positions),
                orientation_by_slot_deg=by_slot(angles),transitions=transitions,trajectories=trajectories)
    with open(output,'x') as stream: json.dump(report,stream,indent=2,allow_nan=False)
    return {k:v for k,v in report.items() if k not in ('transitions','trajectories')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','validate','train','evaluate-offline'):
        p=sub.add_parser(command);p.add_argument('--experiment',choices=EXPERIMENTS,required=True)
        p.add_argument('--source' if command=='prepare' else '--dataset',type=Path,required=True)
        if command in ('prepare','train','evaluate-offline'):p.add_argument('--output',type=Path,required=True)
        if command=='train':p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
        if command=='evaluate-offline':
            p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--stride',type=int,default=8)
            p.add_argument('--max-windows',type=int)
    args=vars(parser.parse_args());command=args.pop('command')
    fn={'prepare':prepare,'validate':validate,'train':train,'evaluate-offline':evaluate_offline}[command]
    result=fn(**args)
    if result is not None:print(json.dumps(result,indent=2))


if __name__=='__main__':main()
