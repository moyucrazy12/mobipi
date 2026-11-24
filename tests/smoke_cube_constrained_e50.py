"""Explicit CPU smoke check: one optimizer step, never the training entry point.

Uses the full production architecture and image sizes, a smoke batch of one,
checks the configured batch-eight loader, and writes only a temporary checkpoint.
"""
import argparse
import json
import math
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from robomimic.algo import algo_factory
from robomimic.utils import file_utils, train_utils
from rby1_mobipi.policies.cube_in_cup_experiments import (
    configuration, load_training_data, sha256,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    dataset_hash = sha256(args.dataset)
    with tempfile.TemporaryDirectory(prefix='cube-e50-smoke-') as temporary:
        cfg = configuration('relative_10d_constrained_e50', args.dataset, Path(temporary)/'unused')
        assert cfg.experiment.env_meta_update_dict.cube_required_dataset_sha256 == dataset_hash
        cfg.experiment.env_meta_update_dict.cube_dataset_sha256 = dataset_hash
        train, valid = load_training_data(cfg)
        assert len(train) == 11002 and len(valid) == 1426
        assert train.get_dataset_sampler() is None and valid.get_dataset_sampler() is None
        stats = train.get_action_normalization_stats()
        valid.set_action_normalization_stats(stats)
        loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=0, drop_last=cfg.train.drop_last)
        valid_loader = DataLoader(valid, batch_size=cfg.train.batch_size, shuffle=False,
                                  num_workers=0, drop_last=cfg.train.drop_last)
        assert len(loader) == 1376 and len(valid_loader) == 179
        assert math.ceil(len(train)/8)*50 == 68800
        batch = next(iter(loader))
        assert batch['actions'].shape == (8, 16, 10)
        assert np.isfinite(batch['actions'].numpy()).all()
        batch = {key: ({k: v[:1] for k, v in value.items()}
                       if isinstance(value, dict) else value[:1])
                 for key, value in batch.items()}
        keys = cfg.observation.modalities.obs.low_dim + cfg.observation.modalities.obs.rgb
        shapes = file_utils.get_shape_metadata_from_dataset(str(args.dataset),
                    action_keys=cfg.train.action_keys, all_obs_keys=keys, verbose=False)
        assert shapes['ac_dim'] == 10
        model = algo_factory('diffusion_policy', cfg, obs_key_shapes=shapes['all_shapes'],
                             ac_dim=10, device=torch.device('cpu'))
        assert isinstance(model.optimizers['policy'], torch.optim.Adam)
        assert model.optimizers['policy'].param_groups[0]['lr'] == 1e-4
        model.set_train()
        processed = model.process_batch_for_training(batch)
        processed = model.postprocess_batch_for_training(processed, obs_normalization_stats=None)
        parameter = next(model.nets.parameters())
        before = parameter.detach().clone()
        info = model.train_on_batch(processed, epoch=1, validate=False)
        assert not torch.equal(before, parameter)
        assert all(torch.isfinite(v).all() for v in info['losses'].values())
        checkpoint = Path(temporary)/'cpu_smoke.pth'
        train_utils.save_model(model, cfg, file_utils.get_env_metadata_from_dataset(str(args.dataset)),
                               shapes, str(checkpoint), action_normalization_stats=stats)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        saved_config = json.loads(saved['config'])
        assert saved_config['train']['num_epochs'] == 50
        assert saved_config['experiment']['save']['every_n_epochs'] == 5
        assert saved_config['experiment']['save']['on_best_validation']
        assert saved_config['experiment']['ckpt_path'] is None
        assert set(saved['action_normalization_stats']) == {'actions_eef_cmd', 'actions_gripper'}
        for key in stats:
            for field in ('scale', 'offset'):
                np.testing.assert_array_equal(saved['action_normalization_stats'][key][field], stats[key][field])
        with h5py.File(args.dataset, 'r') as f:
            assert len(f['data']) == 31
        assert sha256(args.dataset) == dataset_hash
        print('PASS: one CPU forward/backward Adam step, native checkpoint/config, unchanged dataset.')
        print('PASS: 11002 train windows / 1376 batches; 1426 validation windows / 179 batches; 68800 updates for 50 epochs.')


if __name__ == '__main__':
    main()
