"""Explicit CPU-only integration check: one optimizer step per variant, never a run.

python tests/smoke_cube_experiments.py --dataset-dir /private/datasets --output /tmp/check
"""
import argparse
import gc
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from robomimic.algo import algo_factory
from robomimic.utils import file_utils, train_utils
from rby1_mobipi.policies.cube_in_cup_experiments import (
    configuration, load_training_data, evaluate_offline, sha256,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    results=[]
    for name in ('binary_7d','rot6d_10d','relative_10d'):
        torch.manual_seed(2026)
        path=args.dataset_dir/(name+'.hdf5')
        cfg=configuration(name,path,args.output/name)
        cfg.experiment.env_meta_update_dict.cube_dataset_sha256=sha256(path)
        train,valid=load_training_data(cfg)
        stats=train.get_action_normalization_stats();valid.set_action_normalization_stats(stats)
        loader=DataLoader(train,batch_size=8,shuffle=False,num_workers=0,drop_last=False)
        assert len(loader)==2587 and len(train)==20693 and len(valid)==2187
        batch=next(iter(loader))
        assert batch['actions'].shape==(8,16,7 if name=='binary_7d' else 10)
        # Reduce only smoke batch count, never architecture or image resolution.
        batch={k:({ok:ov[:1] for ok,ov in v.items()} if isinstance(v,dict) else v[:1]) for k,v in batch.items()}
        keys=cfg.observation.modalities.obs.low_dim+cfg.observation.modalities.obs.rgb
        shape=file_utils.get_shape_metadata_from_dataset(str(path),action_keys=cfg.train.action_keys,all_obs_keys=keys,verbose=False)
        model=algo_factory('diffusion_policy',cfg,obs_key_shapes=shape['all_shapes'],ac_dim=shape['ac_dim'],device=torch.device('cpu'))
        model.set_train()
        processed=model.process_batch_for_training(batch)
        processed=model.postprocess_batch_for_training(processed,obs_normalization_stats=None)
        parameter=next(model.nets.parameters());before=parameter.detach().clone()
        info=model.train_on_batch(processed,epoch=1,validate=False)
        assert not torch.equal(before,parameter)
        assert all(torch.isfinite(v).all() for v in info['losses'].values())
        ckpt=args.output/(name+'.pth')
        train_utils.save_model(model,cfg,file_utils.get_env_metadata_from_dataset(str(path)),shape,str(ckpt),action_normalization_stats=stats)
        del model,batch,processed,loader,train,valid,parameter,before,info;gc.collect()
        report=evaluate_offline(name,path,ckpt,args.output/(name+'.json'),max_windows=1)
        # Matching dimension does not make C and D checkpoints interchangeable.
        wrong='relative_10d' if name=='rot6d_10d' else 'rot6d_10d'
        try:
            evaluate_offline(wrong,args.dataset_dir/(wrong+'.hdf5'),ckpt,args.output/'must-not-exist.json',max_windows=1)
        except ValueError:pass
        else:raise AssertionError('Incompatible checkpoint accepted')
        results.append(report);print('PASSED',name,flush=True);gc.collect()
    (args.output/'summary.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':main()
