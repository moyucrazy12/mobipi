"""Contracts for portable achieved-motion CubeInCup comparisons."""
import json
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from rby1_mobipi.policies.cube_actions import (
    binary_gripper, GripperHysteresis, decode_chunk, rotation6d_from_euler,
)
from rby1_mobipi.policies.cube_in_cup_experiments import (
    OBS_KEYS, prepare, validate, configuration, load_training_data, sha256, evaluate_offline, main, train as launch_train,
)
from robomimic.utils.rby1_relative_cartesian import tcp_pose, encode_relative_target


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name); self.source=self.root/'a.hdf5'
        with h5py.File(self.source,'w') as f:
            data=f.create_group('data');data.attrs['env_args']=json.dumps(dict(env_name='CubeInCup',env_kwargs={},type=None))
            for j in range(3):
                ep=data.create_group(f'demo_{j}'); n=5;ep.attrs['num_samples']=n
                p=np.array([[.1*i,.02*i*i,.03*i] for i in range(n)])
                r=Rotation.from_euler('xyz',np.array([[.1*i,-.05*i,.2*i] for i in range(n)]))
                q=r.as_quat()[:,[3,0,1,2]]; width=np.array([.08,.04,.03,0,.08])
                obs=ep.create_group('obs')
                values=dict(right_ee_pos=p,right_ee_quat=q,right_gripper_width=width[:,None],
                            right_joint_pos=np.zeros((n,7)),right_joint_vel=np.zeros((n,7)),
                            head_rgb=np.zeros((n,32,32,3),np.uint8),right_wrist_rgb=np.zeros((n,32,32,3),np.uint8))
                for k in OBS_KEYS:obs[k]=values[k]
                future=np.minimum(np.arange(n)+1,n-1)
                a=np.c_[p[future]-p,(r[future]*r.inv()).as_euler('xyz'),width[future]].astype(np.float32)
                ep['actions']=a;ep['timestamps']=np.array([0,.08,.2,.31,.48]);ep['dones']=np.array([0,0,0,0,1])
            f['mask/train']=np.array(['demo_0','demo_1'],dtype='S');f['mask/valid']=np.array(['demo_2'],dtype='S')

    def test_constrained_fifty_epoch_profile_reuses_relative_representation(self):
        path=self.root/'relative.hdf5'
        prepare('relative_10d',self.source,path)
        cfg=configuration('relative_10d_constrained_e50',path,self.root/'fresh')
        old=configuration('relative_10d',path,self.root/'old')
        self.assertEqual(cfg.observation,old.observation)
        self.assertEqual(cfg.train.action_config,old.train.action_config)
        self.assertEqual(cfg.algo.relative_actions,old.algo.relative_actions)
        self.assertEqual(cfg.algo.unet,old.algo.unet)
        self.assertEqual(cfg.algo.horizon,old.algo.horizon)
        self.assertEqual(cfg.train.num_epochs,50)
        self.assertEqual(cfg.train.batch_size,8)
        self.assertEqual(cfg.algo.optim_params.policy.optimizer_type,'adam')
        self.assertEqual(cfg.algo.optim_params.policy.learning_rate.initial,1e-4)
        self.assertEqual(cfg.algo.optim_params.policy.learning_rate.scheduler_type,'constant')
        self.assertFalse(cfg.train.drop_last)
        self.assertIsNone(cfg.experiment.epoch_every_n_steps)
        self.assertIsNone(cfg.experiment.validation_epoch_every_n_steps)
        self.assertIsNone(cfg.experiment.ckpt_path)
        self.assertTrue(cfg.experiment.validate)
        self.assertTrue(cfg.experiment.logging.log_tb)
        self.assertEqual(cfg.experiment.save.every_n_epochs,5)
        self.assertTrue(cfg.experiment.save.on_best_validation)
        validate('relative_10d_constrained_e50',path)
        dataset,_=load_training_data(cfg)
        self.assertEqual(dataset[0]['actions'].shape,(16,10))
        with patch('sys.argv',['cube','train','--experiment','relative_10d_constrained_e50',
                              '--dataset',str(path),'--output',str(self.root/'fresh')]), \
             patch('rby1_mobipi.policies.cube_in_cup_experiments.train',return_value=None) as call:
            main()
            self.assertEqual(call.call_args.kwargs['experiment'],'relative_10d_constrained_e50')

    def test_constrained_profile_rejects_wrong_dataset_before_output_creation(self):
        path=self.root/'relative.hdf5'
        prepare('relative_10d',self.source,path)
        with self.assertRaisesRegex(ValueError,'constrained-layout dataset checksum'):
            launch_train('relative_10d_constrained_e50',path,self.root/'must-not-exist','cpu')
        self.assertFalse((self.root/'must-not-exist').exists())

    def test_all_conversions_and_native_windows(self):
        before=self.source.read_bytes()
        for name in ('binary_7d','rot6d_10d','relative_10d'):
            path=self.root/(name+'.hdf5'); prepare(name,self.source,path)
            report=validate(name,path,source=self.source)
            duplicate=self.root/(name+'_again.hdf5'); prepare(name,self.source,duplicate)
            self.assertEqual(sha256(path),sha256(duplicate))
            self.assertLess(report['max_position_error_m'],1e-6)
            cfg=configuration(name,path,self.root/name)
            self.assertIsNone(cfg.experiment.epoch_every_n_steps)
            self.assertFalse(cfg.train.drop_last)
            train,valid=load_training_data(cfg);stats=train.get_action_normalization_stats()
            valid.set_action_normalization_stats(stats)
            self.assertEqual((len(train),len(valid)),(10,5))
            item=train[0];self.assertEqual(item['actions'].shape,(16,7 if name=='binary_7d' else 10))
            self.assertTrue(np.isfinite(item['actions']).all())
            if name=='relative_10d':
                # First row padding shares anchor zero, including target slot zero.
                with h5py.File(path,'r') as f:
                    pose=tcp_pose(f['data/demo_0/obs/right_ee_pos'][0],f['data/demo_0/obs/right_ee_quat'][0],'wxyz')
                    from robomimic.utils.rby1_relative_cartesian import pose9_to_transform
                    targets=pose9_to_transform(f['data/demo_0/actions_eef_cmd'][:])
                    expected=encode_relative_target(pose,targets[np.clip(np.arange(-1,15),0,4)])
                    st=stats['actions_eef_cmd'];actual=item['actions'][:,:9]*st['scale']+st['offset']
                    np.testing.assert_allclose(actual,expected,atol=1e-6)
            with self.assertRaises(FileExistsError):prepare(name,self.source,path)
        self.assertEqual(before,self.source.read_bytes())

    def test_offline_oracle_uses_correct_future_pose_and_gripper_rows(self):
        class Oracle:
            def __init__(self, dim): self.ac_dim=dim
            def set_eval(self): pass
            def process_batch_for_training(self, batch):
                self.batch=batch
                return batch
            def postprocess_batch_for_training(self, batch, obs_normalization_stats): return batch
            def _get_action_trajectory(self, obs): return self.batch['actions'][:,1:9]
        from robomimic.utils import tensor_utils
        for name in ('binary_7d','rot6d_10d','relative_10d','relative_10d_constrained_e50'):
            path=self.root/(name+'.hdf5');prepare(name,self.source,path)
            cfg=configuration(name,path,self.root/name)
            cfg.experiment.env_meta_update_dict.cube_dataset_sha256=sha256(path)
            train,_=load_training_data(cfg);stats=train.get_action_normalization_stats()
            ckpt=dict(config=cfg.dump(),action_normalization_stats=tensor_utils.to_list(stats))
            oracle=Oracle(7 if name=='binary_7d' else 10)
            with patch('robomimic.utils.file_utils.policy_from_checkpoint',return_value=(SimpleNamespace(policy=oracle),ckpt)):
                report=evaluate_offline(name,path,'unused.pth',self.root/(name+'.json'),max_windows=1)
            self.assertLess(report['translation_mean_m'],1e-6)
            self.assertLess(report['orientation_mean_deg'],1e-4)
            self.assertEqual(report['gripper_accuracy'],1.)
            detail=json.loads((self.root/(name+'.json')).read_text())
            self.assertEqual(detail['transitions'][0]['first_transition_error_slots'],0)
            self.assertEqual(len(detail['trajectories'][0]['predicted_xyz']),4)

    def test_spatial_rotation_composition(self):
        anchor=tcp_pose([1,2,3],Rotation.from_euler('xyz',[.5,.3,-.8]).as_quat())
        euler=np.array([[.2,-.3,.4],[.1,.2,-.1]])
        a7=np.c_[np.ones((2,3))*.01,euler,np.ones(2)]
        a10=np.c_[a7[:,:3],rotation6d_from_euler(euler),a7[:,-1]]
        np.testing.assert_allclose(decode_chunk('binary_7d',a7,anchor),decode_chunk('rot6d_10d',a10,anchor),atol=1e-12)
        with self.assertRaises(ValueError):decode_chunk('rot6d_10d',np.zeros((1,10)),anchor)

    def test_binary_threshold_and_hysteresis(self):
        np.testing.assert_array_equal(binary_gripper([0,.039,.04,.08]),[1,1,-1,-1])
        decoder=GripperHysteresis()
        self.assertEqual([decoder.decode(v) for v in [.1,.3,0,-.1,-.3]],[.08,0,0,0,.08])

    def test_cli_help_and_invalid_experiment(self):
        for command in ('prepare','validate','train','evaluate-offline'):
            with patch('sys.argv',['cube',command,'--help']), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as result:main()
                self.assertEqual(result.exception.code,0)
        with patch('sys.argv',['cube','validate','--experiment','wrong','--dataset','unused']), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:main()
            self.assertEqual(result.exception.code,2)

    def test_split_leakage_is_rejected(self):
        with h5py.File(self.source,'r+') as f:
            del f['mask/valid'];f['mask/valid']=np.array(['demo_0'],dtype='S')
        with self.assertRaisesRegex(ValueError,'leakage'):validate('baseline_7d',self.source)


if __name__=='__main__':unittest.main()
