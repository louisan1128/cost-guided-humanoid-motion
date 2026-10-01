#!/usr/bin/env python3
"""View one selected transition reference with a specified physical PPO tracker."""
import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
CHOICES=('start_walk','walk_left','left_walk','walk_right','right_walk','walk_stop','walk_stop_hold5s')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('transition',choices=CHOICES)
    p.add_argument('--model',type=Path,default=None,
                   help='Defaults to the selected v2 checkpoint, or the 500-iteration stop-hold checkpoint for walk_stop_hold5s')
    p.add_argument('--mimickit-root',type=Path,default=Path('/home/an/Desktop/MimicKit-main'))
    p.add_argument('--episodes',type=int,default=1000000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.transition=='walk_stop_hold5s':
        cfg=ROOT/'configs/envs/transition_inertial_walk_stop_hold5s_probe.yaml'
        default_model=ROOT/'outputs/ppo_transition_stop_hold5s_500_seed0/model.pt'
    else:
        cfg=ROOT/'configs/envs'/f'transition_inertial_{a.transition}_v2_eval.yaml'
        default_model=ROOT/'outputs/ppo_transition_inertial_v2_add200_seed0/int_models/model_0000000050.pt'
    model=(a.model or default_model).resolve();mimic=a.mimickit_root.resolve()
    for path in (cfg,model,mimic/'mimickit/run.py'):
        if not path.is_file():p.error(f'File does not exist: {path}')
    if a.episodes<1:p.error('--episodes must be positive')
    cmd=[sys.executable,str(mimic/'mimickit/run.py'),'--arg_file','args/deepmimic_humanoid_ppo_args.txt',
         '--mode','test','--num_envs','1','--visualize','true','--test_episodes',str(a.episodes),
         '--rand_seed',str(a.seed),'--env_config',str(cfg),'--model_file',str(model)]
    if a.dry_run:
        import shlex
        print('cwd:',mimic);print(shlex.join(cmd));return
    with tempfile.TemporaryDirectory(prefix='transition_view_') as out:
        subprocess.run(cmd+['--out_dir',out],cwd=mimic,check=True)

if __name__=='__main__':main()
