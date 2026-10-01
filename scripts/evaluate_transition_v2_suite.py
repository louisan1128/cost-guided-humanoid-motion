#!/usr/bin/env python3
"""Sequential physical evaluation of six transitions and five source motions.

Runs Isaac Lab subprocesses; no training. Outputs are kept per model/run.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
TRANSITIONS=('start_walk','walk_left','left_walk','walk_right','right_walk','walk_stop')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out-dir',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--skip-originals',action='store_true')
    a=p.parse_args();model=a.model.resolve();out=a.out_dir.resolve()
    if not model.is_file():p.error(f'Missing {model}')
    out.mkdir(parents=True,exist_ok=False)
    summaries=[]
    for tag in TRANSITIONS:
        cfg=ROOT/'configs/envs'/f'transition_inertial_{tag}_v2_eval.yaml'
        reference=Path(yaml.safe_load(cfg.read_text())['motion_file']).with_suffix('.json')
        cmd=[sys.executable,str(ROOT/'scripts/trace_transition_tracker.py'),
             '--env-config',str(cfg),'--model-file',str(model),'--reference-report',str(reference),
             '--output-prefix',str(out/tag),'--seed',str(a.seed)]
        with (out/f'{tag}.log').open('w') as stream:
            subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        data=json.loads((out/f'{tag}.json').read_text());summaries.append(data)
        print(tag,data['termination'],data['episode_end_s'],data['last_root_tilt_deg'],flush=True)
    if not a.skip_originals:
        source=yaml.safe_load((ROOT/'configs/datasets/humanoid_five_motion_v1.yaml').read_text())
        base=yaml.safe_load((ROOT/'configs/envs/transition_inertial_start_walk_v2_eval.yaml').read_text())
        for item in source['motions']:
            stem=Path(item['file']).stem
            cfg=dict(base);cfg['motion_file']=item['file']
            config=out/f'original_{stem}.yaml';config.write_text(yaml.safe_dump(cfg,sort_keys=False))
            cmd=[sys.executable,str(ROOT/'scripts/evaluate_mixamo_tracker.py'),
                 '--env-config',str(config),'--model-file',str(model),'--episodes','1','--seed',str(a.seed)]
            logfile=out/f'original_{stem}.log'
            with logfile.open('w') as stream:
                subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            result=[line.removeprefix('TRACKER_EVAL ') for line in logfile.read_text().splitlines() if line.startswith('TRACKER_EVAL ')][-1]
            data=json.loads(result);(out/f'original_{stem}.json').write_text(json.dumps(data,indent=2)+'\n')
            print('original',stem,'success',data['success'],'fail',data['fail'],flush=True)
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')

if __name__=='__main__':main()
