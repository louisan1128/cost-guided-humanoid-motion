#!/usr/bin/env python3
"""Supervised candidate decoder rehearsal on train teacher routes and normal replay."""
import argparse,copy,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from run_future_conditioned_action_v6 import make,BASE as ORIGINAL
from probe_consistent_root_guidance_v8 import DECODER
from pilot_direct_action_bc import sha,write
from humanoid_imitation.planning.future_conditioned_action import root_features,fixed_travel_observation
from humanoid_imitation.planning.state_action_sequence import yaw,local_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir=a.out_dir.resolve();a.out_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(0);tic=time.monotonic();original=torch.load(ORIGINAL/'dataset/data.pt',weights_only=False);parts={s:[]for s in ['train','validation']}
    for path in sorted((ROOT/'outputs/obstacle_repair_v10/detours_balanced/teacher').glob('episode_*.pt')):
        e=torch.load(path,weights_only=False)
        if not e['row']['usable']:continue
        starts=torch.linspace(0,len(e['obs'])-30,min(300,len(e['obs'])-29)).round().long().unique();obs=e['obs'].clone();pos=e['pre']['root_pos'][:60];h=yaw(e['pre']['root_rot'][:60]);goal=pos.new_tensor(e['goal']['xy']);distance=(pos[:,:2]-goal).norm(dim=-1)
        if (distance<.6).any():raise ValueError('Counterfactual goal enters arrival during collection command')
        obs[:60]=fixed_travel_observation(obs[:60],pos,h,goal);future=starts[:,None]+torch.arange(30);state=local_state({k:v[future]for k,v in e['next'].items()},e['pre']['root_pos'][starts,None],yaw(e['pre']['root_rot'][starts])[:,None]);parts[e['row']['split']].append(dict(obs=obs[starts],feature=root_features(state),target=e['applied_action'][starts]))
    c=torch.load(DECODER,map_location='cpu',weights_only=False);m=make(c);m.load_state_dict(c['model']);ema=copy.deepcopy(m).eval();s={k:v.cuda()for k,v in c['stats'].items()}
    raw={split:{'base':original[split]['base'],'branch':original[split]['branch'],'new':{k:torch.cat([e[k]for e in es])for k in es[0]}}for split,es in parts.items()}
    def normalize(row):return dict(obs=((row['obs'].cuda()-s['obs_mean'])/s['obs_std']).clamp(-10,10),feature=row['feature'].cuda(),target=(row['target'].cuda()-s['action_mean'])/s['action_std'])
    train={g:normalize(row)for g,row in raw['train'].items()};val={g:normalize(row)for g,row in raw['validation'].items()};gen=torch.Generator(device='cuda').manual_seed(5064)
    for g,row in val.items():
        ix=torch.randperm(len(row['obs']),device='cuda',generator=gen)[:1024];val[g]={k:v[ix]for k,v in row.items()}
    opt=torch.optim.Adam([v for v in m.parameters()if v.requires_grad],lr=1e-4);schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,eta_min=1e-5);best=float('inf');curve=[]
    for step in range(4001):
        if step:
            batches=[]
            for group,count in [('base',256),('branch',128),('new',128)]:
                row=train[group];ix=torch.randint(len(row['obs']),(count,),device='cuda');batches.append({k:v[ix]for k,v in row.items()})
            b={k:torch.cat([v[k]for v in batches])for k in batches[0]};pred=m(b['obs'],b['feature'],noise=.03);loss=(pred-b['target']).square().mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();schedule.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),m.parameters()):ep.lerp_(mp,.005)
        if step%500==0:
            with torch.no_grad():scores={g:float((ema(row['obs'],row['feature'])-row['target']).square().mean())for g,row in val.items()};score=sum(scores.values())/3
            curve.append(dict(update=step,validation=scores,score=score));print(curve[-1],flush=True)
            if score<best:
                best=score;torch.save(dict(c,model={k:v.cpu().clone()for k,v in ema.state_dict().items()},update=step,parent_path=str(DECODER),parent_sha256=sha(DECODER),role='candidate shared decoder; normal + actual teacher detour rehearsal'),a.out_dir/'model.pt')
    ck=torch.load(a.out_dir/'model.pt',weights_only=False);exact=all(torch.equal(v,ck['model'][k])for k,v in c['model'].items()if k.startswith('prior.'))
    if not exact:raise ValueError('Prior changed')
    write(a.out_dir/'curve.json',curve);write(a.out_dir/'result.json',dict(status='complete',updates=4000,seed=0,batch=512,mixture='base50% originalbranch25% newteacher25%',new_windows={k:len(v['new']['obs'])for k,v in raw.items()},prior_tensor_exact=exact,parent_sha256=sha(DECODER),model_sha256=sha(a.out_dir/'model.pt'),optimizer_resume=False,wall_s=time.monotonic()-tic,runner_sha256=sha(__file__)))
if __name__=='__main__':main()
