#!/usr/bin/env python3
"""Pilot: frozen motion generators -> one shared root-motion-conditioned action decoder."""
import argparse,copy,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_state_action_sequence as phys
import run_short_action_sequence as core
import run_anchored_action_sequence as anchored
from pilot_direct_action_bc import sha,write
from humanoid_imitation.planning.future_conditioned_action import FutureConditionedAction,root_features,fixed_travel_observation
from humanoid_imitation.planning.direct_action import DirectMLP
from humanoid_imitation.planning.state_action_sequence import yaw,local_state
BASE=ROOT/'outputs/future_conditioned_action_v6'
PRIOR=ROOT/'outputs/direct_action_study_v12/final/mlp/seed0/model.pt'
SOURCE=ROOT/'outputs/turn_branch_v4/teacher'


def build(a):
 d=torch.load(ROOT/'outputs/state_action_sequence_v2/dataset_h30/data.pt',weights_only=False)
 out={s:dict(base=dict(obs=d[s]['context'][:,432:576].clone(),feature=root_features(d[s]['target'][:,:,28:]),target=d[s]['target'][:,0,:28].clone(),source_id=d[s]['episode_id'].clone())) for s in ['train','validation']};parts={s:[] for s in out}
 for path in sorted(SOURCE.glob('branch_*.pt')):
  e=torch.load(path,weights_only=False)
  if not e['valid']:continue
  starts=torch.arange(61);future=starts[:,None]+torch.arange(30);origin=e['pre']['root_pos'][starts,None];angle=yaw(e['pre']['root_rot'][starts])[:,None]
  state=local_state({k:v[future] for k,v in e['next'].items()},origin,angle)
  initial=e['pre']['root_pos'][0];heading=yaw(e['pre']['root_rot'][0]);goal=initial[:2]+5*torch.stack([heading.cos(),heading.sin()])
  if ((e['pre']['root_pos'][:,:2]-goal).norm(dim=-1)<.6).any():raise ValueError('Counterfactual goal enters arrival regime')
  obs=fixed_travel_observation(e['obs'][starts],e['pre']['root_pos'][starts],yaw(e['pre']['root_rot'][starts]),goal)
  if not torch.equal(obs[:,:140],e['obs'][starts,:140]):raise ValueError('Physical obs changed')
  parts[e['case']['split']].append(dict(obs=obs,feature=root_features(state),target=e['applied_action'][starts],source_id=torch.full((61,),e['case']['source_id'],dtype=torch.long)))
 for s,ps in parts.items():out[s]['branch']={k:torch.cat([p[k] for p in ps]) for k in ps[0]}
 features=torch.cat([out['train'][g]['feature'] for g in ['base','branch']]);prior=torch.load(PRIOR,map_location='cpu',weights_only=False)
 out.update(feature_mean=features.mean(0),feature_std=features.std(0).clamp_min(.01),stats=prior['stats'],prior_path=str(PRIOR),prior_sha256=sha(PRIOR))
 torch.save(out,a.out_dir/'data.pt');write(a.out_dir/'result.json',dict(windows={s:{g:len(out[s][g]['obs']) for g in ['base','branch']} for s in ['train','validation']},feature_dim=32,normalization='frozen teacher obs/action; rootfeature train-only',source_split_preserved=True,counterfactual_goal_m=5,data_sha256=sha(a.out_dir/'data.pt')))


def make(c,device='cuda'):
 if sha(c['prior_path'])!=c['prior_sha256']:raise ValueError('Prior changed')
 p=torch.load(c['prior_path'],map_location='cpu',weights_only=False);prior=DirectMLP();prior.load_state_dict(p['model'])
 return FutureConditionedAction(prior,c['stats']['action_std'],c['feature_mean'],c['feature_std'],include_goal=c.get('include_goal',False)).to(device)


def load(path):
 c=torch.load(path,map_location='cpu',weights_only=False);m=make(c);m.load_state_dict(c['model']);m.eval();s={k:v.cuda() for k,v in c['stats'].items()};return m,s


def train(a):
 torch.set_num_threads(4);torch.manual_seed(0);tic=time.monotonic();d=torch.load(BASE/'dataset/data.pt',weights_only=False);s={k:v.cuda() for k,v in d['stats'].items()}
 def normalized(split,group):
  r=d[split][group];return dict(obs=((r['obs'].cuda()-s['obs_mean'])/s['obs_std']).clamp(-10,10),feature=r['feature'].cuda(),target=(r['target'].cuda()-s['action_mean'])/s['action_std'])
 tr={g:normalized('train',g) for g in ['base','branch']};val={g:normalized('validation',g) for g in tr};rng=torch.Generator(device='cuda').manual_seed(4601)
 for g,v in val.items():
  ids=torch.randperm(len(v['obs']),device='cuda',generator=rng)[:1024];val[g]={k:x[ids] for k,x in v.items()}
 c={k:d[k] for k in ['feature_mean','feature_std','stats','prior_path','prior_sha256']};c['include_goal']=a.include_goal;m=make(c);ema=copy.deepcopy(m).eval();opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=3e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,4000,eta_min=3e-5);best=float('inf');curve=[]
 for step in range(1,4001):
  groups=[]
  for g,v in tr.items():
   ix=torch.randint(len(v['obs']),(256,),device='cuda');groups.append({k:x[ix] for k,x in v.items()})
  batch={k:torch.cat([g[k] for g in groups]) for k in groups[0]};pred=m(batch['obs'],batch['feature'],noise=.03);loss=(pred-batch['target']).square().mean()
  if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
  opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();sched.step()
  with torch.no_grad():
   for e,p in zip(ema.parameters(),m.parameters()):e.lerp_(p,.005)
  if step%500==0:
   with torch.no_grad():scores={g:float((ema(v['obs'],v['feature'])-v['target']).square().mean()) for g,v in val.items()};score=sum(scores.values())/2
   row=dict(update=step,train_mse=float(loss),validation=scores,balanced_validation_mse=score);curve.append(row);print(row,flush=True)
   if score<best:
    best=score;torch.save(dict(c,model={k:v.cpu().clone() for k,v in ema.state_dict().items()},update=step,role='shared future-root-conditioned action decoder; frozen MLP prior; no new PPO'),a.out_dir/'model.pt')
 p=torch.load(PRIOR,map_location='cpu',weights_only=False);ck=torch.load(a.out_dir/'model.pt',weights_only=False);assert all(torch.equal(ck['model']['prior.'+k],v) for k,v in p['model'].items())
 write(a.out_dir/'curve.json',curve);write(a.out_dir/'result.json',dict(status='complete',updates=4000,seed=0,batch=512,branch_fraction=.5,train_windows={g:len(v['obs']) for g,v in tr.items()},best_balanced_validation_mse=best,wall_s=time.monotonic()-tic,prior_tensor_exact=True,model_sha256=sha(a.out_dir/'model.pt'),data_sha256=sha(BASE/'dataset/data.pt'),runner_sha256=sha(__file__)))


def branch(a):
 selected=[]
 for path in sorted(SOURCE.glob('branch_*.pt')):
  e=torch.load(path,weights_only=False)
  if e['case']['split']=='validation':selected.append(e)
 if len(selected)!=36:raise ValueError('Expected validation12x3')
 cases=[e['case'] for e in selected];saved={k:torch.stack([e['pre'][k][0] for e in selected]) for k in phys.STATE_METHODS};saved['standing']=torch.zeros(36,dtype=torch.bool)
 a.root=ROOT/'outputs/state_action_sequence_v1';a.seed=4504;phys.check(a);tic=time.monotonic();env,agent=phys.init_env(a,cases,saved);cid=env._get_char_id();decoder,s=load(a.decoder_path);generator=None
 if a.mode=='generated':
  core.make_model=anchored.make_anchored;generator=core.ShortActionPolicy(ROOT/f'outputs/short_action_sequence_v3/repair2/models/{a.method}/model.pt',a.seed)
  generator.hist=[torch.stack([e['prefix_obs'][j] for e in selected]).cuda() for j in range(3)];generator.actions=[torch.stack([e['prefix_action'][j] for e in selected]).cuda() for j in range(3)]
 chunks=[]
 with torch.no_grad():
  env.reset();initial=phys.state(env);errors={k:float((initial[k].cpu()-saved[k]).abs().max()) for k in phys.STATE_METHODS}
  if max(errors.values())>2e-5:raise ValueError('Reset mismatch')
  heading=yaw(initial['root_rot']);goal=initial['root_pos'][:,:2]+5*torch.stack([heading.cos(),heading.sin()],-1)
  offsets=torch.tensor([math.radians(c['branch_deg']) for c in cases],device='cuda');midangle=heading+offsets;midpoint=initial['root_pos'][:,:2]+4*torch.stack([midangle.cos(),midangle.sin()],-1)
  env.goal_xy[:]=goal;env.goal_yaw[:]=heading;env._refresh_changed_goal();env._update_observations(None);obs=env._obs_buf
  for step in range(60):
   pre=phys.state(env);current=obs.clone();x=((obs-s['obs_mean'])/s['obs_std']).clamp(-10,10)
   if a.mode=='generated':
    planning_obs=fixed_travel_observation(obs,pre['root_pos'],yaw(pre['root_rot']),midpoint)
    generator(planning_obs);feature=root_features(generator.last[:,:,28:]);plan=generator.last[:,:,28:]
   else:
    future={k:torch.stack([e['next'][k][step:step+30] for e in selected]).cuda() for k in phys.STATE_METHODS};plan=local_state(future,pre['root_pos'][:,None],yaw(pre['root_rot'])[:,None]);feature=root_features(plan)
   y=decoder.prior(x) if a.mode=='prior' else decoder(x,feature);requested=y*s['action_std']+s['action_mean'];action=requested.clamp(env._action_bound_low,env._action_bound_high)
   if generator:generator.actions[-1]=action.clone()
   obs,_,done,info=env.step(action);post=phys.state(env)
   chunks.append(dict(obs=current.cpu(),applied_action=action.cpu(),requested_action=requested.cpu(),done=done.cpu(),feature=feature.cpu(),pre={k:v.cpu() for k,v in pre.items()},next={k:v.cpu() for k,v in post.items()}))
 rows=[]
 for i,c in enumerate(cases):
  episode={k:torch.stack([d[k][i] for d in chunks]) for k in ['obs','applied_action','requested_action','done','feature']}
  for group in ['pre','next']:episode[group]={k:torch.stack([d[group][k][i] for d in chunks]) for k in phys.STATE_METHODS}
  delta=yaw(episode['next']['root_rot'][-1])-yaw(initial['root_rot'][i].cpu());valid=not (episode['done']!=0).any().item();row=dict(c,valid=valid,yaw_delta_deg=float(torch.atan2(delta.sin(),delta.cos()).rad2deg()));episode.update(case=c,goal=dict(xy=goal[i].cpu().tolist(),yaw=float(heading[i])),dt=1/30)
  torch.save(episode,a.out_dir/f'branch_{i:03d}.pt');rows.append(row)
 pairs=[dict(source_id=cases[i]['source_id'],valid=all(x['valid'] for x in rows[i:i+3]),lr_yaw_deg=rows[i+1]['yaw_delta_deg']-rows[i+2]['yaw_delta_deg']) for i in range(0,36,3)];passed=sum(x['valid'] and x['lr_yaw_deg']>=10 for x in pairs)>=10 and all(x['valid'] for x in rows)
 write(a.out_dir/'result.json',dict(status='complete',mode=a.mode,method=a.method,rows=rows,pairs=pairs,gate_pass=passed,reset_errors=errors,wall_s=time.monotonic()-tic,final_goal_fixed=True,oracle_future=a.mode in ['oracle','prior'],future_condition='teacher world future (diagnostic)' if not generator else 'frozen generator receives explicit intermediate target; common decoder receives unchanged final task',shared_decoder_sha256=sha(a.decoder_path),original_joint_actions_executed=False,runner_sha256=sha(__file__)))
 print('BRANCH COMPLETE',a.mode,a.method,passed,flush=True)


def nominal(a):
 decoder,s=load(a.decoder_path);core.make_model=anchored.make_anchored
 class Decoded(core.ShortActionPolicy):
  def __call__(self,obs):
   super().__call__(obs);x=((obs-s['obs_mean'])/s['obs_std']).clamp(-10,10);action=decoder(x,root_features(self.last[:,:,28:]))*s['action_std']+s['action_mean'];self.actions[-1]=action.detach().clone();return action
 phys.SequencePolicy=Decoded;a.root=ROOT/'outputs/state_action_sequence_v1';a.source=a.root/'collection_certified';a.policy=ROOT/f'outputs/short_action_sequence_v3/repair2/models/{a.method}/model.pt';a.seed=4414;a.cohort='validation';a.limit=0;a.guide=0.;a.guide_target=0.;a.unit_gradient=False;a.execute_steps=1;a.guide_mode='heading';a.guide_start_seconds=2.;a.mode='evaluate';a.out_dir.rmdir()
 with torch.no_grad():phys.simulate(a)
 result=json.loads((a.out_dir/'result.json').read_text());result.update(pipeline='frozen joint model state output -> shared future-root action decoder -> PD; joint action predictions not executed',shared_decoder_sha256=sha(a.decoder_path));write(a.out_dir/'result.json',result)


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['build','train','oracle','prior','generated','nominal']);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--method',choices=['mlp','diffusion'],default='mlp');p.add_argument('--decoder-path',type=Path,default=BASE/'decoder/model.pt');p.add_argument('--include-goal',action='store_true');a=p.parse_args();a.out_dir=a.out_dir.resolve();a.out_dir.mkdir(parents=True,exist_ok=False)
 if a.mode=='build':build(a)
 elif a.mode=='train':train(a)
 elif a.mode=='nominal':nominal(a)
 else:branch(a)
if __name__=='__main__':main()
