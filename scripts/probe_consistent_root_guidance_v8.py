#!/usr/bin/env python3
"""Frozen V6 decoder, physically consistent root cost, train-only scale choice."""
import argparse,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_short_action_sequence as core
import run_anchored_action_sequence as anchored
from run_future_conditioned_action_v6 import load
from pilot_direct_action_bc import write,sha
from humanoid_imitation.planning.future_conditioned_action import root_features
from humanoid_imitation.planning.consistent_root_guidance import curved_root_target,root_path_cost
from humanoid_imitation.planning.state_action_sequence import sample_sequence,yaw
BASE=ROOT/'outputs/consistent_root_guidance_v8'
MODEL=ROOT/'outputs/short_action_sequence_v3/repair2/models/diffusion/model.pt'
DECODER=ROOT/'outputs/future_conditioned_action_v6/repair/decoder/model.pt'


def physical(z,policy):
    return (z*policy.stats['target_std']+policy.stats['target_mean']).reshape(-1,30,99)


def sample(policy,context,noise,anchor,desired,scale):
    with torch.no_grad():nominal=physical(sample_sequence(policy.model,context,policy.alpha,noise,16),policy)
    target=curved_root_target(nominal[:,:,28:],anchor,desired)
    def cost(z):return root_path_cost(physical(z,policy)[:,:,28:],target)
    with torch.no_grad():
        z=sample_sequence(policy.model,context,policy.alpha,noise,16,cost,scale,True)
        result=physical(z,policy)
    return result,nominal,target


def audit(a):
    torch.set_num_threads(4);tic=time.monotonic();core.make_model=anchored.make_anchored;p=core.ShortActionPolicy(MODEL,4824);decoder,s=load(DECODER)
    data=torch.load(ROOT/'outputs/state_action_sequence_v2/dataset_h30/data.pt',weights_only=False)
    rng=torch.Generator().manual_seed(4824);ids=torch.randperm(len(data['train']['context']),generator=rng)[:32];raw=data['train']['context'][ids].cuda()
    context=((raw-p.stats['context_mean'])/p.stats['context_std']).clamp(-10,10)
    noise=torch.randn(32,2970,device='cuda',generator=p.rng);anchor=torch.zeros(32,device='cuda')
    with torch.no_grad():nominal=physical(sample_sequence(p.model,context,p.alpha,noise),p);base_dir=torch.atan2(nominal[:,-1,29],nominal[:,-1,28]);base_feature=root_features(nominal[:,:,28:]);obs=((raw[:,432:576]-s['obs_mean'])/s['obs_std']).clamp(-10,10);base_action=decoder(obs,base_feature)*s['action_std']+s['action_mean']
    rows=[]
    for scale in [.3,1.,3.,10.]:
        results=[]
        for sign in [1,-1]:
            result,nom,target=sample(p,context,noise,anchor,base_dir+sign*math.radians(15),scale)
            with torch.no_grad():
                f=root_features(result[:,:,28:]);action=decoder(obs,f)*s['action_std']+s['action_mean'];before=root_path_cost(nom[:,:,28:],target);after=root_path_cost(result[:,:,28:],target)
            results.append(dict(result=result,shift=float(((f-base_feature)/decoder.feature_std).square().mean().sqrt()),action_rms=float((action-base_action).square().mean().sqrt()),reduction=float((before-after).mean()),relative_reduction=float(((before-after)/before.clamp_min(1e-6)).mean())))
        l,r=[x['result'] for x in results];heading=torch.atan2(l[:,-1,32],l[:,-1,31])-torch.atan2(r[:,-1,32],r[:,-1,31]);heading=torch.atan2(heading.sin(),heading.cos()).rad2deg();lateral=l[:,-1,29]-r[:,-1,29]
        row=dict(scale=scale,mean_lr_heading_deg=float(heading.mean()),mean_lr_lateral_m=float(lateral.mean()),pairs_heading_ge3=int((heading>=3).sum()),pairs_lateral_ge005=int((lateral>=.05).sum()),feature_shift_rms=max(x['shift'] for x in results),mean_relative_cost_reduction=sum(x['relative_reduction'] for x in results)/2,mean_decoder_action_change_rad=sum(x['action_rms'] for x in results)/2)
        row['eligible']=row['feature_shift_rms']<=1 and row['mean_relative_cost_reduction']>0 and all(math.isfinite(v) for v in row.values());rows.append(row);print(row,flush=True)
    eligible=[x for x in rows if x['eligible']];selected=max(eligible,key=lambda x:x['mean_relative_cost_reduction']) if eligible else None
    write(a.out_dir/'result.json',dict(status='complete',selection='maximum mean relative cost reduction among preregistered scales with feature shift RMS <=1; train only',rows=rows,selected_scale=None if selected is None else selected['scale'],train_window_indices=ids.tolist(),source_ids=data['train']['episode_id'][ids].tolist(),wall_s=time.monotonic()-tic,model_sha256=sha(MODEL),decoder_sha256=sha(DECODER)))


class GuidedDecoded(core.ShortActionPolicy):
    def __init__(self,*args,env,scale,offset,**kwargs):
        super().__init__(*args,**kwargs);self.env=env;self.scale=scale;self.offset=offset;self.desired=None;self.onset=None;self.decoder,self.ds=load(DECODER)
    def __call__(self,obs):
        self.hist=(self.hist+[obs.detach().clone()])[-4:];history=[self.hist[0]]*(4-len(self.hist))+self.hist;past=[torch.zeros(len(obs),28,device=obs.device)]*(3-len(self.actions))+self.actions
        context=torch.cat([*history,*past],-1);context=((context-self.stats['context_mean'])/self.stats['context_std']).clamp(-10,10);noise=torch.randn(len(obs),2970,device=obs.device,generator=self.rng)
        anchor=yaw(self.env._engine.get_root_rot(self.env._get_char_id())).detach()
        with torch.no_grad():nom=physical(sample_sequence(self.model,context,self.alpha,noise),self)
        if self.guide and self.offset and self.queries<120:
            if self.desired is None:
                self.desired=(anchor+torch.atan2(nom[:,-1,29],nom[:,-1,28])+math.radians(self.offset)).detach();self.onset=dict(anchor_yaw=anchor.cpu().tolist(),desired_world_direction=self.desired.cpu().tolist())
            self.last,_,_=sample(self,context,noise,anchor,self.desired,self.scale)
        else:self.last=nom
        with torch.no_grad():
            x=((obs-self.ds['obs_mean'])/self.ds['obs_std']).clamp(-10,10);action=self.decoder(x,root_features(self.last[:,:,28:]))*self.ds['action_std']+self.ds['action_mean']
        self.actions=(self.actions+[action.detach().clone()])[-3:];self.queries+=1;self.plan_age=0
        return action


def probe(a):
    selected=json.loads((BASE/'audit/result.json').read_text())['selected_scale']
    if selected is None:raise ValueError('Offline calibration failed')
    a.root=ROOT/'outputs/state_action_sequence_v1';a.source=a.root/'collection_certified';a.policy=MODEL;a.mode='evaluate' if a.full else 'probe';a.cohort='validation';a.limit=12;a.seed=4424;a.probe_steps=120;a.execute_steps=1;a.guide_mode='heading';a.guide=1. if a.offset else 0.;a.guide_target=a.offset;a.unit_gradient=True;a.guide_start_seconds=2.
    core.make_model=anchored.make_anchored;holder={};original=core.physics.init_env
    def init(*args,**kwargs):
        env,agent=original(*args,**kwargs);holder['env']=env;return env,agent
    def policy(*args,**kwargs):
        p=GuidedDecoded(*args,env=holder['env'],scale=selected,offset=a.offset,**kwargs);holder['policy']=p;return p
    core.physics.init_env=init;core.physics.SequencePolicy=policy
    with torch.no_grad():core.physics.simulate(a)
    r=json.loads((a.out_dir/'result.json').read_text());r.update(shared_decoder_sha256=sha(DECODER),selected_scale=selected,midpoint_observation_override=False,guidance_onset=holder['policy'].onset,root_cost_source_sha256=sha(ROOT/'src/humanoid_imitation/planning/consistent_root_guidance.py'),runner_sha256=sha(__file__));write(a.out_dir/'result.json',r)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','probe']);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--offset',type=float,choices=[0,15,-15],default=0);p.add_argument('--full',action='store_true');a=p.parse_args();a.out_dir=a.out_dir.resolve()
    if a.mode=='audit':a.out_dir.mkdir(parents=True,exist_ok=False);audit(a)
    else:probe(a)
if __name__=='__main__':main()
