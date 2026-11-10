#!/usr/bin/env python3
"""Bounded S0-S3 sequence study. Teacher labels and applied actions are distinct."""
import argparse,copy,json,math,os,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from pilot_direct_action_bc import STATE_METHODS,TEACHER,sha,write
from humanoid_imitation.planning.state_action_sequence import (SequenceNet,build_windows,local_state,yaw,alpha_schedule,noisy_action,sample_sequence,PlanExecutionBuffer,heading_guidance_cost)


def prepare(a):
    a.root.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(4404);cases=[]
    for i in range(208):
        split='train'if i<128 else('validation'if i<160 else'test')
        local=i if i<128 else(i-128 if i<160 else i-160)
        cases.append(dict(id=i,split=split,distance_m=float(rng.uniform(3,7)),bearing_deg=[0,45,90,135,180,-135,-90,-45][local%8]+float(rng.uniform(-12,12)),yaw_offset_deg=[0,90,-90,180][(local//8)%4]))
    frozen=[TEACHER,ROOT/'configs/direct_action/evaluated_v12.json',ROOT/'configs/releases/final_results_20261021.json',ROOT/'configs/goal_teacher/view_sequential_arrival_v34.yaml',ROOT/'configs/goal_teacher/agent_sequential_v31.yaml',ROOT/'configs/engines/isaac_lab.yaml',ROOT/'src/humanoid_imitation/envs/goal_amp_env.py']
    # Also freeze every checkpoint explicitly selected in both manifests.
    manifest=json.loads((ROOT/'configs/direct_action/evaluated_v12.json').read_text())
    frozen.extend(Path(row['path'])for row in manifest['models'].values())
    old=json.loads((ROOT/'configs/releases/final_results_20261021.json').read_text())
    def walk(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if isinstance(x,str) and (ROOT/x).is_file() and (x.endswith('.pt')or x.endswith('.yaml')):frozen.append(ROOT/x)
                else:walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
    walk(old)
    protocol=dict(date='2026-11-04',status='preregistered',cases=cases,seconds=45,control_hz=30,seed=4404,history=4,
        frozen={str(p.resolve()):sha(p)for p in frozen},
        scoring=dict(position_m=1.25,yaw_deg=45,speed_mps=.15,wz_radps=.2,hold_s=5,drift_m=.1,post_s=2),
        gates=dict(S0_replay_xy_max_m=.02,S0_replay_q_rmse_rad=.1,S1_success_drop=2,S1_fall_increase=1,S2_success_drop=2,S2_fall_increase=1,S2_xy_p95_m=.15,S2_yaw_p95_deg=15,S3_pairs=12,S3_correct_pairs=10,S3_min_actual_y_delta_m=.01),
        training=dict(updates=8000,batch_size=512,lr=.0003,ema=.995,seed=0,selection='minimum validation sampled first-action normalized MSE every500 updates; full sequence errors also reported'),
        planned_arms=['mlp_h1_action','diffusion_h1_action','mlp_h8_action','diffusion_h8_action','mlp_h8_joint','diffusion_h8_joint'],
        budget='two cause-based repairs per failed gate maximum; S3 required before S4; no new test access before model selection',
        state_order=['root_xyz','rotation6d_columns','root_linear_velocity','root_angular_velocity','q28','dq28'],action='physical PD joint targets,28',joint_token='[applied_action_t, local_state_t+1]',evaluation_assistance=False)
    write(a.root/'protocol.json',protocol)
    for source in [Path(__file__),ROOT/'src/humanoid_imitation/planning/state_action_sequence.py']:
        (a.root/source.name).write_bytes(source.read_bytes())
    print('Prepared',a.root)


def check(a):
    p=json.loads((a.root/'protocol.json').read_text())
    for path,h in p['frozen'].items():
        if sha(path)!=h:raise ValueError('Frozen input changed: '+path)
    return p


def state(env):
    cid=env._get_char_id()
    return {k:getattr(env._engine,'get_'+m)(cid).detach().clone()for k,m in STATE_METHODS.items()}


def extra_state(env):
    cid=env._get_char_id()
    return dict(body_pos=env._engine.get_body_pos(cid).detach().clone(),contact_force=env._engine.get_contact_forces(cid).detach().clone(),
                arrival_align=env.arrival_align.clone(),arrival_turn_sign=env.arrival_turn_sign.clone())


def init_env(a,cases,saved=None):
    sys.path.insert(0,'/home/an/Desktop/MimicKit-main/mimickit')
    from humanoid_imitation.envs.goal_amp_env import GoalAMPEnv
    from util import mp_util,util
    from learning import agent_builder,base_agent
    cfg=yaml.safe_load((ROOT/'configs/goal_teacher/view_sequential_arrival_v34.yaml').read_text())
    motions=yaml.safe_load((ROOT/cfg['motion_file']).read_text())
    for r in motions['motions']:r['file']=str((ROOT/r['file']).resolve())
    (a.out_dir/'motions.yaml').write_text(yaml.safe_dump(motions));cfg['motion_file']=str(a.out_dir/'motions.yaml');cfg['episode_length']=46
    for k in ('interactive_goal','fixed_test_goal','measured_standing_reset','stand_alignment_rehearsal'):cfg['goal'].pop(k,None)
    if cfg['enable_tar_obs']or cfg['enable_phase_obs']:raise ValueError('Reference input leakage')
    (a.out_dir/'env.yaml').write_text(yaml.safe_dump(cfg))
    class Fixed(GoalAMPEnv):
        def _ref_state_init(self,ids):
            super()._ref_state_init(ids)
            if saved is not None:
                for k,m in STATE_METHODS.items():getattr(self._engine,'set_'+m)(ids,self._get_char_id(),saved[k].cuda()[ids])
    os.chdir('/home/an/Desktop/MimicKit-main');mp_util.init(0,1,'cuda:0',6368);util.set_rand_seed(a.seed);torch.set_num_threads(4)
    env=Fixed(env_config=cfg,engine_config=yaml.safe_load((ROOT/'configs/engines/isaac_lab.yaml').read_text()),num_envs=len(cases),device='cuda:0',visualize=False)
    agent=agent_builder.build_agent(str(ROOT/'configs/goal_teacher/agent_sequential_v31.yaml'),env,'cuda:0');agent.load(str(TEACHER));agent.eval();agent.set_mode(base_agent.AgentMode.TEST)
    return env,agent


class SequencePolicy:
    def __init__(self,path,seed=0,guide=0.,guide_target=0.,unit_gradient=False,execute_steps=1,guide_mode="lateral"):
        c=torch.load(path,map_location='cuda',weights_only=False)
        self.c=c;self.stats={k:v.cuda()for k,v in c['stats'].items()};self.h=c['horizon'];self.joint=c['joint'];self.token=c['token_dim'];self.hist=[];self.actions=[]
        self.model=SequenceNet(c['context_dim'],self.h*self.token,c['method']).cuda().eval();self.model.load_state_dict(c['model'])
        self.alpha=alpha_schedule().cuda();self.rng=torch.Generator(device='cuda').manual_seed(seed)
        self.guide=guide;self.guide_target=guide_target;self.last=None;self.unit_gradient=unit_gradient
        self.execution=PlanExecutionBuffer(execute_steps);self.guide_mode=guide_mode;self.plan_age=0;self.queries=0
    def __call__(self,obs):
        self.hist.append(obs.detach().clone());self.hist=self.hist[-4:]
        if not self.execution.needs_plan:
            self.plan_age=self.execution.age
            action=self.execution.take();self.actions=(self.actions+[action.detach().clone()])[-3:]
            return action
        h=[self.hist[0]]*(4-len(self.hist))+self.hist
        past=[torch.zeros(len(obs),28,device=obs.device)]*(3-len(self.actions))+self.actions
        context=torch.cat([*h,*past],-1)
        x=((context-self.stats['context_mean'])/self.stats['context_std']).clamp(-10,10)
        noise=torch.randn(len(x),self.h*self.token,device='cuda',generator=self.rng)
        nominal_endpoint=None
        if self.guide and self.guide_mode=="heading":
            if not self.joint:raise ValueError("Heading guidance needs joint state/action output")
            with torch.no_grad():
                nominal=sample_sequence(self.model,x,self.alpha,noise,16)
                nominal_endpoint=((nominal*self.stats["target_std"]+self.stats["target_mean"]).reshape(len(x),self.h,self.token)[:,-1]).detach()
        def cost(z):
            physical=(z*self.stats['target_std']+self.stats['target_mean']).reshape(len(z),self.h,self.token)
            if self.guide_mode=="heading":return heading_guidance_cost(physical,nominal_endpoint,math.radians(self.guide_target))
            # Local root y is token action28 + state y1.
            return (physical[:,-1,29]-self.guide_target).square()
        with torch.no_grad():
            y=sample_sequence(self.model,x,self.alpha,noise,16,cost if self.guide else None,self.guide,self.unit_gradient)
            seq=(y*self.stats['target_std']+self.stats['target_mean']).reshape(len(x),self.h,self.token)
        self.last=seq.detach();self.execution.set(self.last);self.plan_age=0;self.queries+=1
        action=self.execution.take();self.actions=(self.actions+[action.detach().clone()])[-3:]
        return action


def simulate(a):
    protocol=check(a);a.out_dir.mkdir(parents=True,exist_ok=False)
    collect=a.mode=='collect';replay=a.mode=='replay';probe=a.mode=='probe';openloop=a.mode=='openloop'
    if collect:
        cases=protocol['cases'][:a.limit]if a.limit else protocol['cases'][:160]
        saved=None;source=None
    else:
        source=a.source or a.root/'collection'
        index=json.loads((source/'result.json').read_text());original={c['id']:c for c in protocol['cases']}
        cases=[original[r['id']]for r in index['rows']if r['split']==a.cohort]
        if a.limit:cases=cases[:a.limit]
        allsaved=torch.load(source/'initial.pt',weights_only=False);lookup={r['id']:i for i,r in enumerate(index['rows'])};ix=[lookup[c['id']]for c in cases]
        saved={k:v[ix]for k,v in allsaved.items()}
    start=time.monotonic();env,agent=init_env(a,cases,saved)
    from humanoid_imitation.planning.arrival_success import ArrivalHoldMonitor
    from envs.base_env import DoneFlags
    n=len(cases);dt=env._engine.get_timestep();cid=env._get_char_id()
    torch.save(dict(low=env._action_bound_low.cpu(),high=env._action_bound_high.cpu()),a.out_dir/'action_bounds.pt')
    policy=SequencePolicy(a.policy,a.seed,a.guide,a.guide_target,a.unit_gradient,a.execute_steps,a.guide_mode)if a.policy else None
    replayeps=[torch.load(source/f"episode_{c['id']:03d}.pt",weights_only=False)for c in cases]if replay else None
    initial_errors={};chunks=[];alive=torch.ones(n,dtype=torch.bool,device='cuda');rows=[None]*n;timings=[];predictions=[];discrepancies=[];plan_ages=[]
    monitors=[ArrivalHoldMonitor(position_m=1.25,yaw_deg=45,speed_mps=.15,yaw_rate_radps=.2,hold_s=5,drift_m=.1)for c in cases]
    maxsteps=8 if replay else(policy.h if openloop else(a.probe_steps if probe else round(protocol['seconds']/dt)))
    sequence_errors=[];open_sequence=None
    with torch.no_grad():
        obs,info=env.reset()
        if saved is not None:env._last_reset_standing[:]=saved['standing'].cuda()
        initial={k:v.cpu()for k,v in state(env).items()};initial['standing']=env._last_reset_standing.cpu().clone();torch.save(initial,a.out_dir/'initial.pt')
        if saved is not None:
            initial_errors={k:float((initial[k]-saved[k]).abs().max())for k in STATE_METHODS}
            if max(initial_errors.values())>2e-5:raise ValueError('Reset mismatch')
            goals=json.loads((source/'goals.json').read_text());goals=[goals[i]for i in ix]
        else:
            p=env._engine.get_root_pos(cid);heading=yaw(env._engine.get_root_rot(cid))
            b=heading+torch.tensor([math.radians(c['bearing_deg'])for c in cases],device='cuda');d=torch.tensor([c['distance_m']for c in cases],device='cuda')
            xy=p[:,:2]+d[:,None]*torch.stack((b.cos(),b.sin()),-1);gy=b+torch.tensor([math.radians(c['yaw_offset_deg'])for c in cases],device='cuda')
            goals=[dict(xy=xy[i].cpu().tolist(),yaw=float(gy[i]))for i in range(n)]
        env.goal_xy[:]=torch.tensor([g['xy']for g in goals],device='cuda');env.goal_yaw[:]=torch.tensor([g['yaw']for g in goals],device='cuda')
        env._refresh_changed_goal();env._update_observations(None);obs,info=env._obs_buf,env._info;write(a.out_dir/'goals.json',goals)
        for step in range(maxsteps):
            pre=state(env);ex=extra_state(env);teacher,_=agent._decide_action(obs,info)
            if replay:action=torch.stack([e['applied_action'][step]for e in replayeps]).cuda()
            elif policy is not None:
                if openloop and open_sequence is not None:action=open_sequence[:,step,:28]
                else:
                    policy.guide=a.guide if step*dt>=a.guide_start_seconds else 0.
                    torch.cuda.synchronize();tic=time.perf_counter();action=policy(obs);torch.cuda.synchronize();timings.append((time.perf_counter()-tic)*1000)
                    if openloop:open_sequence=policy.last.clone()
                predictions.append(policy.last.cpu());plan_ages.append(policy.plan_age)
            else:action=teacher.clone()
            if not torch.isfinite(action).all():raise ValueError('Nonfinite action')
            requested_action=action.clone()
            action=action.clamp(env._action_bound_low,env._action_bound_high)
            if policy is not None and not openloop:policy.actions[-1]=action.detach().clone()
            before_obs=obs.clone();obs,_,done,info=env.step(action);post=state(env)
            chunks.append(dict(obs=before_obs.cpu(),teacher_action=teacher.cpu(),requested_action=requested_action.cpu(),applied_action=action.cpu(),pre={k:v.cpu()for k,v in pre.items()},next={k:v.cpu()for k,v in post.items()},extra={k:v.cpu()for k,v in ex.items()},done=done.cpu(),active=alive.cpu().clone()))
            if openloop and policy.joint:
                actual=local_state(post,initial['root_pos'].cuda(),yaw(initial['root_rot'].cuda()))
                predicted=open_sequence[:,step,28:]
                de=torch.atan2(torch.sin(torch.atan2(predicted[:,4],predicted[:,3])-torch.atan2(actual[:,4],actual[:,3])),torch.cos(torch.atan2(predicted[:,4],predicted[:,3])-torch.atan2(actual[:,4],actual[:,3])))
                sequence_errors.append(dict(step=step,xy_m=(predicted[:,:2]-actual[:,:2]).norm(dim=-1).cpu().tolist(),yaw_deg=de.abs().rad2deg().cpu().tolist(),q_rmse_rad=(predicted[:,15:43]-actual[:,15:43]).square().mean(-1).sqrt().cpu().tolist()))
            if replay:
                expected={k:torch.stack([e['next'][k][step]for e in replayeps]).cuda()for k in STATE_METHODS}
                discrepancies.append(dict(step=step,xy_max_m=float((post['root_pos'][:,:2]-expected['root_pos'][:,:2]).norm(dim=-1).max()),q_rmse_rad=float((post['q']-expected['q']).square().mean().sqrt())))
            pos=post['root_pos'];heading=yaw(post['root_rot']);t=(step+1)*dt
            for i,c in enumerate(cases):
                if not alive[i]:continue
                d=float((pos[i,:2]-env.goal_xy[i]).norm());err=float(torch.atan2(torch.sin(heading[i]-env.goal_yaw[i]),torch.cos(heading[i]-env.goal_yaw[i])));speed=float(post['root_vel'][i,:2].norm());wz=float(post['root_ang_vel'][i,2]);flag=int(done[i])
                monitors[i].update(t,d,err,speed,wz,pos[i,:2].cpu().numpy());first=monitors[i].first_success_time_s;fail=flag==DoneFlags.FAIL.value
                if fail or(first is not None and t>=first+2-1e-8)or flag!=DoneFlags.NULL.value or step==maxsteps-1:
                    alive[i]=False;rows[i]=dict(**c,success=bool(first is not None and not fail),fall=fail,steps=step+1,duration_s=t,position_error_m=d,yaw_error_deg=math.degrees(abs(err)),speed_mps=speed,wz_radps=wz)
            if not alive.any():break
    stats=dict(obs_mean=agent._obs_norm.get_mean().cpu(),obs_std=agent._obs_norm.get_std().cpu(),action_mean=agent._a_norm.get_mean().cpu(),action_std=agent._a_norm.get_std().cpu())
    for i,c in enumerate(cases):
        count=rows[i]['steps'];episode={}
        for key in ['obs','teacher_action','requested_action','applied_action','done']:
            episode[key]=torch.stack([x[key][i]for x in chunks[:count]])
        for key in ['pre','next','extra']:
            episode[key]={k:torch.stack([x[key][k][i]for x in chunks[:count]])for k in chunks[0][key]}
        episode.update(case=c,goal=goals[i],dt=dt,success=rows[i]['success'],action_source='teacher'if policy is None else'student',terminal_observation_auto_reset=False)
        if predictions:
            episode['predicted_sequence']=torch.stack([x[i]for x in predictions[:count]])
            episode['prediction_plan_age']=torch.tensor(plan_ages[:count])
        torch.save(episode,a.out_dir/f"episode_{c['id']:03d}.pt")
    torch.save(stats,a.out_dir/'teacher_stats.pt')
    finite=all(torch.isfinite(x[k]).all().item()for x in chunks for k in ['obs','teacher_action','applied_action'])
    finite=finite and all(torch.isfinite(v).all().item()for x in chunks for key in ['pre','next']for v in x[key].values())
    continuity=max((float((chunks[j]['next'][k]-chunks[j+1]['pre'][k]).abs().max())for j in range(len(chunks)-1)for k in STATE_METHODS),default=0.)
    audit=dict(finite=finite,state_continuity_max=continuity,applied_equals_label=all(torch.equal(x['teacher_action'],x['applied_action'])for x in chunks),clipped_action_elements=sum(int((x['requested_action']!=x['applied_action']).sum())for x in chunks),transitions=sum(r['steps']for r in rows))
    if not finite or continuity>1e-6:raise ValueError('Transition contract failed')
    if replay:
        audit.update(max_xy_m=max(d['xy_max_m']for d in discrepancies),q_rmse_rad=max(d['q_rmse_rad']for d in discrepancies))
        audit['gate_pass']=audit['max_xy_m']<=protocol['gates']['S0_replay_xy_max_m']and audit['q_rmse_rad']<=protocol['gates']['S0_replay_q_rmse_rad']
        write(a.out_dir/'replay_errors.json',discrepancies)
    if sequence_errors:
        write(a.out_dir/'sequence_errors.json',sequence_errors)
        audit.update(xy_p95_m=float(np.quantile(sequence_errors[-1]['xy_m'],.95)),yaw_p95_deg=float(np.quantile(sequence_errors[-1]['yaw_deg'],.95)))
    write(a.out_dir/'result.json',dict(status='complete',mode=a.mode,seed=a.seed,rows=rows,successes=sum(r['success']for r in rows),falls=sum(r['fall']for r in rows),wall_s=time.monotonic()-start,reset_errors=initial_errors,audit=audit,dt=dt,policy=str(a.policy)if a.policy else None,policy_sha256=sha(a.policy)if a.policy else None,teacher_sha256=sha(TEACHER),reference_tracker=False,teacher_action_in_evaluation=False,guide=a.guide,guide_target=a.guide_target,unit_gradient=a.unit_gradient,guide_mode=a.guide_mode,guide_start_seconds=a.guide_start_seconds,execute_steps=a.execute_steps,generator_queries=policy.queries if policy else 0,latency_ms=None if not timings else dict(mean=float(np.mean(timings)),p95=float(np.quantile(timings,.95)),batch=n),script_sha256=sha(__file__)))
    print(json.dumps(dict(status='complete',episodes=n,successes=sum(r['success']for r in rows),audit=audit)),flush=True)


def dataset(a):
    protocol=check(a);source=a.source or a.root/'collection';items={'train':[],'validation':[]};counts={};start=time.monotonic()
    for path in sorted(source.glob('episode_*.pt')):
        e=torch.load(path,weights_only=False)
        split=e['case']['split']
        if split not in items:continue
        if not e['success']:continue
        if e['action_source']!='teacher':raise ValueError('Only physical teacher sequence data admitted in initial study')
        bounds=torch.load(source/'action_bounds.pt',weights_only=False)
        if not torch.equal(e['applied_action'],e['teacher_action'].clamp(bounds['low'],bounds['high'])):raise ValueError('Mislabelled applied action')
        built=build_windows(e,a.horizon,4,a.joint)
        if built is None:continue
        x,y,ids=built;items[split].append((x,y,e['case']['id']));counts[e['case']['id']]=len(x)
    a.out_dir.mkdir(parents=True,exist_ok=False)
    out={}
    for split,parts in items.items():
        if not parts:raise ValueError('Empty '+split)
        out[split]=dict(context=torch.cat([p[0]for p in parts]),target=torch.cat([p[1]for p in parts]),episode_id=torch.cat([torch.full((len(p[0]),),p[2],dtype=torch.long)for p in parts]))
    train=out['train'];teacher=torch.load(source/'teacher_stats.pt',weights_only=False)
    # Teacher action normalization for all past/action target channels; state stats train-only.
    cm=torch.cat([teacher['obs_mean']]*4+[teacher['action_mean']]*3);cs=torch.cat([teacher['obs_std']]*4+[teacher['action_std']]*3).clamp_min(1e-4)
    tm=train['target'].mean(0);ts=train['target'].std(0).clamp_min(.01)
    tm[:,:28]=teacher['action_mean'];ts[:,:28]=teacher['action_std'].clamp_min(1e-4)
    out.update(stats=dict(context_mean=cm,context_std=cs,target_mean=tm.flatten(),target_std=ts.flatten()),horizon=a.horizon,joint=a.joint,history=4)
    torch.save(out,a.out_dir/'data.pt');write(a.out_dir/'result.json',dict(status='complete',horizon=a.horizon,joint=a.joint,windows={k:len(out[k]['context'])for k in items},episodes=counts,source=str(source),data_sha256=sha(a.out_dir/'data.pt'),wall_s=time.monotonic()-start))
    print('Built',a.out_dir,flush=True)


def train(a):
    check(a);a.out_dir.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.set_num_threads(4);torch.manual_seed(a.seed)
    d=torch.load(a.data,weights_only=False);stats={k:v.cuda()for k,v in d['stats'].items()};h=d['horizon'];token=d['train']['target'].shape[-1]
    def norm(split):
        x=d[split]['context'].cuda();y=d[split]['target'].flatten(1).cuda()
        return ((x-stats['context_mean'])/stats['context_std']).clamp(-10,10),(y-stats['target_mean'])/stats['target_std']
    x,y=norm('train');vx,vy=norm('validation');g=torch.Generator(device='cuda').manual_seed(4405);ids=torch.randperm(len(vx),generator=g,device='cuda')[:2048];vx,vy=vx[ids],vy[ids]
    model=SequenceNet(x.shape[-1],y.shape[-1],a.method).cuda()
    ema=copy.deepcopy(model).eval();opt=torch.optim.Adam(model.parameters(),lr=3e-4);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.updates,eta_min=3e-5);alpha=alpha_schedule().cuda();noise=torch.randn(vy.shape,generator=g,device='cuda');best=float('inf');curve=[]
    for step in range(1,a.updates+1):
        ix=torch.randint(len(x),(512,),device='cuda');bx,by=x[ix],y[ix]
        if a.method=='mlp':pred=model(bx)
        else:
            t=torch.randint(100,(len(bx),),device='cuda');t[:len(t)//2]=torch.randint(75,100,(len(t)//2,),device='cuda');pred=model(bx,noisy_action(by,torch.randn_like(by),t,alpha),t)
        loss=(pred-by).square().reshape(len(bx),h,token)
        # Equal normalized action/state group weight independent of dimension count.
        weights=torch.ones(h,device='cuda');weights[0]=a.first_weight
        action_loss=(loss[:,:,:28].mean(-1)*weights[None,:]).sum(-1).mean()/weights.sum()
        state_loss=loss[:,:,28:].mean()if d['joint']else loss.new_zeros(())
        loss=action_loss+a.state_weight*state_loss
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();sched.step()
        with torch.no_grad():
            for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,.005)
        if step%500==0 or step==a.updates:
            with torch.no_grad():
                preds=torch.cat([sample_sequence(ema,vx[k:k+256],alpha,noise[k:k+256])for k in range(0,len(vx),256)])
                err=(preds-vy).reshape(len(vy),h,token);score=float(err[:,0,:28].square().mean());physical=err*stats['target_std'].reshape(h,token)
                row=dict(update=step,training_loss=float(loss),action_loss=float(action_loss),state_loss=float(state_loss),validation_first_action_mse=score,validation_action_rmse_rad=float(physical[:,:,:28].square().mean().sqrt()),validation_sequence_mse=float(err.square().mean()))
            curve.append(row);print(json.dumps(row),flush=True)
            if score<best:
                best=score
                torch.save(dict(model={k:v.cpu().clone()for k,v in ema.state_dict().items()},stats=d['stats'],method=a.method,horizon=h,joint=d['joint'],token_dim=token,context_dim=x.shape[-1],seed=a.seed,update=step,history=4,execute=1,teacher_sha256=sha(TEACHER)),a.out_dir/'model.pt')
    write(a.out_dir/'learning_curve.json',curve);write(a.out_dir/'result.json',dict(status='complete',method=a.method,horizon=h,joint=d['joint'],seed=a.seed,updates=a.updates,train_windows=len(x),validation_windows=len(vx),wall_s=time.monotonic()-start,model_sha256=sha(a.out_dir/'model.pt'),data_sha256=sha(a.data),best_validation_first_action_mse=best,parameters=sum(p.numel()for p in model.parameters()),normalization='teacher obs/action, train-only physical states',first_action_weight=a.first_weight,state_loss_weight=a.state_weight,optimizer_resume=False,script_sha256=sha(__file__)))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','collect','replay','dataset','train','evaluate','probe','openloop']);p.add_argument('--root',type=Path,required=True);p.add_argument('--out-dir',type=Path);p.add_argument('--source',type=Path);p.add_argument('--data',type=Path);p.add_argument('--policy',type=Path);p.add_argument('--cohort',choices=['train','validation','test'],default='validation');p.add_argument('--method',choices=['mlp','diffusion'],default='mlp');p.add_argument('--horizon',type=int,default=8);p.add_argument('--joint',action='store_true');p.add_argument('--seed',type=int,default=4404);p.add_argument('--limit',type=int,default=0);p.add_argument('--updates',type=int,default=8000);p.add_argument('--first-weight',type=float,default=1.);p.add_argument('--state-weight',type=float,default=.25);p.add_argument('--guide',type=float,default=0.);p.add_argument('--unit-gradient',action='store_true');p.add_argument('--guide-target',type=float,default=0.);p.add_argument('--probe-steps',type=int,default=16)
    p.add_argument("--execute-steps",type=int,default=1);p.add_argument("--guide-mode",choices=["lateral","heading"],default="lateral")
    p.add_argument("--guide-start-seconds",type=float,default=0.)
    a=p.parse_args()
    for key in ['root','out_dir','source','data','policy']:
        if getattr(a,key)is not None:setattr(a,key,getattr(a,key).resolve())
    if a.out_dir is None:a.out_dir=a.root/'collection'
    if a.mode=='prepare':prepare(a)
    elif a.mode=='dataset':dataset(a)
    elif a.mode=='train':train(a)
    else:simulate(a)
if __name__=='__main__':main()
