#!/usr/bin/env python3
"""Teacher route collection, frozen decoder certification, causal common dataset."""
import argparse,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_state_action_sequence as phys
from run_future_conditioned_action_v6 import SOURCE,load
from probe_consistent_root_guidance_v8 import DECODER
from pilot_direct_action_bc import write,sha
from humanoid_imitation.planning.state_action_sequence import yaw,local_state
from humanoid_imitation.planning.future_conditioned_action import root_features
from humanoid_imitation.planning.arrival_success import ArrivalHoldMonitor
BASE=ROOT/'outputs/obstacle_repair_v10/detours'


def simulate(a):
    source=[]
    if a.mode=='collect':
        if a.extra_sources:
            excluded=set()
            for path in SOURCE.glob('branch_*.pt'):
                e=torch.load(path,weights_only=False);excluded.add(e['case']['source_id'])
            for folder in a.exclude_source_dir:
                excluded.update(row['source_id']for row in json.loads((folder/'result.json').read_text())['rows'])
            folder=ROOT/'outputs/state_action_sequence_v1/collection_certified';index=json.loads((folder/'result.json').read_text());counts={'train':0,'validation':0}
            for row in index['rows']:
                split=row['split'];limit=a.extra_train_limit if split=='train'else a.extra_validation_limit
                if split not in counts or counts[split]>=limit or row['id']in excluded or not row['success']:continue
                e=torch.load(folder/f"episode_{row['id']:03d}.pt",weights_only=False);chosen=None
                for step in [60,75,90,120]:
                    if len(e['obs'])<=step+60:continue
                    v=e['pre']['root_vel'][step,:2];h=yaw(e['pre']['root_rot'][step]);alignment=(v*torch.stack((h.cos(),h.sin()))).sum()/v.norm().clamp_min(1e-6)
                    if v.norm()>=.3 and abs(float(e['pre']['root_ang_vel'][step,2]))<=.6 and alignment>=.8:chosen=step;break
                if chosen is None:continue
                base=dict(e,case=dict(e['case'],source_id=row['id'],branch_deg=0),pre={k:v[chosen:]for k,v in e['pre'].items()},prefix_obs=e['obs'][chosen-3:chosen],prefix_action=e['applied_action'][chosen-3:chosen]);counts[split]+=1
                for angle in [0,a.left_angle,-a.right_angle]:source.append(dict(base,route_deg=angle))
            print('EXTRA SOURCE COUNTS',counts,flush=True)
        else:
            for path in sorted(SOURCE.glob('branch_*.pt')):
                e=torch.load(path,weights_only=False)
                if e['case']['branch_deg']==0:
                    for angle in ([0,a.left_angle,-a.right_angle]if a.smooth_collection else[0,45,-45]):source.append(dict(e,route_deg=angle))
    else:
        for path in sorted((BASE/'teacher').glob('episode_*.pt')):
            e=torch.load(path,weights_only=False)
            if e['row']['usable']:source.append(e)
    if not source:raise ValueError('No eligible source')
    cases=[dict(e['case'],route_deg=e['route_deg'])for e in source];n=len(cases);saved={k:torch.stack([e['pre'][k][0]for e in source])for k in phys.STATE_METHODS};saved['standing']=torch.zeros(n,dtype=torch.bool);a.root=ROOT/'outputs/state_action_sequence_v1';a.seed=5054;phys.check(a);tic=time.monotonic();env,agent=phys.init_env(a,cases,saved);decoder,stats=load(DECODER)if a.mode=='certify'else(None,None)
    from envs.base_env import DoneFlags
    dt=env._engine.get_timestep();chunks=[];alive=torch.ones(n,dtype=torch.bool,device='cuda');rows=[None]*n;monitors=[ArrivalHoldMonitor(position_m=1.25,yaw_deg=45,speed_mps=.15,yaw_rate_radps=.2,hold_s=5,drift_m=.1)for _ in cases]
    with torch.no_grad():
        env.reset();initial=phys.state(env);err={k:float((initial[k].cpu()-saved[k]).abs().max())for k in phys.STATE_METHODS}
        if max(err.values())>2e-5:raise ValueError('Reset mismatch')
        heading=yaw(initial['root_rot']);forward=torch.stack((heading.cos(),heading.sin()),-1);final=initial['root_pos'][:,:2]+5*forward;angle=heading+torch.tensor([math.radians(c['route_deg'])for c in cases],device='cuda');mid=initial['root_pos'][:,:2]+4*torch.stack((angle.cos(),angle.sin()),-1)
        env.goal_xy[:]=mid if a.mode=='collect'else final;env.goal_yaw[:]=angle if a.mode=='collect'else heading;env._refresh_changed_goal();env._update_observations(None);obs,info=env._obs_buf,env._info
        for step in range(1350):
            if a.mode=='collect' and a.smooth_collection and step<60:
                request=heading+torch.tensor([math.radians(c['route_deg'])for c in cases],device='cuda')*math.sin(math.pi*(step+1)/60)
                current=phys.state(env)['root_pos'];env.goal_xy[:]=current[:,:2]+4*torch.stack((request.cos(),request.sin()),-1);env.goal_yaw[:]=request;env._refresh_changed_goal();env._update_observations(None);obs,info=env._obs_buf,env._info
            if a.mode=='collect' and step==(60 if a.smooth_collection else 75):
                env.goal_xy[:]=final;env.goal_yaw[:]=heading;env._refresh_changed_goal();env._update_observations(None);obs,info=env._obs_buf,env._info
            pre=phys.state(env);before=obs.clone()
            if a.mode=='collect':requested,_=agent._decide_action(obs,info)
            else:
                future={k:torch.stack([e['next'][k][(torch.arange(30)+step).clamp(max=len(e['obs'])-1)]for e in source]).cuda()for k in phys.STATE_METHODS};plan=local_state(future,pre['root_pos'][:,None],yaw(pre['root_rot'])[:,None]);feature=root_features(plan);requested=decoder(((obs-stats['obs_mean'])/stats['obs_std']).clamp(-10,10),feature)*stats['action_std']+stats['action_mean']
            action=requested.clamp(env._action_bound_low,env._action_bound_high);obs,_,done,info=env.step(action);post=phys.state(env)
            if not all(torch.isfinite(v).all()for v in post.values()):raise ValueError('Nonfinite state')
            chunks.append(dict(obs=before.cpu(),applied_action=action.cpu(),done=done.cpu(),pre={k:v.cpu()for k,v in pre.items()},next={k:v.cpu()for k,v in post.items()}));t=(step+1)*dt;yaws=yaw(post['root_rot'])
            for i,c in enumerate(cases):
                if not alive[i]:continue
                dist=float((post['root_pos'][i,:2]-final[i]).norm());delta=yaws[i]-heading[i];angle_error=float(torch.atan2(delta.sin(),delta.cos()));speed=float(post['root_vel'][i,:2].norm());wz=float(post['root_ang_vel'][i,2]);monitors[i].update(t,dist,angle_error,speed,wz,post['root_pos'][i,:2].cpu().numpy());first=monitors[i].first_success_time_s;flag=int(done[i]);fall=flag==DoneFlags.FAIL.value;complete=first is not None and t>=first+2-1e-8
                if fall or complete or flag!=DoneFlags.NULL.value or step==1349:
                    alive[i]=False;rows[i]=dict(index=i,source_id=c['source_id'],split=c['split'],route_deg=c['route_deg'],goal_success=bool(complete and not fall),fall=fall,steps=step+1,duration_s=t,position_error_m=dist,yaw_error_deg=math.degrees(abs(angle_error)))
            if not alive.any():break
    for i,c in enumerate(cases):
        row=rows[i];count=row['steps'];e={k:torch.stack([v[k][i]for v in chunks[:count]])for k in ['obs','applied_action','done']}
        for g in ['pre','next']:e[g]={k:torch.stack([v[g][k][i]for v in chunks[:count]])for k in phys.STATE_METHODS}
        delta=e['next']['root_pos'][:,:2]-initial['root_pos'][i,:2].cpu();h=heading[i].cpu();lat=-h.sin()*delta[:,0]+h.cos()*delta[:,1];forward_x=h.cos()*delta[:,0]+h.sin()*delta[:,1];early=forward_x<3.;sign=1 if c['route_deg']>0 else -1;excursion=float((sign*lat[early]).max())if early.any()and c['route_deg']else 0.;yy=yaw(e['next']['root_rot']);rotation=float(torch.atan2((yy[1:]-yy[:-1]).sin(),(yy[1:]-yy[:-1]).cos()).abs().sum().rad2deg());row.update(early_route_excursion_m=excursion,total_rotation_deg=rotation)
        row['usable']=row['goal_success']and not(e['done']!=0).any().item()and rotation<270 and (c['route_deg']==0 or excursion>=.4)
        if a.mode=='certify':
            target=source[i]['next']['root_pos'][(torch.arange(count)).clamp(max=len(source[i]['obs'])-1),:2];rmse=float((e['next']['root_pos'][:,:2]-target).square().sum(-1).mean().sqrt());row['reference_root_xy_rmse_m']=rmse;row['usable']&=rmse<=.5
        e.update(case=c,row=row,route_deg=c['route_deg'],goal=dict(xy=final[i].cpu().tolist(),yaw=float(heading[i])),dt=dt,prefix_obs=source[i]['prefix_obs'],prefix_action=source[i]['prefix_action'],reference_source='V31 teacher routed rollout'if a.mode=='certify'else'V4 source initial state; routed V31 teacher',action_source='common V6 decoder'if a.mode=='certify'else'V31 teacher');torch.save(e,a.out_dir/f'episode_{i:03d}.pt')
    continuity=max(float((chunks[j]['next'][k]-chunks[j+1]['pre'][k]).abs().max())for j in range(len(chunks)-1)for k in phys.STATE_METHODS)
    if continuity>1e-6:raise ValueError('Continuity failed')
    write(a.out_dir/'result.json',dict(status='complete',mode=a.mode,rows=rows,episodes=n,usable=sum(x['usable']for x in rows),goal_successes=sum(x['goal_success']for x in rows),falls=sum(x['fall']for x in rows),reset_errors=err,continuity_max=continuity,transitions=sum(x['steps']for x in rows),wall_s=time.monotonic()-tic,teacher_sha256=sha(phys.TEACHER),decoder_sha256=sha(DECODER),runner_sha256=sha(__file__),evaluation_waypoints=False,collection_waypoints=a.mode=='collect',reference_future_diagnostic=a.mode=='certify'))
    print('COMPLETE',a.mode,n,sum(x['usable']for x in rows),flush=True)


def dataset(a):
    old=torch.load(ROOT/'outputs/state_action_sequence_v2/dataset_h30/data.pt',weights_only=False);parts={s:[]for s in ['train','validation']};coverage={s:{}for s in parts};certificates=[a.certificate_dir or BASE/'certified',*a.additional_certificate_dir];rows=[]
    for certificate in certificates:
        result=json.loads((certificate/'result.json').read_text())
        if result['decoder_sha256']!=sha(DECODER):raise ValueError('Certificate decoder version mismatch')
        rows.extend(result['rows'])
    for row in rows:
        if not row['usable']:continue
        coverage[row['split']].setdefault(str(row['source_id']),[]).append(row['route_deg'])
    route_set={0,a.left_angle,-a.right_angle}if a.smooth_collection else{0,45,-45};triplets={s:sum(set(v)==route_set for v in c.values())for s,c in coverage.items()}
    if triplets['train']<16 or triplets['validation']<6:raise ValueError('Insufficient certified route triplets; no dataset adopted')
    for path in sorted(p for folder in certificates for p in folder.glob('episode_*.pt')):
        e=torch.load(path,weights_only=False);row=e['row']
        if not row['usable']:continue
        starts=torch.linspace(0,len(e['obs'])-30,min(300,len(e['obs'])-29)).round().long().unique();obs=torch.cat((e['prefix_obs'],e['obs']));acts=torch.cat((e['prefix_action'],e['applied_action']));h=starts[:,None]+torch.arange(4);past=starts[:,None]+torch.arange(3);future=starts[:,None]+torch.arange(30);x=torch.cat((obs[h].flatten(1),acts[past].flatten(1)),-1);state=local_state({k:v[future]for k,v in e['next'].items()},e['pre']['root_pos'][starts,None],yaw(e['pre']['root_rot'][starts])[:,None]);y=torch.cat((e['applied_action'][future],state),-1);parts[row['split']].append(dict(context=x,target=y,episode_id=torch.full((len(x),),row['source_id'],dtype=torch.long)))
    for s,ps in parts.items():
        n=len(old[s]['context']);old[s]['group']=torch.cat((torch.zeros(n,dtype=torch.long),torch.ones(sum(len(v['context'])for v in ps),dtype=torch.long)))
        for k in ['context','target','episode_id']:old[s][k]=torch.cat((old[s][k],*[v[k]for v in ps]))
    old['execution_decoder_path']=str(DECODER);old['execution_decoder_sha256']=sha(DECODER)
    torch.save(old,a.out_dir/'data.pt');write(a.out_dir/'result.json',dict(status='complete',certified_triplets=triplets,windows={s:len(old[s]['context'])for s in parts},new_windows={s:int(old[s]['group'].sum())for s in parts},execution_decoder_path=str(DECODER),execution_decoder_sha256=sha(DECODER),normalization='unchanged original train-only stats',goal_input='fixed final goal from actual decoder observation; no route label/waypoint input',target='actual common decoder applied action and next physical state',data_sha256=sha(a.out_dir/'data.pt')))
    print('DATASET COMPLETE',triplets,flush=True)


def main():
    global BASE
    global DECODER
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','certify','dataset']);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--base-dir',type=Path,default=BASE);p.add_argument('--smooth-collection',action='store_true');p.add_argument('--left-angle',type=float,default=30);p.add_argument('--right-angle',type=float,default=30);p.add_argument('--decoder-path',type=Path,default=DECODER);p.add_argument('--certificate-dir',type=Path);p.add_argument('--additional-certificate-dir',type=Path,action='append',default=[]);p.add_argument('--extra-sources',action='store_true');p.add_argument('--extra-train-limit',type=int,default=32);p.add_argument('--extra-validation-limit',type=int,default=12);p.add_argument('--exclude-source-dir',type=Path,action='append',default=[]);a=p.parse_args();BASE=a.base_dir.resolve();DECODER=a.decoder_path.resolve();a.out_dir=a.out_dir.resolve();a.out_dir.mkdir(parents=True,exist_ok=False)
    if a.mode=='dataset':dataset(a)
    else:simulate(a)
if __name__=='__main__':main()
