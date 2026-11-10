#!/usr/bin/env python3
"""Frozen closeout comparison: MLP/DP, post optimization, guidance and analytic root."""
import argparse,json,math,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src'),'/home/an/Desktop/MimicKit-main/mimickit']
import run_state_action_sequence as phys
import run_short_action_sequence as core
import run_anchored_action_sequence as anchored
from run_future_conditioned_action_v6 import load,SOURCE
from probe_consistent_root_guidance_v8 import MODEL,DECODER
from pilot_direct_action_bc import write,sha
from humanoid_imitation.planning.state_action_sequence import yaw,rotate_z,sample_sequence
from humanoid_imitation.planning.future_conditioned_action import root_features as first_second_features
from humanoid_imitation.planning.obstacle_geometry import motion_cost,avoidance_direction,guidance_target,body_clearance
from humanoid_imitation.planning.arrival_success import ArrivalHoldMonitor
from humanoid_imitation.planning.obstacle_root_retarget import retarget_root
from humanoid_imitation.planning.root_output_optimization import optimize_root




def main():
    global MODEL,DECODER
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path,default=ROOT/'configs/releases/closeout_guidance_manifest.json')
    p.add_argument('--arm',choices=['M0','D0','M1','D1','D2','D3'],required=True)
    p.add_argument('--scene-file',type=Path)
    p.add_argument('--cohort',choices=['train','validation'],default='validation')
    p.add_argument('--indices',type=int,nargs='+')
    p.add_argument('--seed',type=int,default=4504)
    p.add_argument('--gate',choices=['all','active'],default='all')
    p.add_argument('--lr',type=float,default=.03)
    p.add_argument('--no-curve',action='store_true')
    p.add_argument('--prior-only',action='store_true')
    p.add_argument('--clear',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();a.out_dir=a.out_dir.resolve();a.out_dir.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.manifest.resolve().read_text())
    for path,digest in manifest['frozen'].items():
        if sha(path)!=digest:raise ValueError('Frozen input changed: '+path)
    MODEL=Path(manifest['models']['mlp' if a.arm.startswith('M') else 'diffusion']);DECODER=Path(manifest['models']['decoder'])
    a.root=ROOT/'outputs/state_action_sequence_v1';phys.check(a)
    a.mode='clear' if a.clear else 'obstacle_guided';a.arrival_route='preserve'
    sources={}
    for path in sorted(SOURCE.glob('branch_*.pt')):
        e=torch.load(path,weights_only=False)
        if e['case']['split']==a.cohort and e['case']['branch_deg']==0:sources[e['case']['source_id']]=e
    if a.scene_file:
        specs=json.loads(a.scene_file.resolve().read_text())
    else:
        specs=[]
        for i,source in enumerate(list(sources)[:12]):
            kind=['wide','box','wall','angled'][i%4]
            specs.append(dict(id=i,source_id=source,kind=kind,half_extents_m={'wide':[.6,.6],'box':[.4,.6],'wall':[.125,1.],'angled':[.125,1.]}[kind],circle=kind=='wide',goal_distance_m=6.,goal_bearing_deg=[-20,0,20][i%3],obstacle_distance_m=2.8,obstacle_side_offset_m=[-.3,0,.3][(i//3)%3],relative_shape_yaw_deg=35 if kind=='angled'else 0))
    if a.indices is not None:specs=[specs[i] for i in a.indices]
    es=[sources[v['source_id']] for v in specs]
    n=len(es)
    if not n:raise ValueError('No sources')
    saved={k:torch.stack([e['pre'][k][0] for e in es]) for k in phys.STATE_METHODS};saved['standing']=torch.zeros(n,dtype=torch.bool);cases=[e['case'] for e in es]
    heading=yaw(saved['root_rot'])
    bearing=torch.tensor([math.radians(v['goal_bearing_deg']) for v in specs])
    travel_heading=heading+bearing
    forward=torch.stack((travel_heading.cos(),travel_heading.sin()),-1);side=torch.stack((-travel_heading.sin(),travel_heading.cos()),-1)
    obstacle_distance=torch.tensor([v['obstacle_distance_m'] for v in specs]);goal_distance=torch.tensor([v['goal_distance_m'] for v in specs])
    offset=torch.tensor([v['obstacle_side_offset_m'] for v in specs])
    kinds=[v['kind'] for v in specs]
    half=torch.tensor([v['half_extents_m'] for v in specs]);is_circle=torch.tensor([v['circle'] for v in specs])
    shape_angle=travel_heading+torch.tensor([math.radians(v['relative_shape_yaw_deg']) for v in specs])
    center=saved['root_pos'][:,:2]+obstacle_distance[:,None]*forward+offset[:,None]*side
    radius=.6  # Only a legacy display field; collision uses actual signed geometry.
    assets=[];scenario_rows=[]
    for i in range(n):
        asset=a.out_dir/f'obstacle_{i:03d}.usd';assets.append(asset)
        if is_circle[i]:asset.write_text((ROOT/'configs/assets/guidance_pole.usd').read_text().replace('double radius = 0.15',f'double radius = {float(half[i,0])}'))
        else:
            text=(ROOT/'configs/assets/guidance_pole.usd').read_text();start=text.index('    def Cylinder')
            text=text[:start]+f'''    def Cube "Collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
        double size = 1
        double3 xformOp:scale = ({2*half[i,0]}, {2*half[i,1]}, 1.5)
        double xformOp:rotateZ = {math.degrees(float(shape_angle[i]))}
        uniform token[] xformOpOrder = ["xformOp:rotateZ", "xformOp:scale"]
        color3f[] primvars:displayColor = [(0.85, 0.25, 0.15)]
    }}
}}
''';asset.write_text(text)
        scenario_rows.append(dict(specs[i],index=i,source_id=cases[i]['source_id'],kind=kinds[i],goal_distance_m=float(goal_distance[i]),goal_bearing_deg=math.degrees(float(bearing[i])),final_yaw_relative_initial_deg=0,obstacle_distance_m=float(obstacle_distance[i]),obstacle_side_offset_m=float(offset[i]),half_extents_m=half[i].tolist(),angle_world_rad=float(shape_angle[i]),circle=bool(is_circle[i])))
    write(a.out_dir/'scenarios.json',scenario_rows)
    import humanoid_imitation.envs.goal_amp_env as goal_module
    import engines.engine as engine
    original=goal_module.GoalAMPEnv
    class ObstacleEnv(original):
        def __init__(self,env_config,**kwargs):
            if a.arrival_route!='preserve':
                env_config['goal']['sequential_arrival']['turn_route']=a.arrival_route
                import yaml
                (a.out_dir/'env.yaml').write_text(yaml.safe_dump(env_config))
            super().__init__(env_config=env_config,**kwargs)
        def _build_env(self,env_id,config):
            super()._build_env(env_id,config)
            if a.mode!='clear':
                pos=torch.cat((center[env_id],torch.tensor([.75]))).numpy()
                self._engine.create_obj(env_id,engine.ObjType.rigid,str(assets[env_id]),'obstacle',fix_root=True,start_pos=pos)
    goal_module.GoalAMPEnv=ObstacleEnv
    tic=time.monotonic();sampling_seed=a.seed;a.seed=4504;env,agent=phys.init_env(a,cases,saved);a.seed=sampling_seed;decoder,stats=load(DECODER);core.make_model=anchored.make_anchored;generator=core.ShortActionPolicy(MODEL,a.seed);generator.model.requires_grad_(False);decoder.requires_grad_(False)
    generator.hist=[torch.stack([e['prefix_obs'][j] for e in es]).cuda() for j in range(3)];generator.actions=[torch.stack([e['prefix_action'][j] for e in es]).cuda() for j in range(3)]
    from envs.base_env import DoneFlags
    dt=env._engine.get_timestep();center=center.cuda();heading=heading.cuda();half=half.cuda();shape_angle=shape_angle.cuda();is_circle=is_circle.cuda();travel_heading=travel_heading.cuda();avoid_side=torch.where(offset.cuda()>0,-1.,1.); chunks=[];alive=torch.ones(n,dtype=torch.bool,device='cuda');rows=[None]*n;monitors=[ArrivalHoldMonitor(position_m=1.25,yaw_deg=45,speed_mps=.15,yaw_rate_radps=.2,hold_s=5,drift_m=.1) for _ in es];collision=torch.zeros(n,dtype=torch.bool,device='cuda');minroot=torch.full((n,),float('inf'),device='cuda');minbody=minroot.clone();minsegment=minroot.clone();maxforce=torch.zeros(n,device='cuda')
    guidance_seconds=0.;guidance_calls=0;inference_ms=[];prediction_errors=[];feature_audit=[];collision_force=torch.zeros(n,dtype=torch.bool,device='cuda');first_collision=torch.full((n,),-1.,device='cuda');first_force=first_collision.clone();passed=collision_force.clone();passed_safe=collision_force.clone();foot_slip=[]
    with torch.no_grad():
        env.reset();initial=phys.state(env);errors={k:float((initial[k].cpu()-saved[k]).abs().max()) for k in phys.STATE_METHODS}
        if max(errors.values())>2e-5:raise ValueError('Reset mismatch')
        goal=initial['root_pos'][:,:2]+goal_distance.cuda()[:,None]*forward.cuda();env.goal_xy[:]=goal;env.goal_yaw[:]=heading;env._refresh_changed_goal();env._update_observations(None);obs=env._obs_buf
        parents=torch.tensor([max(0,env._kin_char_model.get_parent_id(j)) for j in range(len(env._kin_char_model.get_body_names()))],device='cuda')
        for step in range(round(45/dt)):
            pre=phys.state(env);current=obs.clone()
            if a.profile:torch.cuda.synchronize();profile_start=time.monotonic()
            generator.hist=(generator.hist+[obs.clone()])[-4:];history=[generator.hist[0]]*(4-len(generator.hist))+generator.hist;past=[torch.zeros(n,28,device='cuda')]*(3-len(generator.actions))+generator.actions;context=torch.cat([*history,*past],-1);context=((context-generator.stats['context_mean'])/generator.stats['context_std']).clamp(-10,10);noise=torch.randn(n,generator.h*99,device='cuda',generator=generator.rng)
            def physical(z):return(z*generator.stats['target_std']+generator.stats['target_mean']).reshape(-1,generator.h,99)
            nominal=physical(sample_sequence(generator.model,context,generator.alpha,noise));anchor=yaw(pre['root_rot']);delta=torch.cat((center-pre['root_pos'][:,:2],torch.zeros(n,1,device='cuda')),-1);local=rotate_z(delta,-anchor)[:,:2]
            desired,active=avoidance_direction(pre['root_pos'][:,:2],goal,center,half,shape_angle,is_circle,travel_heading,avoid_side)
            target=guidance_target(nominal[:,:,28:],anchor,desired);target=torch.where(active[:,None,None],target,nominal[:,:,28:])
            cost_active=active if a.gate=='active' else torch.ones_like(active)
            def root_cost(root):
                value=motion_cost(root,nominal[:,:,28:],local,half,shape_angle-anchor,is_circle,None if a.no_curve else target,anticipate=True)
                return value*cost_active
            def cost(z):return root_cost(physical(z)[:,:,28:])
            plan=nominal
            if not a.clear:
                if a.arm=='D2':
                    if a.profile:torch.cuda.synchronize()
                    guide_start=time.monotonic()
                    plan=physical(sample_sequence(generator.model,context,generator.alpha,noise,16,cost,30.,True))
                    if a.profile:torch.cuda.synchronize()
                    guidance_seconds+=time.monotonic()-guide_start;guidance_calls+=1
                elif a.arm in ['M1','D1']:
                    plan=nominal.clone();plan[:,:,28:]=optimize_root(nominal[:,:,28:],root_cost,cost_active,lr=a.lr,steps=32)
                elif a.arm=='D3':
                    desired,active=avoidance_direction(pre['root_pos'][:,:2],goal,center,half,shape_angle,is_circle,travel_heading,avoid_side,margin=1.)
                    plan=nominal.clone();plan[:,:,28:]=retarget_root(nominal[:,:,28:],anchor,desired,active)
            feature=first_second_features(plan[:,:30,28:]);x=((obs-stats['obs_mean'])/stats['obs_std']).clamp(-10,10)
            decoded=decoder.prior(x) if a.prior_only else decoder(x,feature)
            action=(decoded*stats['action_std']+stats['action_mean']).clamp(env._action_bound_low,env._action_bound_high)
            if a.profile:
                torch.cuda.synchronize()
                if step>=10:inference_ms.append((time.monotonic()-profile_start)*1000)
            if step==0:
                feature_audit.append(dict(feature_max_difference=float((feature-first_second_features(plan[:,:30,28:])).abs().max()),feature_dim=feature.shape[-1],action_reconstruction_max=float((action-(decoded*stats['action_std']+stats['action_mean']).clamp(env._action_bound_low,env._action_bound_high)).abs().max())))
            generator.actions=(generator.actions+[action.clone()])[-3:]
            if not torch.isfinite(plan).all() or not torch.isfinite(action).all():raise ValueError('Nonfinite prediction/action')
            obs,_,done,info=env.step(action);post=phys.state(env)
            if not all(torch.isfinite(v).all() for v in post.values()):raise ValueError('Nonfinite physical state')
            body=env._engine.get_body_pos(env._get_char_id()).clone();rootdist=(post['root_pos'][:,:2]-center).norm(dim=-1);bodydist=(body[:,:,:2]-center[:,None]).norm(dim=-1).min(-1).values
            segmentdist=body_clearance(body,parents,center,half,shape_angle,is_circle)
            # Backend creates no contact sensor for a rigid object. Subtract
            # character-ground contacts from all character contacts instead.
            force=(env._engine.get_contact_forces(env._get_char_id())-env._engine.get_ground_contact_forces(env._get_char_id())).sum(1).norm(dim=-1) if a.mode!='clear' else torch.zeros_like(maxforce)
            minroot=torch.where(alive,torch.minimum(minroot,rootdist),minroot);minbody=torch.where(alive,torch.minimum(minbody,bodydist),minbody);minsegment=torch.where(alive,torch.minimum(minsegment,segmentdist),minsegment);maxforce=torch.where(alive,torch.maximum(maxforce,force),maxforce)
            hit=alive & ((segmentdist<.08)|(force>1)) if not a.clear else torch.zeros_like(collision)
            first_collision=torch.where(hit & (first_collision<0),torch.full_like(first_collision,(step+1)*dt),first_collision)
            forcehit=alive & (force>1);first_force=torch.where(forcehit & (first_force<0),torch.full_like(first_force,(step+1)*dt),first_force)
            collision|=hit;collision_force|=forcehit
            along=((post['root_pos'][:,:2]-center)*forward.cuda()).sum(-1)
            rel=shape_angle-travel_heading;depth=torch.where(is_circle,half[:,0],half[:,0]*rel.cos().abs()+half[:,1]*rel.sin().abs())
            crossing=(~passed)&alive&(along>depth+.6);passed_safe|=crossing & ~collision;passed|=crossing
            predicted_world=pre['root_pos'][:,:2]+rotate_z(torch.cat((plan[:,0,28:30],torch.zeros(n,1,device='cuda')),-1),anchor)[:,:2]
            prediction_errors.append((predicted_world-post['root_pos'][:,:2]).norm(dim=-1).cpu())
            if step==0:
                feet=[j for j,name in enumerate(env._kin_char_model.get_body_names()) if name in ['left_foot','right_foot']]
            ground=env._engine.get_ground_contact_forces(env._get_char_id())
            prev_body=env._engine.get_body_pos(env._get_char_id()).clone() if step==0 else last_body
            speed_foot=(body[:,feet,:2]-prev_body[:,feet,:2]).norm(dim=-1)/dt
            contact_foot=ground[:,feet,2]>10
            moving=post['root_vel'][:,:2].norm(dim=-1)>.3
            foot_slip.append(torch.where(contact_foot & moving[:,None],speed_foot,torch.full_like(speed_foot,float('nan'))).cpu())
            last_body=body.clone()
            # Sensor contacts plus conservative body segments+8cm; no exact mesh-clearance claim.
            chunks.append(dict(obs=current.cpu(),applied_action=action.cpu(),done=done.cpu(),pre={k:v.cpu() for k,v in pre.items()},next={k:v.cpu() for k,v in post.items()},body_pos=body.cpu(),obstacle_contact_force_n=force.cpu(),predicted_root=plan[:,:,28:43].cpu(),guidance_active=cost_active.cpu()))
            yaws=yaw(post['root_rot']);t=(step+1)*dt
            for i,c in enumerate(cases):
                if not alive[i]:continue
                dist=float((post['root_pos'][i,:2]-goal[i]).norm());angle=yaws[i]-heading[i];angle=float(torch.atan2(angle.sin(),angle.cos()));speed=float(post['root_vel'][i,:2].norm());wz=float(post['root_ang_vel'][i,2]);monitors[i].update(t,dist,angle,speed,wz,post['root_pos'][i,:2].cpu().numpy());first=monitors[i].first_success_time_s;flag=int(done[i]);fall=flag==DoneFlags.FAIL.value;complete=first is not None and t>=first+2-1e-8
                if fall or complete or flag!=DoneFlags.NULL.value or step==1349:
                    alive[i]=False;rows[i]=dict(index=i,source_id=c['source_id'],goal_success=bool(complete and not fall),success=bool(complete and not fall and not collision[i]),fall=fall,conservative_collision=bool(collision[i]),duration_s=t,steps=step+1,position_error_m=dist,yaw_error_deg=math.degrees(abs(angle)),min_root_distance_m=float(minroot[i]),min_body_center_distance_m=float(minbody[i]),min_body_surface_clearance_lower_bound_m=float(minsegment[i]),max_obstacle_contact_force_n=float(maxforce[i]))
            if not alive.any():break
    for i,e in enumerate(es):
        n=rows[i]['steps'];out={k:torch.stack([x[k][i] for x in chunks[:n]]) for k in ['obs','applied_action','done','body_pos','obstacle_contact_force_n','predicted_root','guidance_active']}
        for g in ['pre','next']:out[g]={k:torch.stack([x[g][k][i] for x in chunks[:n]]) for k in phys.STATE_METHODS}
        rows[i].update(physical_contact=bool(collision_force[i]),first_contact_s=float(first_force[i]) if first_force[i]>=0 else None,first_conservative_collision_s=float(first_collision[i]) if first_collision[i]>=0 else None,passed_obstacle=bool(passed[i]),passed_obstacle_safely=bool(passed_safe[i]),root_prediction_mean_error_m=float(torch.stack(prediction_errors)[:n,i].mean()),moving_contact_foot_slip_proxy_mps=float(torch.stack(foot_slip)[:n,i].nanmean()))
        out.update(row=rows[i],goal=dict(xy=goal[i].cpu().tolist(),yaw=float(heading[i])),obstacle_xy=center[i].cpu().tolist(),obstacle_radius_m=radius,geometry=scenario_rows[i],scenario=scenario_rows[i],dt=dt);torch.save(out,a.out_dir/f'episode_{i:03d}.pt')
    continuity=max(float((chunks[j]['next'][k]-chunks[j+1]['pre'][k]).abs().max()) for j in range(len(chunks)-1) for k in phys.STATE_METHODS)
    if continuity>1e-6:raise ValueError('Continuity error')
    write(a.out_dir/'result.json',dict(status='complete',arm=a.arm,seed=a.seed,physics_seed=4504,gate=a.gate,lr=a.lr,no_curve=a.no_curve,prior_only=a.prior_only,cohort=a.cohort,scenarios=scenario_rows,guidance_margin_m=.5,guidance_time_s=guidance_seconds,guidance_calls=guidance_calls,mode=a.mode,rows=rows,feature_audit=feature_audit,inference_ms=inference_ms,manifest_sha256=sha(a.manifest.resolve()),successes=sum(x['success'] for x in rows),goal_successes=sum(x['goal_success'] for x in rows),falls=sum(x['fall'] for x in rows),conservative_collisions=sum(x['conservative_collision'] for x in rows),wall_s=time.monotonic()-tic,transitions=sum(x['steps'] for x in rows),reset_errors=errors,state_continuity_max=continuity,decoder_sha256=sha(DECODER),generator_sha256=sha(MODEL),runner_sha256=sha(__file__),physical_obstacle=a.mode!='clear',waypoints=False,new_training=False,horizon=generator.h,scale=30 if a.arm=='D2' else 0,arrival_route=a.arrival_route,collision_metric='contact force>1N OR segment signed surface clearance lower bound<.08m; 33 samples plus Lipschitz correction; conservative proxy'))
    print('COMPLETE',a.arm,sum(x['success'] for x in rows),sum(x['fall'] for x in rows),flush=True)
if __name__=='__main__':main()
