from pathlib import Path
import sys,json,csv,math,hashlib
import numpy as np,torch
R=Path(__file__).resolve().parents[1];B=R/'outputs/guidance_comparison_v2';sys.path.insert(0,str(R/'src'));torch.set_num_threads(2)
from humanoid_imitation.planning.state_action_sequence import yaw
from humanoid_imitation.planning.obstacle_geometry import signed_distance

def world(e,p,step):
 a=float(yaw(e['pre']['root_rot'][step]));c,s=np.cos(a),np.sin(a)
 return p[...,:2].numpy()@np.array([[c,s],[-s,c]])+e['pre']['root_pos'][step,:2].numpy()

def sdf(e,p):
 g=e['geometry'];pts=torch.as_tensor(p,dtype=torch.float32)[None]
 return signed_distance(pts,torch.tensor(e['obstacle_xy'])[None],torch.tensor(g['half_extents_m'])[None],torch.tensor([g['angle_world_rad']]),torch.tensor([g['circle']]))[0].numpy()

def item(e,path,method,seed):
 r=dict(e['row']);g=e['geometry'];T=len(e['next']['root_pos']);dt=e['dt'];v=e['predicted_root'][:,:,9:11].norm(dim=-1);w=e['predicted_root'][:,:,14].abs()
 # Report one-second preview during first physical second, matching V1 diagnosis.
 mask=min(T,30);pv=v[:mask,:30].numpy();pw=w[:mask,:30].numpy();acc=torch.diff(e['predicted_root'][:mask,:30,9:11],dim=1).norm(dim=-1)*30
 pos=e['next']['root_pos'][:,:2]; goal=torch.tensor(e['goal']['xy']);dist=(pos-goal).norm(dim=-1);angles=yaw(e['next']['root_rot'])-e['goal']['yaw'];angles=torch.atan2(angles.sin(),angles.cos()).abs()*180/torch.pi
 entry=(dist<=1.25).nonzero();postturn=None
 if len(entry):
  a=yaw(e['next']['root_rot'])[int(entry[0]):];d=torch.diff(a);postturn=float(torch.atan2(d.sin(),d.cos()).abs().sum()*180/torch.pi)
 one=[]
 for k in range(0,T,15):one.append(np.linalg.norm(world(e,e['predicted_root'][k,:1],k)[0]-pos[k].numpy()))
 interventions=[]
 if 'nominal_root'in e:
  for k in range(0,T,15):
   npath=world(e,e['nominal_root'][k,:30],k);ppath=world(e,e['predicted_root'][k,:30],k);nc=float(sdf(e,npath).min())
   if nc<.5:interventions.append(float(sdf(e,ppath).min())-nc)
 category='success';diag={}
 if not r['success']:
  contact=r.get('first_contact_s')or r.get('first_conservative_collision_s')
  if contact is not None:
   k=max(0,min(T-1,round((contact-.5)/dt)-1));plan=world(e,e['predicted_root'][k,:30],k);margin=float(sdf(e,plan).min())
   if margin<0:category='generated_root_enters_obstacle'
   elif margin<.5:category='generated_plan_insufficient_margin'
   else:category='safe_root_plan_but_body_execution_contact'
   diag=dict(diagnostic_time_s=(k+1)*dt,contact_time_s=contact,predicted_1s_min_root_sdf_m=margin,diagnostic='root proxy at .5s pre-contact; not proof of exclusive cause')
  elif r['fall']:category='no_contact_fall_generation_execution_unresolved'
  else:category='no_contact_goal_or_hold_failure'
 width=2*max(g['half_extents_m'])
 return dict(method=method,seed=seed,scene_id=g['id'],source_id=r['source_id'],kind=g['kind'],width_m=width,scope='primary_small_circle_box'if g['kind']in ['circle','box']and width<=.80001 else 'stress_large_or_wall',success=r['success'],fall=r['fall'],contact=r.get('physical_contact',False),collision=r['conservative_collision'],duration_s=r['duration_s'],position_error_m=r['position_error_m'],yaw_error_deg=r['yaw_error_deg'],post_entry_turn_deg=postturn,plan_speed_p95_mps=float(np.percentile(pv,95)),plan_yaw_rate_p95_radps=float(np.percentile(pw,95)),plan_accel_p95_mps2=float(torch.quantile(acc.flatten(),.95)),mean_1step_root_error_m=float(np.mean(one)),risky_nominal_steps=len(interventions),mean_root_clearance_change_m=float(np.mean(interventions))if interventions else None,failure_category=category,trajectory_file=str(path),**diag)

def main():
 rows=[]
 for arm in ['M0','M1','D2','D3']:
  for seed in ([4504,4505]if arm=='D2'else[4504]):
   folder=B/f'final/{arm}_{seed}'
   for p in sorted(folder.glob('episode_*.pt')):rows.append(item(torch.load(p,map_location='cpu',weights_only=False),p,arm,seed))
 for seed in [4504,4505]:
  folder=R/f'outputs/guidance_only_v1/final/D0_legacy_{seed}'
  for p in sorted(folder.glob('episode_*.pt')):rows.append(item(torch.load(p,map_location='cpu',weights_only=False),p,'D0_reused',seed))
 fields=sorted(set().union(*(x.keys()for x in rows)))
 with (B/'episodes.csv').open('w')as f:
  out=csv.DictWriter(f,fieldnames=fields);out.writeheader();out.writerows(rows)
 methods={}
 for method in sorted({x['method']for x in rows}):
  rr=[x for x in rows if x['method']==method];scopes={}
  for key in ['all','primary_small_circle_box','stress_large_or_wall','circle','box','wall']+[f"{kind}_width_{width:.1f}"for kind,width in sorted({(x['kind'],round(x['width_m'],1))for x in rr})]:
   selected=rr if key=='all'else[x for x in rr if x['scope']==key or x['kind']==key or f"{x['kind']}_width_{x['width_m']:.1f}"==key]
   if not selected:continue
   scopes[key]=dict(n=len(selected),successes=sum(x['success']for x in selected),falls=sum(x['fall']for x in selected),contacts=sum(x['contact']for x in selected),collisions=sum(x['collision']for x in selected),mean_position_error_m=float(np.mean([x['position_error_m']for x in selected])),mean_yaw_error_deg=float(np.mean([x['yaw_error_deg']for x in selected])))
  fails={}
  for x in rr:fails[x['failure_category']]=fails.get(x['failure_category'],0)+1
  profile_arm=method.replace('_reused','');profile=json.loads((B/f'profile/{profile_arm}/result.json').read_text());ms=profile['inference_ms']
  methods[method]=dict(groups=scopes,failure_categories=fails,plan_speed_p95_episode_mean=float(np.mean([x['plan_speed_p95_mps']for x in rr])),plan_yaw_rate_p95_episode_mean=float(np.mean([x['plan_yaw_rate_p95_radps']for x in rr])),single_env_planning_decoder_p50_ms=float(np.percentile(ms,50)),single_env_planning_decoder_p95_ms=float(np.percentile(ms,95)),profile_samples=len(ms))
 # Verify every scored run shares exact goal, initial state and obstacle per scene.
 match={};checks=0
 for x in rows:
  e=torch.load(x['trajectory_file'],map_location='cpu',weights_only=False);s=x['scene_id'];canonical=(e['goal'],e['obstacle_xy'],e['geometry'],{k:v[0]for k,v in e['pre'].items()})
  if s not in match:match[s]=canonical;continue
  old=match[s];assert canonical[:3]==old[:3]
  for k in old[3]:assert torch.equal(old[3][k],canonical[3][k]),(s,k)
  checks+=1
 executions=[json.loads(p.read_text())for p in B.glob('*_execution.json')];assert all(x['exitcode']==0 for x in executions)
 # Paired source-cluster bootstrap: sampling-seed averaging is not a new learning seed.
 differences=[]
 for source in sorted({x['source_id']for x in rows}):
  m=[x['success']for x in rows if x['method']=='M1'and x['source_id']==source]
  d=[x['success']for x in rows if x['method']=='D2'and x['source_id']==source]
  differences.append(float(np.mean(d)-np.mean(m)))
 rng=np.random.default_rng(4504);boot=np.array(differences)[rng.integers(0,len(differences),size=(10000,len(differences)))].mean(1)
 paired=dict(source_clusters=len(differences),difference_percentage_points=100*float(np.mean(differences)),percentile_95_interval_percentage_points=(100*np.percentile(boot,[2.5,97.5])).tolist(),scope='fixed checkpoints and reused development-source scenes only; not architecture or learning-seed uncertainty')
 summary=dict(status='completed',paired_source_cluster_bootstrap=paired,methods=methods,new_scored_episodes=168,reused_scored_episodes=48,profile_runs=5,training_updates=0,matched_scenario_state_comparisons=checks,subprocess_wall_sum_s=sum(x['wall_s']for x in executions),selection=json.loads((B/'selection.json').read_text()),limits=json.loads((B/'protocol.json').read_text())['kinematic_limits'],diagnosis_not_causal_ground_truth=True)
 (B/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(methods,indent=2))
if __name__=='__main__':main()
