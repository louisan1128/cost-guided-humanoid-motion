from pathlib import Path
import sys,json,csv,math
import numpy as np,torch,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parents[1];B=R/'outputs/guidance_comparison_v2';sys.path[:0]=[str(B),str(R/'scripts'),str(R/'src'),'/home/an/Desktop/MimicKit-main/mimickit'];torch.set_num_threads(2)
from analyze_guidance_comparison_v2 import world,sdf
from humanoid_imitation.planning.state_action_sequence import yaw
from anim.mjcf_char_model import MJCFCharModel
from motion_video_renderer import load_geometry,render_geometry,obstacle_outline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
rows=list(csv.DictReader((B/'episodes.csv').open()));eligible=[x for x in rows if x['method']=='D2'and x['seed']=='4504'and x['success']=='True'and x['scope']=='primary_small_circle_box']
if not eligible:eligible=[x for x in rows if x['method']=='D2'and x['seed']=='4504']
scene=min(eligible,key=lambda x:int(x['scene_id']))['scene_id'];examples=[]
for m in ['D0_reused','M1','D2']:
 examples += [x for x in rows if x['scene_id']==scene and x['seed']=='4504'and x['method']==m][:1]
fails=[x for x in rows if x['method']=='D2'and x['seed']=='4504'and x['success']=='False']
if fails:examples.append(min(fails,key=lambda x:int(x['scene_id'])))
contrasts=[x for x in eligible if any(y['method']=='M1'and y['seed']=='4504'and y['scene_id']==x['scene_id']and y['success']=='False'for y in rows)]
if contrasts:
 contrast=min(contrasts,key=lambda x:int(x['scene_id']))['scene_id']
 for method in ['M1','D2']:
  examples += [x for x in rows if x['scene_id']==contrast and x['method']==method and x['seed']=='4504'][:1]
out=B/'visuals';out.mkdir(exist_ok=False);kin=MJCFCharModel('cpu');asset='/home/an/Desktop/MimicKit-main/data/assets/humanoid/humanoid.xml';kin.load(asset);names=kin.get_body_names();geo=load_geometry(asset,names)
fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';font=ImageFont.truetype(fontpath,18);small=ImageFont.truetype(fontpath,16);manifest=[]
for rr in examples:
 e=torch.load(rr['trajectory_file'],map_location='cpu',weights_only=False);g=e['geometry'];s=e['next'];T=len(s['root_pos']);dt=e['dt'];label=f"{rr['method']}_scene{int(rr['scene_id']):02d}_{'PASS'if rr['success']=='True'else'FAIL'}"
 # Preserve full saved physics in timeline; video focuses on at most first 20s.
 stop=min(T,600);st={k:v[:stop]for k,v in s.items()}
 with torch.no_grad():pos,rot=kin.forward_kinematics(st['root_pos'],st['root_rot'],kin.dof_to_rot(st['q']))
 obstacle=dict(xy=e['obstacle_xy'],height=1.5,kind='circle'if g['circle']else'box',radius=g['half_extents_m'][0],half=g['half_extents_m'],angle=g['angle_world_rad'])
 base=out/(label+'_physics.mp4');render_geometry(base,pos.numpy(),rot.numpy(),np.arange(1,stop+1)*dt,label,'Recorded actual physics; root previews are predictions, not body replay',names,'#8060b0',geo,dict(e['goal'],position_tolerance_m=1.25,yaw_tolerance_deg=45),obstacles=[obstacle])
 actual=s['root_pos'][:,:2].numpy();goal=np.array(e['goal']['xy']);outline=obstacle_outline(obstacle);bounds=np.vstack((actual[:stop],goal[None],outline));lo=bounds.min(0)-1.3;hi=bounds.max(0)+1.3;scale=min(420/(hi[0]-lo[0]),390/(hi[1]-lo[1]));center=(hi+lo)/2
 def project(points):
  p=np.asarray(points);return np.c_[1200+(p[:,0]-center[0])*scale,360-(p[:,1]-center[1])*scale]
 def line(d,p,color,width=3):
  pp=[tuple(v)for v in project(p)]
  if len(pp)>1:d.line(pp,fill=color,width=width)
 target=out/(label+'_analysis.mp4');reader=imageio.get_reader(base);frames={};count=0
 with imageio.get_writer(target,fps=30,codec='libx264',quality=7,macro_block_size=16,output_params=['-movflags','+faststart'])as writer:
  for i,frame in enumerate(reader):
   image=Image.new('RGB',(1440,800),'white');image.paste(Image.fromarray(frame).resize((960,600)),(0,70));d=ImageDraw.Draw(image)
   d.text((20,10),label+' | same frozen models / known geometry / 1x',fill='black',font=font)
   d.text((970,35),'TOP VIEW: predictions from actual state',fill='black',font=small)
   d.text((970,63),'gray: nominal / purple: selected plan',fill='black',font=small)
   d.text((970,87),'blue: actual executed root path',fill='black',font=small)
   d.polygon([tuple(x)for x in project(outline)],fill='#db8c77');line(d,actual[:i+1],'#286fa6',3)
   k=min(i,len(e['predicted_root'])-1);nominal=e.get('nominal_root',e['predicted_root'])[k]
   line(d,world(e,nominal[:30],k),'#8b939c',3);line(d,world(e,e['predicted_root'][k,:30],k),'#8b39b0',4)
   a=project(goal[None])[0];r=1.25*scale;d.ellipse((a[0]-r,a[1]-r,a[0]+r,a[1]+r),outline='#bf8300',width=2);d.ellipse((a[0]-4,a[1]-4,a[0]+4,a[1]+4),fill='#bf8300')
   dist=np.linalg.norm(actual[i]-goal);angle=float(yaw(s['root_rot'][i])-e['goal']['yaw']);angle=abs(math.degrees(math.atan2(math.sin(angle),math.cos(angle))));speed=float(s['root_vel'][i,:2].norm());force=float(e['obstacle_contact_force_n'][i]);margin=float(sdf(e,world(e,e['predicted_root'][k,:30],k)).min());ms=float(e['planning_ms'][i])if'planning_ms'in e else None
   values=[f't={(i+1)*dt:.2f}s / recorded {T*dt:.1f}s',f'goal distance {dist:.2f}m / yaw {angle:.1f}deg',f'actual speed {speed:.2f}m/s',f'obstacle force proxy {force:.1f}N',f'plan 1s minimum root SDF {margin:.2f}m',f'planning+decoder {ms:.1f}ms (batch)'if ms is not None else'latency: old run not instrumented']
   for j,v in enumerate(values):d.text((970,590+28*j),v,fill='#b02020'if j==3 and force>1 else'#263449',font=small)
   d.text((20,738),'Physics is replayed from measured states. Purple/gray lines are root forecasts; full-body feasibility is not guaranteed.',font=small,fill='black')
   d.text((20,765),'Clip capped at 20s; full episode and endpoint errors are included in the timeline PNG / CSV.',font=small,fill='black')
   writer.append_data(np.asarray(image));count+=1
   if i in {0,stop//3,2*stop//3,stop-1}:frames[i]=image.copy()
 reader.close();base.unlink()
 sheet=Image.new('RGB',(1440,800),'white')
 for j,(i,im)in enumerate(frames.items()):sheet.paste(im.resize((720,400)),((j%2)*720,(j//2)*400))
 sheet.save(out/(label+'_frames.jpg'))
 # Detailed endpoint/force/plan-clearance timeline using all recorded states.
 times=np.arange(1,T+1)*dt;dist=np.linalg.norm(actual-goal,axis=1);angle=yaw(s['root_rot']).numpy()-e['goal']['yaw'];angle=np.abs(np.degrees(np.arctan2(np.sin(angle),np.cos(angle))))
 idx=np.arange(0,T,6);margins=np.array([sdf(e,world(e,e['predicted_root'][k,:30],k)).min()for k in idx]);nomarg=np.array([sdf(e,world(e,e.get('nominal_root',e['predicted_root'])[k,:30],k)).min()for k in idx])
 fig,axes=plt.subplots(4,1,figsize=(10,10),sharex=True);axes[0].plot(times,dist,label='goal distance m');axes[0].axhline(1.25,color='gray',ls='--');axes[0].legend();axes[1].plot(times,angle,label='heading error deg');axes[1].axhline(45,color='gray',ls='--');axes[1].legend();axes[2].plot(times,e['obstacle_contact_force_n'].numpy(),label='contact force proxy N');axes[2].legend();axes[3].plot(times[idx],nomarg,label='nominal root clearance');axes[3].plot(times[idx],margins,label='selected root clearance');axes[3].axhline(.5,color='gray',ls='--',label='planning margin');axes[3].axhline(0,color='red',ls=':');axes[3].legend();axes[3].set_xlabel('physical time s');fig.suptitle(label+' | full saved episode');fig.tight_layout();fig.savefig(out/(label+'_timeline.png'),dpi=140);plt.close(fig)
 # Decode output completely, not just container header.
 verify=imageio.get_reader(target);decoded=sum(1 for _ in verify);verify.close();assert decoded==count
 manifest.append(dict(file=target.name,frames=decoded,duration_s=decoded/30,trajectory_file=rr['trajectory_file'],scene_id=int(rr['scene_id']),method=rr['method'],success=rr['success'],frame_sheet=label+'_frames.jpg',timeline=label+'_timeline.png'));print(label,decoded,flush=True)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
