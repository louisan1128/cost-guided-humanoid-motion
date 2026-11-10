"""Render a saved physical episode in Isaac Sim; no policy rerun or new score."""
import argparse,os,sys,json,hashlib,math
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--episode',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--seconds',type=float,default=10);p.add_argument('--mimickit-root',type=Path,default=Path('/home/an/Desktop/MimicKit-main'));a=p.parse_args()
 src=a.episode.resolve();out=a.out_dir.resolve();out.mkdir(parents=True,exist_ok=False);e=torch.load(src,map_location='cpu',weights_only=False);assert e['row']['success'];n=min(round(a.seconds/e['dt']),len(e['next']['q']));g=e['geometry']
 sys.path.insert(0,str(a.mimickit_root/'mimickit'));os.chdir(a.mimickit_root)
 from engines.isaac_lab_engine import IsaacLabEngine
 from engines.engine import ObjType
 eng=IsaacLabEngine(dict(control_mode='pos',control_freq=30,sim_freq=120,env_spacing=5),1,'cuda:0',False,True)
 try:
  eng.create_env();cid=eng.create_obj(0,ObjType.articulated,'data/assets/humanoid/humanoid.xml','character',color=np.array([.35,.48,.75]))
  import isaaclab.sim as su
  from pxr import UsdGeom,Gf
  xy=np.array(e['obstacle_xy']);half=g['half_extents_m'];angle=g['angle_world_rad']
  wall=su.CuboidCfg(size=(2*half[0],2*half[1],1.5),visual_material=su.PreviewSurfaceCfg(diffuse_color=(.8,.22,.12)))
  wall.func('/World/ReplayWall',wall,translation=(float(xy[0]),float(xy[1]),.75),orientation=(math.cos(angle/2),0,0,math.sin(angle/2)))
  goal=np.array(e['goal']['xy']);start=e['pre']['root_pos'][0,:2].numpy();direction=(goal-start);direction/=np.linalg.norm(direction);side=np.array([-direction[1],direction[0]])
  # Goal ring and final-heading arrow are visual annotations, never physics controls.
  theta=np.linspace(0,2*np.pi,65);ring=np.c_[goal[0]+.23*np.cos(theta),goal[1]+.23*np.sin(theta),np.full(65,.025)]
  yaw=e['goal']['yaw'];tip=goal+.7*np.array([math.cos(yaw),math.sin(yaw)]);v=np.array([math.cos(yaw),math.sin(yaw)]);normal=np.array([-v[1],v[0]])
  points=np.vstack([ring,np.r_[goal,.025],np.r_[tip,.025],np.r_[tip-.15*v+.09*normal,.025],np.r_[tip,.025],np.r_[tip-.15*v-.09*normal,.025]])
  curves=UsdGeom.BasisCurves.Define(eng._stage,'/World/GoalMark');curves.CreateTypeAttr('linear');curves.CreateCurveVertexCountsAttr([65,2,3]);curves.CreatePointsAttr([Gf.Vec3f(*map(float,x))for x in points]);curves.CreateWidthsAttr([.025]);curves.CreateDisplayColorAttr([Gf.Vec3f(.95,.65,.03)])
  eng.initialize_sim()
  import omni.replicator.core as rep
  product=rep.create.render_product('/OmniverseKit_Persp',(1280,720));rgb=rep.AnnotatorRegistry.get_annotator('rgb',device='cpu');rgb.attach([product])
  middle=(start+goal)/2;eye=middle+side*8-direction*2
  eng.set_camera_pose(np.r_[eye,5.5],np.r_[middle,.65])
  def place(i):
   s=e['next'];
   for key,fn in [('root_pos','set_root_pos'),('root_rot','set_root_rot'),('root_vel','set_root_vel'),('root_ang_vel','set_root_ang_vel'),('q','set_dof_pos'),('dq','set_dof_vel')]:getattr(eng,fn)(None,cid,s[key][i:i+1].cuda())
   eng._update_reset_objs()
  place(0)
  for _ in range(40):eng._sim.render()
  import imageio.v2 as imageio
  video=out/'wall_scene11_isaac_10s.mp4'
  with imageio.get_writer(video,fps=30,codec='libx264',quality=8,macro_block_size=16,output_params=['-pix_fmt','yuv420p','-movflags','+faststart'])as writer:
   for i in range(n):
    place(i);eng._sim.render();frame=np.asarray(rgb.get_data())[:,:,:3].copy()
    if not frame.size or frame.std()<2:raise RuntimeError('Empty render')
    writer.append_data(frame)
    if i in [0,75,150,225,n-1]:imageio.imwrite(out/f'frame_{i:03}.jpg',frame)
    if i%60==0:print('frame',i,flush=True)
  (out/'manifest.json').write_text(json.dumps(dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),frames=n,fps=30,duration_s=n/30,renderer='Isaac Sim RTX via Isaac Lab; recorded-state playback',new_physics_evaluation=False,new_training=False,source_result=e['row'],geometry=g,video=video.name),indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)));print('COMPLETE',video,flush=True);rgb.detach([product])
 finally:eng._app_launcher.app.close()
if __name__=='__main__':main()
