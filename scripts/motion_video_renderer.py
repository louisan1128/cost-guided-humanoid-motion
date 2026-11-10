"""CPU orthographic MJCF-geometry renderer for saved motion, without simulation."""
import xml.etree.ElementTree as ET
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageColor
from scipy.spatial.transform import Rotation


def load_geometry(asset, names):
    shapes=[]
    for body in ET.parse(asset).getroot().iter('body'):
        for geom in body.findall('geom'):
            number=names.index(body.attrib['name'])
            kind=geom.get('type','sphere')
            size=np.fromstring(geom.get('size',''),sep=' ')
            local=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
            if kind=='capsule':local=np.fromstring(geom.attrib['fromto'],sep=' ').reshape(2,3)
            elif kind=='box':
                local=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]])*size+local
            else:local=local[None]
            shapes.append(dict(body=number,name=body.attrib['name'],kind=kind,size=size,local=local))
    return shapes


def obstacle_outline(obstacle):
    """Physical footprint vertices, retaining legacy circle dictionaries."""
    x,y=obstacle['xy']
    if obstacle.get('kind','circle')=='box':
        hx,hy=obstacle['half'];a=obstacle.get('angle',0.)
        local=np.array([[-hx,-hy],[hx,-hy],[hx,hy],[-hx,hy],[-hx,-hy]])
        rotation=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        return local@rotation.T+np.array([x,y])
    theta=np.linspace(0,2*np.pi,65);r=obstacle['radius']
    return np.c_[x+r*np.cos(theta),y+r*np.sin(theta)]


def render_geometry(path, positions, rotations, times, title, subtitle, names, color, geometry, goal=None, obstacles=None, bounds_points=None):
    width,height,fps=1280,800,30
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,25);small=ImageFont.truetype(fontpath,18)
    right=np.array([.857493,.514496,0.]);up=np.array([-.23538,.39230,.88976]);up/=np.linalg.norm(up)
    view=np.cross(right,up)
    matrices=Rotation.from_quat(rotations.reshape(-1,4)).as_matrix().reshape(*rotations.shape[:-1],3,3)
    world=[np.einsum('nij,kj->nki',matrices[:,g['body']],g['local'])+positions[:,g['body'],None] for g in geometry]
    bounds=np.concatenate(world,axis=1).reshape(-1,3)
    for obstacle in obstacles or []:
        outline=obstacle_outline(obstacle);h=obstacle['height']
        bounds=np.vstack([bounds,np.c_[outline,np.zeros(len(outline))],np.c_[outline,np.full(len(outline),h)]])
    if goal is not None:
        bounds=np.vstack([bounds,np.r_[goal['xy'],0.],np.r_[goal['xy'],0.]+[0,0,.5]])
        if 'position_tolerance_m' in goal:
            radius=float(goal['position_tolerance_m'])
            bounds=np.vstack([bounds,[[goal['xy'][0]+dx,goal['xy'][1]+dy,0.]
                                      for dx in [-radius,radius] for dy in [-radius,radius]]])
    if bounds_points is not None:
        bounds=np.vstack([bounds,np.asarray(bounds_points).reshape(-1,3)])
    projected=np.c_[bounds@right,bounds@up]
    lo=projected.min(0)-.3;hi=projected.max(0)+.3
    center=(lo+hi)/2
    scale=min(360.,(width-160)/(hi[0]-lo[0]),(height-230)/(hi[1]-lo[1]))
    def project(points):
        points=np.asarray(points)
        return np.stack([width/2+scale*(points@right-center[0]),425-scale*(points@up-center[1])],axis=-1)
    display_times=times[0]+np.arange(int(round((times[-1]-times[0])*fps))+1)/fps
    after=np.searchsorted(times,display_times).clip(0,len(times)-1);before=np.maximum(after-1,0)
    indices=np.where(abs(times[before]-display_times)<abs(times[after]-display_times),before,after)
    base=Image.new('RGB',(width,height),'#edf1f5');draw=ImageDraw.Draw(base)
    lower=bounds[:,:2].min(0)-1;upper=bounds[:,:2].max(0)+1
    for axis in range(2):
        for value in np.arange(np.floor(lower[axis]),np.ceil(upper[axis])+.25,.25):
            ends=np.c_[np.vstack([lower,upper]),np.zeros(2)];ends[:,axis]=value
            draw.line([tuple(p) for p in project(ends)],fill='#d1d9e2',width=1)
    def arrow(draw,start,end,fill,width=3):
        a,b=project([start,end]);d=b-a;length=np.linalg.norm(d)
        if length<1:return
        d/=length;side=np.array([-d[1],d[0]])
        draw.line([tuple(a),tuple(b)],fill=fill,width=width)
        draw.polygon([tuple(b),tuple(b-12*d+5*side),tuple(b-12*d-5*side)],fill=fill)
    if goal is not None:
        point=np.r_[goal['xy'],.02];theta=np.linspace(0,2*np.pi,65)
        ring=point+np.c_[.22*np.cos(theta),.22*np.sin(theta),np.zeros(65)]
        draw.line([tuple(p) for p in project(ring)],fill='#bf8300',width=4)
        if 'position_tolerance_m' in goal:
            radius=float(goal['position_tolerance_m'])
            outer=point+np.c_[radius*np.cos(theta),radius*np.sin(theta),np.zeros(65)]
            draw.line([tuple(p) for p in project(outer)],fill='#d8ba7c',width=2)
            for delta in [-1,1]:
                angle=goal['yaw']+delta*np.deg2rad(goal.get('yaw_tolerance_deg',25.))
                draw.line([tuple(p) for p in project([point,point+.65*np.array([np.cos(angle),np.sin(angle),0.])])],fill='#d8ba7c',width=2)
        arrow(draw,point,point+.65*np.array([np.cos(goal['yaw']),np.sin(goal['yaw']),0]),'#bf8300',4)
    rgb=np.array(ImageColor.getrgb(color))
    def shade(factor):return tuple(np.clip(rgb*factor,0,255).astype(int))
    boxfaces=[[0,1,3,2],[4,6,7,5],[0,4,5,1],[2,3,7,6],[0,2,6,4],[1,5,7,3]]
    with imageio.get_writer(path,fps=fps,codec='libx264',quality=8,macro_block_size=16,output_params=['-movflags','+faststart']) as writer:
        for frame,(stamp,i) in enumerate(zip(display_times,indices)):
            im=base.copy();draw=ImageDraw.Draw(im)
            trail=positions[:i+1:2,0].copy();trail[:,2]=.01
            if len(trail)>1:draw.line([tuple(p) for p in project(trail)],fill='#98a6b6',width=3)
            for obstacle in obstacles or []:
                outline=obstacle_outline(obstacle);h=obstacle['height']
                bottom=np.c_[outline,np.zeros(len(outline))]
                top=bottom+np.array([0,0,h]);bottom,top=project(bottom),project(top)
                for k in range(len(bottom)-1):
                    draw.polygon([tuple(bottom[k]),tuple(bottom[k+1]),tuple(top[k+1]),tuple(top[k])],fill='#d96d58')
                draw.polygon([tuple(p) for p in top],fill='#ec9986',outline='#913c2b')
            # Ground projections make foot height and body displacement visible.
            for name in ['left_foot','right_foot']:
                point=positions[i,names.index(name)].copy();point[2]=.006
                x,y=project(point);r=max(2,int(.055*scale))
                draw.ellipse((x-r,y-r/2,x+r,y+r/2),fill='#c5ced8')
            order=sorted(range(len(geometry)),key=lambda k:world[k][i].mean(0)@view)
            for k in order:
                g=geometry[k];points=world[k][i];screen=project(points)
                if g['kind']=='box':
                    for face in sorted(boxfaces,key=lambda face:points[face].mean(0)@view):
                        p=points[face];normal=np.cross(p[1]-p[0],p[2]-p[0]);normal/=np.linalg.norm(normal)
                        intensity=.65+.4*abs(normal@np.array([.3,-.4,.866]))
                        draw.polygon([tuple(p) for p in screen[face]],fill=shade(intensity),outline=shade(.5))
                elif g['kind']=='capsule':
                    radius=max(2,int(g['size'][0]*scale));a,b=screen
                    draw.line([tuple(a),tuple(b)],fill=shade(.62),width=2*radius)
                    for point in screen:
                        x,y=point;draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=shade(.62))
                    draw.line([tuple(a-[radius*.2,radius*.2]),tuple(b-[radius*.2,radius*.2])],fill=shade(1.12),width=max(2,radius))
                else:
                    x,y=screen[0];radius=max(2,int(g['size'][0]*scale))
                    draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=shade(.65))
                    for layer in range(1,9):
                        r=radius*(1-layer*.07);offset=-radius*layer*.025
                        draw.ellipse((x+offset-r,y+offset-r,x+offset+r,y+offset+r),fill=shade(.65+layer*.065))
            if goal is not None:
                root=positions[i,0];forward=matrices[i,0]@np.array([1.,0.,0.]);forward[2]=0
                norm=np.linalg.norm(forward)
                if norm>1e-6:arrow(draw,np.r_[root[:2],.04],np.r_[root[:2],.04]+.4*forward/norm,'#345a96')
            for obstacle in obstacles or []:
                x,y=obstacle['xy'];h=obstacle['height'];outline=obstacle_outline(obstacle)
                footprint=np.c_[outline,np.full(len(outline),.015)]
                draw.line([tuple(p) for p in project(footprint)],fill='#b32b1c',width=3)
                label=project([x,y,h+.15])
                draw.text((label[0]-48,label[1]-20),'OBSTACLE',font=small,fill='#a32518',stroke_width=2,stroke_fill='white')
            draw.rectangle((0,0,width,112),fill='white')
            draw.text((26,14),title,fill='#172338',font=font)
            draw.text((26,51),subtitle,fill='#34445b',font=small)
            draw.text((26,79),f'{stamp-times[0]:.2f} s  |  1x speed / 30 fps  |  fixed camera / original MJCF geometry',fill='#34445b',font=small)
            draw.rectangle((0,height-48,width,height),fill='white')
            note='Yellow: goal and heading. Blue: body heading. Gray: traveled path.' if goal else 'Recorded training motion. Geometry replay; no new physics or motion smoothing.'
            if goal is not None and 'position_tolerance_m' in goal:
                note=f"Outer ring: {goal['position_tolerance_m']:g}m / heading tolerance: +/-{goal.get('yaw_tolerance_deg',25.):g}deg. Blue: body heading."
            if obstacles:
                note='Red: physical obstacle at recorded dimensions. Yellow: goal. Recorded actual motion.'
            draw.text((26,height-34),note,fill='#34445b',font=small)
            if frame in [0,len(indices)//2,len(indices)-1]:im.save(path.with_name(path.stem+f'_{frame:05d}.png'))
            if frame==len(indices)//2:im.save(path.with_suffix('.png'))
            writer.append_data(np.asarray(im))
    return dict(frames=len(indices),duration_s=len(indices)/fps,renderer='MJCF geometry, fixed camera, nearest native sample, 30fps')
