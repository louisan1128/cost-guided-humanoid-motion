"""Local-frame physical state/action sequences; no simulator or teacher dependencies."""
import math
import torch
from torch import nn
from .direct_action import alpha_schedule, noisy_action


def yaw(q):
    x,y,z,w=q.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


def rotate_z(v, angle):
    c,s=angle.cos(),angle.sin()
    while c.ndim<v.ndim-1:c=c.unsqueeze(-1);s=s.unsqueeze(-1)
    return torch.stack((c*v[...,0]-s*v[...,1],s*v[...,0]+c*v[...,1],v[...,2]),-1)


def quat_matrix(q):
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    x,y,z,w=q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(*q.shape[:-1],3,3)


def local_state(states, origin, angle):
    """Root XY canonicalization, 6D rotation columns, root velocities, q and dq."""
    p=states['root_pos'].clone();p[...,:2]-=origin[...,:2]
    p=rotate_z(p,-angle)
    r=quat_matrix(states['root_rot'])
    # Left multiplication by world-to-anchor yaw; each matrix column is a vector.
    r=rotate_z(r.transpose(-1,-2),-angle).transpose(-1,-2)
    rot6=r[...,:,:2].transpose(-1,-2).flatten(-2)
    return torch.cat((p,rot6,rotate_z(states['root_vel'],-angle),
                      rotate_z(states['root_ang_vel'],-angle),states['q'],states['dq']),-1)


def valid_starts(length, horizon, history=4):
    # Left repeated history is permitted; future padding is never permitted.
    return torch.arange(max(0,length-horizon+1),dtype=torch.long)


class PlanExecutionBuffer:
    """Cache action chunks; observations must still be recorded at control frequency."""
    def __init__(self, execute_steps=1):
        if execute_steps < 1:raise ValueError('execute_steps must be positive')
        self.execute_steps=execute_steps;self.sequence=None;self.age=0

    @property
    def needs_plan(self):
        return self.sequence is None or self.age>=self.execute_steps

    def set(self,sequence):
        if self.execute_steps>sequence.shape[1]:raise ValueError('Execution exceeds prediction horizon')
        self.sequence=sequence;self.age=0

    def take(self):
        if self.needs_plan:raise ValueError('A fresh plan is required')
        action=self.sequence[:,self.age,:28];self.age+=1
        return action


def heading_guidance_cost(physical, nominal_endpoint, offset_rad):
    """Offset terminal heading relative to the unguided sample; preserve forward progress."""
    end=physical[:,-1,28:];base=nominal_endpoint[:,28:]
    angle=torch.atan2(end[:,4],end[:,3])
    desired=torch.atan2(base[:,4],base[:,3])+offset_rad
    return 1-torch.cos(angle-desired)+.25*(end[:,0]-base[:,0]).square()


class SequenceNet(nn.Module):
    def __init__(self,context_dim,output_dim,method,width=512):
        super().__init__();self.method=method
        self.context=nn.Sequential(nn.Linear(context_dim,1024),nn.SiLU(),nn.Linear(1024,width))
        if method=='diffusion':
            self.noise=nn.Sequential(nn.Linear(output_dim+32,width),nn.SiLU(),nn.Linear(width,width))
            self.register_buffer('freq',torch.exp(-math.log(10000)*torch.arange(16)/15))
        self.output=nn.Sequential(nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,output_dim))

    def forward(self,context,noisy=None,t=None):
        z=self.context(context)
        if self.method=='diffusion':
            angle=t[:,None].float()*self.freq
            z=z+self.noise(torch.cat((noisy,angle.sin(),angle.cos()),-1))
        return self.output(z)


def sample_sequence(model,context,alpha,noise,steps=16,guide=None,scale=0.,unit_gradient=False,guidance_updates=1):
    """DDIM, x0 prediction. Guidance differentiates predicted physical state cost."""
    if model.method=='mlp':return model(context)
    x=noise
    times=torch.linspace(len(alpha)-1,0,steps).round().long().tolist()
    for j,t in enumerate(times):
        for _ in range(guidance_updates if guide is not None and scale else 0):
            with torch.enable_grad():
                variable=x.detach().requires_grad_(True)
                clean=model(context,variable,torch.full((len(x),),t,device=x.device,dtype=torch.long))
                cost=guide(clean)
                grad=torch.autograd.grad(cost.sum(),variable)[0]
                if not torch.isfinite(grad).all():raise ValueError('Nonfinite guidance gradient')
                # Per-sample trust bound in normalized sequence space.
                grad=grad/grad.norm(dim=-1,keepdim=True).clamp_min(1e-8 if unit_gradient else 1.)
                x=(variable-scale*(1-alpha[t]).sqrt()*grad).detach()
        with torch.no_grad():
            clean=model(context,x,torch.full((len(x),),t,device=x.device,dtype=torch.long))
            if j==len(times)-1:return clean
            a,b=alpha[t],alpha[times[j+1]]
            eps=(x-a.sqrt()*clean)/(1-a).sqrt()
            x=b.sqrt()*clean+(1-b).sqrt()*eps
    raise RuntimeError('No sample')


def build_windows(episode,horizon,history=4,joint=False):
    """Construct only causal inputs; joint targets always use applied actions."""
    obs=episode['obs'];actions=episode['applied_action'];length=len(actions)
    if len(obs)!=length or len(episode['next']['q'])!=length:raise ValueError('Unaligned transitions')
    if 'done' in episode and bool((episode['done'][:-1]!=0).any()):raise ValueError('Episode crosses terminal/reset boundary')
    starts=valid_starts(length,horizon,history)
    if not len(starts):return None
    hist=(starts[:,None]+torch.arange(-history+1,1)).clamp_min(0)
    past=(starts[:,None]+torch.arange(-history+1,0)).clamp_min(0)
    past_actions=actions[past].clone();past_actions[(starts[:,None]+torch.arange(-history+1,0))<0]=0
    context=torch.cat((obs[hist].flatten(1),past_actions.flatten(1)),-1)
    future=starts[:,None]+torch.arange(horizon)
    target=actions[future]
    if joint:
        origin=episode['pre']['root_pos'][starts][:,None,:]
        angle=yaw(episode['pre']['root_rot'][starts])[:,None]
        local=local_state({k:v[future]for k,v in episode['next'].items()},origin,angle)
        target=torch.cat((target,local),-1)
    return context,target,starts
