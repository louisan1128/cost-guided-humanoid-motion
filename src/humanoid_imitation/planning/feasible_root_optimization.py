"""Route-free output optimization with bounded integrated root kinematics.

Bounds are kinematic safeguards, not a proof of full-body dynamic feasibility.
No waypoint, avoidance side or analytic detour is supplied.
"""
import torch
import torch.nn.functional as F
from .state_action_sequence import rotate_z

LIMITS = dict(speed=2., yaw_rate=1.5, acceleration=6., yaw_acceleration=8.)


def ball(x, radius):
    return x * (radius / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)


def feasibility_cost(root, initial_velocity, initial_yaw_rate, dt=1/30):
    v=root[...,9:11]; w=root[...,14]
    acceleration=torch.diff(torch.cat((initial_velocity[:,None],v),1),dim=1)/dt
    angular=torch.diff(torch.cat((initial_yaw_rate[:,None],w),1),dim=1)/dt
    speed=torch.relu(v.norm(dim=-1)/LIMITS['speed']-1).square().mean(1)
    spin=torch.relu(w.abs()/LIMITS['yaw_rate']-1).square().mean(1)
    accel=torch.relu(acceleration.norm(dim=-1)/LIMITS['acceleration']-1).square().mean(1)
    yaw_accel=torch.relu(angular.abs()/LIMITS['yaw_acceleration']-1).square().mean(1)
    # Common objective for MLP output optimization and diffusion guidance.
    return speed+spin+.1*(accel+yaw_accel)


def integrated_root(nominal, controls, initial_velocity, initial_yaw_rate, dt=1/30):
    dense=F.interpolate(controls.transpose(1,2),size=nominal.shape[1],mode='linear',align_corners=True).transpose(1,2)
    accel=ball(dense[...,:2],LIMITS['acceleration'])
    yaw_accel=dense[...,2].clamp(-LIMITS['yaw_acceleration'],LIMITS['yaw_acceleration'])
    # Projection on the same convex ball is non-expansive: acceleration bounds
    # remain valid after projection if initial velocity is inside the ball.
    velocity=ball(initial_velocity[:,None]+dt*accel.cumsum(1),LIMITS['speed'])
    angular=(initial_yaw_rate[:,None]+dt*yaw_accel.cumsum(1)).clamp(-LIMITS['yaw_rate'],LIMITS['yaw_rate'])
    xy=dt*velocity.cumsum(1); heading=dt*angular.cumsum(1)
    old_heading=torch.atan2(nominal[...,4],nominal[...,3]); change=heading-old_heading
    c1=rotate_z(nominal[...,3:6],change); c2=rotate_z(nominal[...,6:9],change)
    return torch.cat((xy,nominal[...,2:3],c1,c2,velocity,nominal[...,11:14],angular[...,None],nominal[...,15:]),-1)


def optimize_feasible_root(nominal,cost,active,initial_velocity,initial_yaw_rate,lr=.1,steps=32):
    if not bool(active.any()): return nominal.detach()
    nominal=nominal.detach()
    with torch.enable_grad():
        acc=torch.diff(torch.cat((initial_velocity[:,None],nominal[...,9:11]),1),dim=1)*30
        angular=torch.diff(torch.cat((initial_yaw_rate[:,None],nominal[...,14]),1),dim=1)*30
        controls=torch.cat((ball(acc,LIMITS['acceleration']),angular.clamp(-LIMITS['yaw_acceleration'],LIMITS['yaw_acceleration'])[...,None]),-1)
        controls=F.interpolate(controls.transpose(1,2),size=9,mode='linear',align_corners=True).transpose(1,2).detach().requires_grad_(True)
        opt=torch.optim.Adam([controls],lr=lr)
        with torch.no_grad():
            best=integrated_root(nominal,controls,initial_velocity,initial_yaw_rate);best_cost=cost(best)
        for _ in range(steps):
            root=integrated_root(nominal,controls,initial_velocity,initial_yaw_rate)
            grad=torch.autograd.grad((cost(root)*active).sum(),controls)[0]
            if not torch.isfinite(grad).all():raise ValueError('Nonfinite feasible-root gradient')
            opt.zero_grad();controls.grad=grad;opt.step()
            with torch.no_grad():
                candidate=integrated_root(nominal,controls,initial_velocity,initial_yaw_rate); value=cost(candidate)
                if not torch.isfinite(value).all():raise ValueError('Nonfinite feasible-root cost')
                improved=active & (value<best_cost)
                best=torch.where(improved[:,None,None],candidate,best);best_cost=torch.minimum(best_cost,value)
    return torch.where(active[:,None,None],best,nominal).detach()
