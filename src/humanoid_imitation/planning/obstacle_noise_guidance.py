"""Single-sample motion diffusion costs and bounded initial-noise optimization."""
import torch


def cost_terms(root, nominal, center, radius=.65):
    distance=(root[..., :2]-center[:,None,:]).norm(dim=-1)
    risk=(torch.relu(radius-distance)/.2).square()
    collision=risk.topk(min(5,root.shape[1]),dim=1).values.mean(1)
    velocity=torch.diff(root[...,:2],dim=1)*30
    consistency=((root[:,1:,9:11]-velocity)/.5).square().mean((1,2))
    heading=root[...,3:5];heading=heading/heading.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    speed=root[...,9:11].norm(dim=-1);course=(heading*root[...,9:11]).sum(-1)/speed.clamp_min(.05)
    course=((1-course)*(speed.detach()/.5).clamp(max=1)).mean(1)
    direction=nominal[:,-1,:2];direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(.05)
    progress=torch.relu(((nominal[:,-1,:2]-root[:,-1,:2])*direction).sum(-1)-.4).square()
    return dict(collision=collision,consistency=consistency,course=course,progress=progress)


def obstacle_motion_cost(root, nominal, center):
    terms=cost_terms(root,nominal,center)
    return 10*terms['collision']+terms['consistency']+.5*terms['course']+terms['progress']


def ddim_differentiable(model, context, alpha, noise, steps=16):
    """Same deterministic x0 DDIM equations as sample_sequence, retaining gradients."""
    if model.method!='diffusion':return model(context)
    x=noise;times=torch.linspace(len(alpha)-1,0,steps).round().long().tolist()
    for j,t in enumerate(times):
        clean=model(context,x,torch.full((len(x),),t,device=x.device,dtype=torch.long))
        if j==len(times)-1:return clean
        a,b=alpha[t],alpha[times[j+1]]
        eps=(x-a.sqrt()*clean)/(1-a).sqrt()
        x=b.sqrt()*clean+(1-b).sqrt()*eps
    raise ValueError('Empty schedule')


def optimize_noise(model, context, alpha, noise, cost, iterations=32, lr=.1, radius=1., steps=16):
    """Optimize ONE initial sample; retain per-item best cost inside RMS trust region.

    This is inference optimization, not model training or K-candidate selection.
    """
    with torch.enable_grad():
        variable=noise.detach().clone().requires_grad_();optimizer=torch.optim.Adam([variable],lr=lr)
        with torch.no_grad():
            initial=ddim_differentiable(model,context,alpha,variable,steps);best_cost=cost(initial).detach();best=initial.clone();best_noise=variable.detach().clone()
        for _ in range(iterations):
            prediction=ddim_differentiable(model,context,alpha,variable,steps);value=cost(prediction)
            if not torch.isfinite(value).all():raise ValueError('Nonfinite optimization cost')
            grad=torch.autograd.grad(value.sum(),variable)[0]
            optimizer.zero_grad();variable.grad=grad;optimizer.step()
            with torch.no_grad():
                delta=variable-noise;norm=delta.square().mean(-1,keepdim=True).sqrt()
                variable.copy_(noise+delta*(radius/norm.clamp_min(1e-8)).clamp(max=1))
                prediction=ddim_differentiable(model,context,alpha,variable,steps);new_cost=cost(prediction);improved=new_cost<best_cost
                best=torch.where(improved[:,None],prediction,best);best_noise=torch.where(improved[:,None],variable,best_noise);best_cost=torch.minimum(best_cost,new_cost)
    return best.detach(),best_noise.detach(),best_cost.detach()
