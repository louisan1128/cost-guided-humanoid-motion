"""Direct PD-action baselines. Independent of the motion-reference generators.

Both policies observe the teacher's current observation and produce one action.
Diffusion uses clean-action prediction and deterministic DDIM conditional on
one sampled Gaussian candidate. No retrieval, tracker, or terminal action bank.
"""
import math
import torch
from torch import nn


class DirectMLP(nn.Module):
    def __init__(self, obs_dim=144, action_dim=28):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 1024), nn.ReLU(),
                                 nn.Linear(1024, 512), nn.ReLU(), nn.Linear(512, action_dim))

    def forward(self, obs):
        return self.net(obs)


class ActionDenoiser(nn.Module):
    def __init__(self, obs_dim=144, action_dim=28):
        super().__init__()
        self.context = nn.Sequential(nn.Linear(obs_dim, 1024), nn.ReLU(), nn.Linear(1024, 512))
        self.noisy = nn.Sequential(nn.Linear(action_dim+32, 128), nn.SiLU(), nn.Linear(128, 512))
        self.output = nn.Sequential(nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, action_dim))
        self.register_buffer('frequencies', torch.exp(-math.log(10000)*torch.arange(16)/15))

    def forward(self, obs, noisy, t):
        angle = t.float()[:, None]*self.frequencies
        return self.output(self.context(obs)+self.noisy(torch.cat((noisy,angle.sin(),angle.cos()),-1)))


def alpha_schedule(steps=100):
    t = torch.linspace(0, 1, steps+1, dtype=torch.float64)
    f = torch.cos((t+.008)/1.008*math.pi/2).square()
    beta = (1-f[1:]/f[:-1]).clamp(max=.999)
    return torch.cumprod(1-beta, 0).float()


def noisy_action(clean, noise, t, alpha):
    a = alpha[t, None]
    return a.sqrt()*clean+(1-a).sqrt()*noise


@torch.no_grad()
def sample_action(model, obs, alpha, noise, steps=16):
    if not 2 <= steps <= len(alpha): raise ValueError('Invalid denoising steps')
    x = noise.clone()
    times = torch.linspace(len(alpha)-1, 0, steps).round().long().tolist()
    for j,t in enumerate(times):
        clean = model(obs, x, torch.full((len(x),),t,device=x.device,dtype=torch.long))
        if j==len(times)-1: return clean
        a, previous = alpha[t], alpha[times[j+1]]
        eps = (x-a.sqrt()*clean)/(1-a).sqrt()
        x = previous.sqrt()*clean+(1-previous).sqrt()*eps
    raise RuntimeError('Empty sampler')


class DirectPolicy:
    def __init__(self, path, device='cuda', seed=0):
        c = torch.load(path,map_location=device,weights_only=False)
        self.method=c['method']; self.stats=c['stats']; self.device=device
        cls=DirectMLP if self.method=='mlp' else ActionDenoiser
        self.model=cls(c['obs_dim'],c['action_dim']).to(device)
        self.model.load_state_dict(c['model']);self.model.eval()
        self.rng=torch.Generator(device=device).manual_seed(seed)
        self.alpha=alpha_schedule().to(device);self.action_dim=c['action_dim']
        self.steps=c['inference_steps'];self.last_latency_ms=0.

    @torch.no_grad()
    def __call__(self, obs):
        x=((obs-self.stats['obs_mean'])/self.stats['obs_std']).clamp(-10,10)
        if self.method=='mlp': y=self.model(x)
        else:
            noise=torch.randn(len(x),self.action_dim,device=self.device,generator=self.rng)
            y=sample_action(self.model,x,self.alpha,noise,self.steps)
        return y*self.stats['action_std']+self.stats['action_mean']
