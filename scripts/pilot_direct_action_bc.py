#!/usr/bin/env python3
"""Bounded teacher-action imitation pilot. No learned reference tracker.

The teacher's existing sequential goal observation supervisor remains in use.
Outputs are physical PD joint targets, not motion frames or normalized actions.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEACHER = ROOT / 'outputs/goal_teacher_sequential_v31_300/model.pt'
STATE_METHODS = dict(q='dof_pos', dq='dof_vel', root_pos='root_pos',
                     root_rot='root_rot', root_vel='root_vel', root_ang_vel='root_ang_vel')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + '\n')


class ActionMLP(nn.Module):
    def __init__(self, obs_dim, action_dim, width=512):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(), nn.Linear(width, action_dim))

    def forward(self, x):
        return self.net(x)


def normalize(obs, stats):
    return ((obs - stats['obs_mean']) / stats['obs_std']).clamp(-10, 10)


def train(args):
    start = time.monotonic()
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    data = torch.load(args.root / 'collection/data.pt', map_location='cpu', weights_only=False)
    device = args.device
    stats = {k: v.to(device) for k, v in data['stats'].items()}
    # Entire episodes are assigned before simulation. No window-level split.
    train_mask = data['train_mask'] & data['successful_episode']
    val_mask = ~data['train_mask'] & data['successful_episode']
    if min(int(train_mask.sum()), int(val_mask.sum())) == 0:
        raise ValueError('No successful teacher episodes in a split')
    x = normalize(data['obs'][train_mask].to(device), stats)
    y = (data['action'][train_mask].to(device) - stats['action_mean']) / stats['action_std']
    vx = normalize(data['obs'][val_mask].to(device), stats)
    vy = (data['action'][val_mask].to(device) - stats['action_mean']) / stats['action_std']
    model = ActionMLP(x.shape[1], y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    out = args.root / 'bc'; out.mkdir(exist_ok=False)
    best = float('inf'); logs = []
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.
        for ids in torch.randperm(len(x), device=device).split(1024):
            loss = (model(x[ids]) - y[ids]).square().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step(); total += float(loss.detach()) * len(ids)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(b) for b in vx.split(2048)])
            val = float((pred - vy).square().mean())
            rmse = float(((pred - vy) * stats['action_std']).square().mean().sqrt())
        row = dict(epoch=epoch, train_normalized_mse=total / len(x),
                   validation_normalized_mse=val, validation_action_rmse_rad=rmse)
        logs.append(row)
        if val < best:
            best = val
            torch.save(dict(model={k: v.cpu().clone() for k, v in model.state_dict().items()},
                            stats={k: v.cpu() for k, v in stats.items()}, obs_dim=x.shape[1],
                            action_dim=y.shape[1], width=512, epoch=epoch, seed=args.seed,
                            role='scratch single-action behavior cloning; physical PD targets',
                            teacher_sha256=sha(TEACHER)), out / 'model.pt')
        if epoch % 10 == 0: print(json.dumps(row), flush=True)
    write(out / 'learning_curve.json', logs)
    write(out / 'result.json', dict(status='complete', train_samples=len(x), validation_samples=len(vx),
          best_validation_normalized_mse=best, epochs=args.epochs, wall_s=time.monotonic()-start,
          model_sha256=sha(out/'model.pt'), optimizer_resume=False, initialization='random',
          normalization='frozen teacher observation/action statistics; no new validation fitting',
          selection='lowest held-out episode action MSE; physical evaluation not used for selection'))


def simulate(args):
    sys.path[:0] = [str(ROOT / 'src'), '/home/an/Desktop/MimicKit-main/mimickit']
    from humanoid_imitation.envs.goal_amp_env import GoalAMPEnv
    from humanoid_imitation.planning.arrival_success import ArrivalHoldMonitor
    from util import torch_util, mp_util, util
    from learning import agent_builder, base_agent
    from envs.base_env import DoneFlags
    protocol = json.loads((args.root / 'protocol.json').read_text())
    for path, digest in protocol['frozen'].items():
        if sha(path) != digest: raise ValueError('Changed frozen input ' + path)
    collect = args.mode == 'collect'
    out = args.root / ('collection' if collect else 'student_evaluation')
    out.mkdir(exist_ok=False)
    cfg = yaml.safe_load((ROOT / 'configs/goal_teacher/view_sequential_arrival_v34.yaml').read_text())
    motions = yaml.safe_load((ROOT / cfg['motion_file']).read_text())
    for row in motions['motions']: row['file'] = str((ROOT / row['file']).resolve())
    (out / 'motions.yaml').write_text(yaml.safe_dump(motions))
    cfg['motion_file'] = str(out / 'motions.yaml'); cfg['episode_length'] = protocol['seconds'] + 1
    for k in ('interactive_goal', 'fixed_test_goal', 'measured_standing_reset', 'stand_alignment_rehearsal'):
        cfg['goal'].pop(k, None)
    if cfg['enable_tar_obs'] or cfg['enable_phase_obs']:
        raise ValueError('Teacher policy unexpectedly observes reference or phase')
    (out / 'env.yaml').write_text(yaml.safe_dump(cfg))
    cases = protocol['cases'] if collect else [c for c in protocol['cases'] if c['split']=='validation']
    states = None
    if not collect:
        saved = torch.load(args.root/'collection/initial.pt', weights_only=False)
        ids = [c['id'] for c in cases]
        states = {k: v[ids].to('cuda') for k, v in saved.items()}
    os.chdir('/home/an/Desktop/MimicKit-main')
    mp_util.init(0, 1, 'cuda:0', 6351); util.set_rand_seed(args.seed)
    torch.set_num_threads(4)
    class PilotEnv(GoalAMPEnv):
        def _ref_state_init(self, ids):
            super()._ref_state_init(ids)
            if states is not None:
                for key, method in STATE_METHODS.items():
                    getattr(self._engine, 'set_'+method)(ids, self._get_char_id(), states[key][ids])
    start = time.monotonic()
    env = PilotEnv(env_config=cfg, engine_config=yaml.safe_load((ROOT/'configs/engines/isaac_lab.yaml').read_text()),
                   num_envs=len(cases), device='cuda:0', visualize=False)
    agent = agent_builder.build_agent(str(ROOT/'configs/goal_teacher/agent_sequential_v31.yaml'), env, 'cuda:0')
    agent.load(str(TEACHER)); agent.eval(); agent.set_mode(base_agent.AgentMode.TEST)
    stats = dict(obs_mean=agent._obs_norm.get_mean().detach(), obs_std=agent._obs_norm.get_std().detach(),
                 action_mean=agent._a_norm.get_mean().detach(), action_std=agent._a_norm.get_std().detach())
    model = None
    if not collect:
        ckpt = torch.load(args.root/'bc/model.pt', map_location='cuda', weights_only=False)
        model = ActionMLP(ckpt['obs_dim'], ckpt['action_dim'], ckpt['width']).cuda()
        model.load_state_dict(ckpt['model']); model.eval(); stats = ckpt['stats']
    cid = env._get_char_id(); dt = float(env._engine.get_timestep()); n = len(cases)
    monitors = [ArrivalHoldMonitor(position_m=1.25, yaw_deg=45, speed_mps=.15,
                yaw_rate_radps=.2, hold_s=5., drift_m=.1) for _ in cases]
    alive = np.ones(n, dtype=bool); rows = [None]*n; records=[]; traces=[]
    with torch.no_grad():
        obs, info = env.reset()
        if states is not None: env._last_reset_standing[:] = states['standing']
        initial = {k: getattr(env._engine, 'get_'+m)(cid).clone().cpu() for k, m in STATE_METHODS.items()}
        reset_errors = {} if states is None else {k: float((initial[k]-states[k].cpu()).abs().max()) for k in STATE_METHODS}
        if reset_errors and max(reset_errors.values()) > 2e-5: raise ValueError('Initial reset mismatch')
        initial['standing'] = env._last_reset_standing.cpu().clone()
        pos = env._engine.get_root_pos(cid); heading = torch_util.calc_heading(env._engine.get_root_rot(cid))
        if collect:
            bearing = heading + torch.tensor([math.radians(c['bearing_deg']) for c in cases], device='cuda')
            distance = torch.tensor([c['distance_m'] for c in cases], device='cuda')
            env.goal_xy[:] = pos[:,:2] + distance[:,None]*torch.stack((bearing.cos(),bearing.sin()),-1)
            env.goal_yaw[:] = bearing + torch.tensor([math.radians(c['yaw_offset_deg']) for c in cases],device='cuda')
        else:
            goals = json.loads((args.root/'collection/goals.json').read_text())
            env.goal_xy[:] = torch.tensor([goals[c['id']]['xy'] for c in cases],device='cuda')
            env.goal_yaw[:] = torch.tensor([goals[c['id']]['yaw'] for c in cases],device='cuda')
        goals=[dict(xy=env.goal_xy[i].cpu().tolist(),yaw=float(env.goal_yaw[i])) for i in range(n)]
        write(out/'goals.json',goals); torch.save(initial,out/'initial.pt')
        env._refresh_changed_goal(); env._update_observations(None); obs,info=env._obs_buf,env._info
        for step in range(round(protocol['seconds']/dt)):
            if collect: action,_=agent._decide_action(obs,info)
            else: action=model(normalize(obs,stats))*stats['action_std']+stats['action_mean']
            if not torch.isfinite(action).all(): raise ValueError('Nonfinite action')
            active = torch.tensor(alive,device='cuda')
            if collect:
                records.append((obs[active].cpu().clone(),action[active].cpu().clone(),
                                torch.arange(n,device='cuda')[active].cpu()))
            action[~active]=0
            obs,_,done,info=env.step(action)
            pos=env._engine.get_root_pos(cid).cpu().numpy()
            yaw=torch_util.calc_heading(env._engine.get_root_rot(cid)).cpu().numpy()
            vel=env._engine.get_root_vel(cid).cpu().numpy();ang=env._engine.get_root_ang_vel(cid).cpu().numpy()
            traces.append(dict(q=env._engine.get_dof_pos(cid).cpu().numpy().copy(), root_pos=pos.copy(),
                               root_rot=env._engine.get_root_rot(cid).cpu().numpy().copy()))
            flags=done.cpu().numpy();t=(step+1)*dt
            for i,c in enumerate(cases):
                if not alive[i]:continue
                d=float(np.linalg.norm(pos[i,:2]-goals[i]['xy']))
                error=math.atan2(math.sin(yaw[i]-goals[i]['yaw']),math.cos(yaw[i]-goals[i]['yaw']))
                speed=float(np.linalg.norm(vel[i,:2]));wz=float(ang[i,2])
                monitors[i].update(t,d,error,speed,wz,pos[i,:2]);first=monitors[i].first_success_time_s
                fail=flags[i]==DoneFlags.FAIL.value
                stop=fail or (first is not None and t>=first+2-1e-8) or flags[i]!=DoneFlags.NULL.value or step==round(protocol['seconds']/dt)-1
                if stop:
                    alive[i]=False
                    rows[i]=dict(**c,success=bool(first is not None and not fail),fall=bool(fail),
                        duration_s=t,steps=step+1,first_success_s=first,position_error_m=d,
                        yaw_error_deg=math.degrees(abs(error)),speed_mps=speed,wz_radps=wz)
            if not alive.any():break
    if collect:
        ids=torch.cat([r[2] for r in records]);successful=torch.tensor([r['success'] for r in rows])
        torch.save(dict(obs=torch.cat([r[0] for r in records]),action=torch.cat([r[1] for r in records]),
            episode_id=ids,train_mask=ids<protocol['train_episodes'],successful_episode=successful[ids],
            stats={k:v.cpu() for k,v in stats.items()}),out/'data.pt')
    for i,c in enumerate(cases):
        length=rows[i]['steps']
        np.savez_compressed(out/f"episode_{c['id']:03d}.npz",**{k:np.stack([s[k][i] for s in traces[:length]]) for k in traces[0]},time_s=np.arange(1,length+1)*dt)
    write(out/'result.json',dict(status='complete',role='teacher' if collect else 'direct action MLP',
        rows=rows,successes=sum(r['success'] for r in rows),falls=sum(r['fall'] for r in rows),
        wall_s=time.monotonic()-start,reset_errors=reset_errors,dt_s=dt,
        reference_tracker=False,external_stop_reference=False,sequential_goal_observation_supervisor=True,
        teacher_sha256=sha(TEACHER),script_sha256=sha(__file__)))
    print(json.dumps(dict(mode=args.mode,successes=sum(r['success'] for r in rows),episodes=n)),flush=True)


def prepare(args):
    args.root.mkdir(exist_ok=False,parents=True)
    rng=np.random.default_rng(args.seed);cases=[]
    bearings=[0,45,90,135,180,-135,-90,-45]
    for i in range(80):
        cases.append(dict(id=i,split='train' if i<64 else 'validation',
            distance_m=float(rng.uniform(3,6)),bearing_deg=bearings[i%8],
            yaw_offset_deg=[0,90,-90,180][(i//8)%4]))
    files=[TEACHER,Path(__file__),ROOT/'configs/goal_teacher/view_sequential_arrival_v34.yaml',
           ROOT/'configs/goal_teacher/agent_sequential_v31.yaml',ROOT/'configs/engines/isaac_lab.yaml',
           ROOT/'src/humanoid_imitation/envs/goal_amp_env.py']
    write(args.root/'protocol.json',dict(date='2026-11-03',seed=args.seed,cases=cases,
        train_episodes=64,seconds=40,physics_budget=dict(teacher=80,student=16),
        epochs=args.epochs,frozen={str(p):sha(p) for p in files},
        role='bounded direct action BC feasibility; one-step action every1/30s; no diffusion or DAgger',
        thresholds=dict(position_m=1.25,yaw_deg=45,speed_mps=.15,wz_radps=.2,hold_s=5,drift_m=.1,post_s=2),
        selection='validation episode action MSE; single physical evaluation of selected model',
        limitation='teacher sequential goal observation supervisor retained; source motions shared; not independent motion generalization'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','collect','train','evaluate'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--seed',type=int,default=3911)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--device',default='cuda')
    args=p.parse_args();args.root=args.root.resolve()
    if args.mode=='prepare':prepare(args)
    elif args.mode=='train':train(args)
    else:simulate(args)


if __name__=='__main__':main()
