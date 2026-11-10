#!/usr/bin/env python3
"""Same source groups, H60 targets, train-only normalization, explicit provenance."""
import argparse,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from pilot_direct_action_bc import write,sha
from humanoid_imitation.planning.state_action_sequence import build_windows,local_state,yaw

def main():
    p=argparse.ArgumentParser();p.add_argument('--out-dir',type=Path,required=True);a=p.parse_args();a.out_dir=a.out_dir.resolve();a.out_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);tic=time.monotonic();parts={s:[]for s in ['train','validation']};provenance=[]
    source=ROOT/'outputs/state_action_sequence_v1/collection_certified';index=json.loads((source/'result.json').read_text())
    for row in index['rows']:
        s=row['split']
        if s not in parts:continue
        path=source/f"episode_{row['id']:03d}.pt";e=torch.load(path,weights_only=False);result=build_windows(e,60,joint=True)
        if result is None:continue
        x,y,starts=result;ix=torch.linspace(0,len(x)-1,min(500,len(x))).round().long().unique();parts[s].append(dict(context=x[ix],target=y[ix],episode_id=torch.full((len(ix),),row['id'],dtype=torch.long),group=torch.zeros(len(ix),dtype=torch.long)));provenance.append(dict(file=str(path),source_id=row['id'],split=s,group='legacy_teacher',starts=starts[ix].tolist()))
    manifest=torch.load(ROOT/'outputs/obstacle_repair_v10/dataset_certified_final/window_provenance.pt',weights_only=False)
    for record in manifest['episodes']:
        path=Path(record['file']);e=torch.load(path,weights_only=False);row=e['row'];s=row['split'];starts=torch.linspace(0,len(e['obs'])-60,min(300,len(e['obs'])-59)).round().long().unique();obs=torch.cat((e['prefix_obs'],e['obs']));acts=torch.cat((e['prefix_action'],e['applied_action']));h=starts[:,None]+torch.arange(4);past=starts[:,None]+torch.arange(3);future=starts[:,None]+torch.arange(60);x=torch.cat((obs[h].flatten(1),acts[past].flatten(1)),-1);state=local_state({k:v[future]for k,v in e['next'].items()},e['pre']['root_pos'][starts,None],yaw(e['pre']['root_rot'][starts])[:,None]);y=torch.cat((e['applied_action'][future],state),-1);parts[s].append(dict(context=x,target=y,episode_id=torch.full((len(x),),row['source_id'],dtype=torch.long),group=torch.ones(len(x),dtype=torch.long)));provenance.append(dict(record,group='certified_decoder',starts=starts.tolist()))
    data={s:{k:torch.cat([e[k]for e in ps])for k in ps[0]}for s,ps in parts.items()};old=torch.load(ROOT/'outputs/short_action_sequence_v3/repair2/models/diffusion/model.pt',weights_only=False)['stats'];target=data['train']['target'].flatten(1);mean=target.mean(0);std=target.std(0).clamp_min(.01);mean[:28]=old['target_mean'][:28];std[:28]=old['target_std'][:28];data.update(stats=dict(context_mean=old['context_mean'],context_std=old['context_std'],target_mean=mean,target_std=std),horizon=60,joint=True,execution_decoder_path=str(ROOT/'outputs/obstacle_repair_v10/decoder_repair/model.pt'),execution_decoder_sha256=manifest['decoder_sha256'])
    if set(data['train']['episode_id'].tolist())&set(data['validation']['episode_id'].tolist()):raise ValueError('Source leakage')
    torch.save(data,a.out_dir/'data.pt');write(a.out_dir/'provenance.json',provenance);write(a.out_dir/'result.json',dict(status='complete',horizon=60,windows={s:len(data[s]['context'])for s in parts},new_windows={s:int(data[s]['group'].sum())for s in parts},normalization='context/action0 inherited frozen train-only; H60 target stats recomputed train only',execution_decoder_sha256=manifest['decoder_sha256'],data_sha256=sha(a.out_dir/'data.pt'),wall_s=time.monotonic()-tic));print('H60 DATA COMPLETE',flush=True)
if __name__=='__main__':main()
