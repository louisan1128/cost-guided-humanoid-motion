#!/usr/bin/env python3
"""Same-data/architecture loss repair; frozen action prior, future-motion selection."""
import argparse,copy,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_short_action_sequence as core
import run_anchored_action_sequence as anchored
from pilot_direct_action_bc import write,sha
from humanoid_imitation.planning.state_action_sequence import alpha_schedule,noisy_action,sample_sequence


def motion_losses(pred,target,std,mean):
    horizon=pred.shape[-1]//99;error=(pred-target).reshape(-1,horizon,99);state=error[:,:,28:].square().mean();root=error[:,:,28:43].square().mean()
    al,_=core.reconstruction_losses(pred.reshape(-1,horizon,99),target.reshape(-1,horizon,99),8,16/45)
    physical=(pred*std+mean).reshape(-1,horizon,99);velocity=torch.diff(physical[:,:,28:30],dim=1)*30
    consistency=((physical[:,1:,37:39]-velocity)/.5).square().mean()
    total=state+root+.05*al+.01*consistency
    return total,dict(state=state,root=root,action=al,consistency=consistency)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['mlp','diffusion'],required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--data',type=Path,default=ROOT/'outputs/state_action_sequence_v2/dataset_h30/data.pt');p.add_argument('--updates',type=int,default=4000);p.add_argument('--balanced-groups',action='store_true');p.add_argument('--denoising-selection',action='store_true');p.add_argument('--new-fraction',type=float,default=.5);a=p.parse_args();a.out_dir=a.out_dir.resolve();a.out_dir.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);torch.manual_seed(0);tic=time.monotonic()
    parent=ROOT/f'outputs/short_action_sequence_v3/repair2/models/{a.method}/model.pt';c=torch.load(parent,map_location='cpu',weights_only=False);d=torch.load(a.data,weights_only=False);c['execution_decoder_sha256']=d.get('execution_decoder_sha256')
    horizon_changed=d['horizon']!=c['horizon']
    if horizon_changed:c['horizon']=d['horizon'];c['stats']=d['stats']
    stats={k:v.cuda() for k,v in c['stats'].items()};model=anchored.make_anchored(c).cuda()
    if horizon_changed:
        current=model.state_dict();compatible={k:v for k,v in c['model'].items()if k in current and v.shape==current[k].shape};model.load_state_dict(compatible,strict=False)
    else:model.load_state_dict(c['model'])
    model.train();ema=copy.deepcopy(model).eval();opt=torch.optim.Adam([v for v in model.parameters()if v.requires_grad],lr=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.updates,eta_min=1e-5);alpha=alpha_schedule().cuda()
    def normalized(split):
        x=d[split]['context'];y=d[split]['target'].flatten(1)
        return ((x-stats['context_mean'].cpu())/stats['context_std'].cpu()).clamp(-10,10).cuda(),((y-stats['target_mean'].cpu())/stats['target_std'].cpu()*model.valid_output.cpu()).cuda()
    x,y=normalized('train');vx,vy=normalized('validation');g=torch.Generator(device='cuda').manual_seed(5034)
    if a.balanced_groups:
        groups=[torch.where(d['train']['group'].cuda()==k)[0]for k in [0,1]];validation_groups=[torch.where(d['validation']['group'].cuda()==k)[0]for k in [0,1]]
        ids=torch.cat([ix[torch.randperm(len(ix),device='cuda',generator=g)[:1024]]for ix in validation_groups])
    else:ids=torch.randperm(len(vx),device='cuda',generator=g)[:2048]
    vx,vy=vx[ids],vy[ids];noise=torch.randn(vy.shape,device='cuda',generator=g);curve=[];best=float('inf');best_step=None
    for step in range(a.updates+1):
        if step:
            if a.balanced_groups:
                new_count=round(512*a.new_fraction);ix=torch.cat([group[torch.randint(len(group),(count,),device='cuda')]for group,count in zip(groups,[512-new_count,new_count])])
            else:ix=torch.randint(len(x),(512,),device='cuda')
            bx,by=x[ix],y[ix]
            if a.method=='mlp':pred=model(bx)
            else:
                t=torch.randint(100,(512,),device='cuda');t[:256]=torch.randint(75,100,(256,),device='cuda');pred=model(bx,noisy_action(by,torch.randn_like(by),t,alpha),t)
            loss,terms=motion_losses(pred,by,stats['target_std'],stats['target_mean'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();scheduler.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),model.parameters()):ep.lerp_(mp,.005)
        if step%500==0:
            with torch.no_grad():
                prediction=torch.cat([sample_sequence(ema,vx[k:k+256],alpha,noise[k:k+256])for k in range(0,len(vx),256)]);_,terms=motion_losses(prediction,vy,stats['target_std'],stats['target_mean']);sample_score=float(terms['state']+terms['root']+.01*terms['consistency']);score=sample_score
                if a.denoising_selection and a.method=='diffusion':
                    vt=torch.arange(len(vx),device='cuda')%100;denoised=ema(vx,noisy_action(vy,noise,vt,alpha),vt);_,validation=motion_losses(denoised,vy,stats['target_std'],stats['target_mean']);score=float(validation['state']+validation['root']+.01*validation['consistency'])
                row=dict(update=step,validation_score=score,sampled_reference_mse_score=sample_score,**{k:float(v)for k,v in terms.items()});curve.append(row);print(row,flush=True)
            if score<best:
                best=score;best_step=step;checkpoint=dict(c,model={k:v.cpu().clone()for k,v in ema.state_dict().items()},update=step,selection='validation denoising state+root+.01consistency'if a.denoising_selection else'minimum validation sampled state+root+.01consistency; action prior frozen',parent_path=str(parent),parent_sha256=sha(parent));torch.save(checkpoint,a.out_dir/'model.pt')
    torch.save(dict(c,model={k:v.cpu().clone()for k,v in ema.state_dict().items()},update=a.updates),a.out_dir/'last.pt')
    ck=torch.load(a.out_dir/'model.pt',weights_only=False);prior_exact=all(torch.equal(v,ck['model'][k])for k,v in c['model'].items()if k.startswith('prior.'))
    if not prior_exact:raise ValueError('Frozen prior changed')
    write(a.out_dir/'curve.json',curve);write(a.out_dir/'result.json',dict(status='complete',method=a.method,updates=a.updates,selected_update=best_step,train_windows=len(x),validation_windows=len(vx),balanced_groups=a.balanced_groups,new_fraction=a.new_fraction,seed=0,optimizer_resume=False,initialization='parent weights; fresh optimizer',loss='state+root+.05action+.01root_velocity_consistency',selection='validation denoising state+root+.01consistency'if a.denoising_selection else'state+root+.01consistency',data_sha256=sha(a.data),model_sha256=sha(a.out_dir/'model.pt'),parent_sha256=sha(parent),prior_tensor_exact=prior_exact,wall_s=time.monotonic()-tic,runner_sha256=sha(__file__)))
if __name__=='__main__':main()
