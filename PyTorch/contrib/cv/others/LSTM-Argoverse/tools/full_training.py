"""Original XY LSTM curriculum on all Argoverse v1.1 trajectories, with resumable epoch state."""
import datetime
import json
import math
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from common import ROOT, PROFILE, own_path, sha256, write_json, parameter_sha, tensor_tree_cpu, finite_tree
import model_task as task

ROLLOUTS = [1, 10, 30]

def save(value,path):
    temp=path.with_suffix(path.suffix+'.partial')
    torch.save(value,temp);temp.replace(path)

def validation_loss(model,dataset,device):
    model.eval();values=[]
    with torch.no_grad():
        for batch in DataLoader(dataset,batch_size=512,shuffle=False):
            prediction=model(batch[0].to(device),rollout_steps=30)
            loss=torch.nn.functional.mse_loss(prediction,batch[1].to(device))
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite original 30-step validation loss.')
            values.append(float(loss.cpu()))
    # The pinned baseline averages batch means, including its shorter final batch.
    return sum(values)/len(values)

def main(args):
    data=own_path(args.data_path,exists=True)
    metadata=task.check_data(data,'full')
    epochs=args.epochs if args.epochs is not None else 5000
    lr=args.lr if args.lr is not None else 0.001
    if epochs<1 or lr<=0 or args.steps not in [None,0]:
        raise RuntimeError('Full curriculum uses complete epochs, positive lr and no step cutoff.')
    if args.plan:
        print(json.dumps({'mode':'full','counts':metadata['counts'],'rollout_lengths':ROLLOUTS,
            'global_epoch_limit':epochs,'validation_every_epochs':5,
            'advance_after_consecutive_validation_increases':3,'validation_horizon':30,
            'global_batch_size_default':512,'per_rank_batch_override':args.batch_size,
            'optimizer':'independent encoder Adam and decoder Adam','lr':lr,
            'npu_compute_executed':False},indent=2));return
    import torch_npu
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    rank=int(os.environ.get('RANK','0'));local_rank=int(os.environ.get('LOCAL_RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    if 512%world and args.batch_size is None:raise RuntimeError('Global batch 512 must divide rank count.')
    per_rank=args.batch_size if args.batch_size is not None else 512//world
    if per_rank<1:raise RuntimeError('Batch size must be positive.')
    global_batch=per_rank*world
    settings={'epochs':epochs,'lr':lr,'steps':0,'batch_size':per_rank,'global_batch_size':global_batch,
              'rollouts':ROLLOUTS,'validation_every_epochs':5,'advance_after_increases':3}
    output=own_path(args.output_dir,exists=True)
    torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(100)
    torch_npu.npu.set_device(local_rank);torch_npu.npu.manual_seed(42+rank)
    device=torch.device('npu',local_rank)
    if world>1:dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=1200))
    model=task.create_model(metadata).to(device)
    if any(p.device.type!='npu' for p in model.parameters()):raise RuntimeError('Full LSTM parameters must reside on NPU.')
    enc=torch.optim.Adam(model.encoder.parameters(),lr=lr)
    dec=torch.optim.Adam(model.decoder.parameters(),lr=lr)
    wrapped=DistributedDataParallel(model,device_ids=[local_rank]) if world>1 else model
    train,validation=task.datasets(data,metadata)
    digest=sha256(data/'metadata.json')
    epoch=step=stage=increases=0;best=previous=math.inf
    history=[];best_states={};stage_epochs={str(x):0 for x in ROLLOUTS}
    if args.resume:
        state=torch.load(own_path(args.resume,exists=True),map_location='cpu',weights_only=True)
        if state.get('checkpoint_role')!='curriculum_continuation':raise RuntimeError('Continue with latest.pt, not selected inference weights.')
        if state['dataset_digest']!=digest or state['model']!=PROFILE['model'] or state['world_size']!=world:
            raise RuntimeError('Resume model, all-data digest or rank count differs.')
        if state['settings']['global_batch_size']!=global_batch or state['settings']['lr']!=lr:
            raise RuntimeError('Resume global batch size or learning rate differs.')
        model.load_state_dict(state['state_dict'],strict=True)
        enc.load_state_dict(state['optimizer']['encoder']);dec.load_state_dict(state['optimizer']['decoder'])
        epoch,step,stage=state['epoch'],state['step'],state['stage_index']
        increases,best,previous=state['increases'],state['best_loss'],state['previous_loss']
        history,best_states,stage_epochs=state['history'],state['best_states'],state['stage_epochs']
        torch.set_rng_state(state['rank_rng'][rank]['torch'])
        torch_npu.npu.set_rng_state(state['rank_rng'][rank]['npu'],device=local_rank)
    initial=parameter_sha(model);reason='global_epoch_limit'
    while epoch<epochs and stage<len(ROLLOUTS):
        rollout=ROLLOUTS[stage]
        model.train();total=count=0;begin=time.monotonic()
        # The original training loader shuffles; epoch-keyed order makes an epoch restart reproducible.
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(42+epoch)).tolist()
        for offset in range(0,len(order),global_batch):
            batch_indices=order[offset:offset+global_batch]
            indices=batch_indices[rank::world];real_count=len(indices)
            if not indices:indices=[batch_indices[0]]
            batch=(train.tensors[0][indices],train.tensors[1][indices])
            enc.zero_grad(set_to_none=True);dec.zero_grad(set_to_none=True)
            prediction=wrapped(batch[0].to(device),rollout_steps=rollout)
            raw=torch.nn.functional.mse_loss(prediction,batch[1][:,:rollout].to(device))
            loss=raw*(world*real_count/len(batch_indices))
            if not bool(torch.isfinite(loss)):raise RuntimeError('Nonfinite full-curriculum loss.')
            loss.backward()
            if step%100==0:
                grads=[p.grad for p in model.parameters() if p.grad is not None]
                if len(grads)!=14 or any(g.device.type!='npu' or not bool(torch.isfinite(g).all()) for g in grads):
                    raise RuntimeError('All 14 original LSTM gradients must be finite NPU tensors.')
            enc.step();dec.step();step+=1
            total+=float(raw.detach().cpu())*real_count;count+=real_count
        sums=torch.tensor([total,count],device=device,dtype=torch.float32)
        if world>1:dist.all_reduce(sums)
        if int(sums[1].cpu())!=len(train):raise RuntimeError('Each official training trajectory must appear once per epoch.')
        epoch+=1;stage_epochs[str(rollout)]+=1
        row={'epoch':epoch,'rollout':rollout,'optimizer_steps':step,
             'train_mse':float((sums[0]/sums[1]).cpu()),'train_sequences_seen':int(sums[1].cpu()),
             'seconds':time.monotonic()-begin}
        if epoch%5==0:
            value=validation_loss(model,validation,device) if rank==0 else 0.0
            tensor=torch.tensor([value],device=device)
            if world>1:dist.broadcast(tensor,src=0)
            value=float(tensor.cpu()[0]);row['validation_30step_mse']=value
            if value<=best:
                best=value;best_states[str(rollout)]={'state_dict':tensor_tree_cpu(model.state_dict()),'epoch':epoch,'step':step,'loss':value}
            increases=0 if value<=previous else increases+1
            previous=value;row['consecutive_validation_increases']=increases
            if increases>=3:
                stage+=1;increases=0;best=previous=math.inf
        history.append(row)
        save({'torch':torch.get_rng_state(),'npu':torch_npu.npu.get_rng_state(local_rank).cpu()},output/('rng_rank_'+str(rank)+'.pt'))
        if world>1:dist.barrier()
        if rank==0:
            checkpoint={'model':PROFILE['model'],'checkpoint_role':'curriculum_continuation',
                'state_dict':tensor_tree_cpu(model.state_dict()),
                'optimizer':{'encoder':tensor_tree_cpu(enc.state_dict()),'decoder':tensor_tree_cpu(dec.state_dict())},
                'epoch':epoch,'step':step,'stage_index':stage,'world_size':world,'dataset_digest':digest,
                'metadata':metadata,'settings':settings,'increases':increases,'best_loss':best,'previous_loss':previous,
                'history':history,'best_states':best_states,'stage_epochs':stage_epochs,
                'rank_rng':[torch.load(output/('rng_rank_'+str(r)+'.pt'),weights_only=True) for r in range(world)]}
            save(checkpoint,output/'latest.pt');write_json(output/'epoch_history.json',history)
            print(json.dumps(row),flush=True)
        if world>1:dist.barrier()
        if args.pause_after_epochs and epoch>=args.pause_after_epochs:
            reason='intentional_epoch_boundary_pause';break
    if stage==len(ROLLOUTS):reason='all_curriculum_stages_advanced_by_reference_rule'
    if parameter_sha(model)==initial or not finite_tree(model.state_dict()) or not finite_tree(enc.state_dict()) or not finite_tree(dec.state_dict()):
        raise RuntimeError('No parameter update or nonfinite curriculum / optimizer state.')
    final=parameter_sha(model)
    if world>1:
        tensor=torch.tensor(list(bytes.fromhex(final)),dtype=torch.int32,device=device)
        values=[torch.empty_like(tensor) for _ in range(world)];dist.all_gather(values,tensor)
        if any(not torch.equal(tensor,value) for value in values):raise RuntimeError('LSTM rank parameter SHA differs.')
    write_json(output/('full_rank_'+str(rank)+'.json'),{'rank':rank,'world_size':world,'epochs_completed':epoch,
        'optimizer_steps':step,'parameters_on_npu':True,'parameter_sha256':final,'state_finite':True,
        'optimizer_finite':True,'all_rank_parameter_hashes_match':True,'stage_epochs':stage_epochs})
    if rank==0:
        selected=best_states.get('30')
        model_state=selected['state_dict'] if selected else tensor_tree_cpu(model.state_dict())
        selected_epoch=selected['epoch'] if selected else epoch
        selected_step=selected['step'] if selected else step
        checkpoint={'model':PROFILE['model'],'checkpoint_role':'curriculum_selected_model' if selected else 'curriculum_intermediate_model',
            'state_dict':model_state,'epoch':selected_epoch,'step':selected_step,'world_size':world,
            'dataset_digest':digest,'metadata':metadata,'settings':settings,'continuation_checkpoint':'latest.pt'}
        save(checkpoint,output/'checkpoint.pt')
        model.load_state_dict(model_state,strict=True);model.eval()
        sample=next(iter(DataLoader(validation,batch_size=512)))
        with torch.no_grad():reference=task.predict(model,sample,device).cpu()
        cpu=task.create_model(metadata).cpu();cpu.load_state_dict(model_state,strict=True);cpu.eval()
        with torch.no_grad():control=task.predict(cpu,sample,torch.device('cpu'))
        torch.testing.assert_close(reference,control,rtol=3e-3,atol=3e-3)
        save({'batch':sample,'prediction':reference},output/'reload_reference.pt')
        complete=reason!='intentional_epoch_boundary_pause' and all(stage_epochs[str(x)]>0 for x in ROLLOUTS) and selected is not None
        result={'model':PROFILE['model'],'mode':'full','result':'full_curriculum_run_finished','stop_reason':reason,
            'epochs_completed':epoch,'optimizer_steps':step,'world_size':world,'stage_epochs':stage_epochs,
            'selected_epoch':selected_epoch,'selected_checkpoint_has_30step_validation_selection':selected is not None,
            'dataset_digest':digest,'data_scope':metadata['scope'],'data_path':data.relative_to(ROOT).as_posix(),
            'settings':settings,'checkpoint_sha256':sha256(output/'checkpoint.pt'),'latest_checkpoint_sha256':sha256(output/'latest.pt'),
            'cpu_npu_max_absolute_error':float((reference-control).abs().max()),
            'full_reference_protocol_completed':complete and epochs==5000 and global_batch==512 and lr==0.001,
            'metrics':task.evaluate(model,validation,device,512,metadata) if complete else None,
            'formal_accuracy_claimed':False,'resume_from':str(args.resume) if args.resume else None}
        write_json(output/'train_result.json',result);print(json.dumps(result,indent=2),flush=True)
    if world>1:dist.barrier();dist.destroy_process_group()
