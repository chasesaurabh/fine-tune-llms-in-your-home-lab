"""Measure fixed-token training capacity; random tokens are not a quality benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time
from .evaluation import verify_files
from .training import load_model,write_json


def tensor_hash(tensor):
    import torch
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def last_logits(model,ids):
    import torch
    model.eval();model.gradient_checkpointing_disable()
    torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        value=model(input_ids=ids,use_cache=False).logits[:,-1].float()
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='stage',required=True)
    train=sub.add_parser('train');train.add_argument('--model',required=True)
    train.add_argument('--revision',required=True)
    train.add_argument('--method',choices=['lora','qlora'],required=True)
    train.add_argument('--tokens',type=int,choices=[512,1024],required=True)
    train.add_argument('--output',type=Path,required=True)
    reload=sub.add_parser('reload');reload.add_argument('--run',type=Path,required=True)
    args=p.parse_args()
    run=args.output if args.stage=='train' else args.run
    if args.stage=='train':
        if run.exists():p.error('use a new output directory')
        if not re.fullmatch('[0-9a-f]{40}',args.revision):p.error('pin a 40-character model revision')
        run.mkdir(parents=True)
        config={'model':args.model,'revision':args.revision,'method':args.method,'tokens':args.tokens,
            'micro_batch':1,'accumulation':8,'optimizer_steps':1,'rank':8,'alpha':16,'dropout':0.05,
            'seed':20261121,'data_seed':91,'supervised_tokens':64,'targets':'all-linear',
            'scope':'Fixed random token IDs; dense attention; capacity only, no quality claim'}
        write_json(run/'config.json',config)
    else:
        if (run/'reload.json').exists():p.error('reload already recorded; inspect it before retrying')
        config=json.loads((run/'config.json').read_text())
        verify_files(run/'adapter',json.loads((run/'adapter-hashes.json').read_text()))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from peft import LoraConfig,PeftModel,get_peft_model,prepare_model_for_kbit_training
    from transformers import set_seed
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    set_seed(config['seed'],deterministic=True)
    torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    model=load_model(config['model'],config['revision'],config['method'])
    if config['method']=='qlora':
        if not getattr(model,'is_loaded_in_4bit',False):raise RuntimeError('QLoRA did not load in four bits')
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=args.stage=='train')
    if args.stage=='train':
        model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,bias='none',
            task_type='CAUSAL_LM',target_modules='all-linear'))
    else:model=PeftModel.from_pretrained(model,run/'adapter',is_trainable=False)
    torch.cuda.synchronize()
    load={'seconds':time.perf_counter()-start,'allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    generator=torch.Generator(device='cpu').manual_seed(config['data_seed'])
    ids=torch.randint(0,model.config.vocab_size,(1,config['tokens']),generator=generator).to(model.device)
    if args.stage=='train':
        write_json(run/'input-ids.json',ids.cpu().tolist())
        labels=ids.clone();labels[:,:-config['supervised_tokens']]=-100
        model.train();model.config.use_cache=False
        model.gradient_checkpointing_enable();model.enable_input_require_grads()
        optimizer=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=2e-4,fused=True)
        torch.backends.cuda.matmul.allow_tf32=True
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();losses=[]
        for _ in range(config['accumulation']):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=labels).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite capacity loss')
            losses.append(float(loss.detach().cpu()));(loss/config['accumulation']).backward()
        torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1)
        optimizer.step();optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        training={'seconds':time.perf_counter()-start,'allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'reserved_gib':torch.cuda.max_memory_reserved()/2**30,'micro_losses':losses}
        model.save_pretrained(run/'adapter');torch.save(optimizer.state_dict(),run/'optimizer.pt')
        write_json(run/'adapter-hashes.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
            for f in (run/'adapter').iterdir() if f.is_file()})
        logits=last_logits(model,ids)
        write_json(run/'capacity.json',{'load':load,'training':training,'actual_tokens':ids.shape[1],
            'actual_supervised_tokens':int((labels!=-100).sum()),'input_ids_sha256':tensor_hash(ids),
            'last_logits_sha256':tensor_hash(logits),'top_token':int(logits.argmax()),
            'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
            'parameter_devices':sorted({str(v.device) for v in model.parameters()}),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    else:
        previous=json.loads((run/'capacity.json').read_text())
        if ids.cpu().tolist()!=json.loads((run/'input-ids.json').read_text()):raise ValueError('Capacity input differs')
        logits=last_logits(model,ids);same=tensor_hash(logits)==previous['last_logits_sha256']
        write_json(run/'reload.json',{'load':load,'last_logits_identical':same,'top_token':int(logits.argmax()),
            'last_logits_sha256':tensor_hash(logits),'input_ids_sha256':tensor_hash(ids)})
        if not same:raise RuntimeError('Fresh-process capacity reconstruction differs')


if __name__=='__main__':main()
