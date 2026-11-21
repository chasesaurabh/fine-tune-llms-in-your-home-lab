"""Continue a verified BF16 SFT adapter with a separate frozen SFT reference."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from .data import audit_splits,load_rows
from .evaluation import generate,verify_files
from .training import check_mask,formatted_rows,load_model,render,write_json


def pair_rows(tokenizer,rows,max_length,instruction):
    # Check both branches, including the EOS token, before TRL can truncate.
    formatted_rows(tokenizer,rows,max_length,instruction)
    formatted_rows(tokenizer,[{**r,'completion':r['rejected']} for r in rows],max_length,instruction)
    if any(r['chosen']==r['rejected'] for r in rows): raise ValueError('identical chosen and rejected responses')
    return [{'prompt':render(tokenizer,r['prompt'],instruction),
        'chosen':r['chosen']+tokenizer.eos_token,'rejected':r['rejected']+tokenizer.eos_token} for r in rows]


def adapter_tensor_hashes(model):
    import torch
    return {name:hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        for name,p in model.named_parameters() if 'lora_' in name}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sft-run',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261121)
    parser.add_argument('--epochs',type=float,default=1)
    parser.add_argument('--steps',type=int,default=-1)
    args=parser.parse_args()
    if args.output.exists(): parser.error('use a new output directory')
    if args.epochs<=0 or args.steps==0 or args.steps < -1: parser.error('invalid training budget')
    config=json.loads((args.sft_run/'config.json').read_text())
    if config['method']!='lora': parser.error('this reference lab requires a BF16 LoRA SFT parent')
    if audit_splits(args.data)!=config['dataset']: raise ValueError('data differ from SFT parent')
    parent_hashes=json.loads((args.sft_run/'adapter-hashes.json').read_text())
    verify_files(args.sft_run/'adapter',parent_hashes)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoTokenizer,set_seed
    from trl import DPOConfig,DPOTrainer
    if not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA is required')
    set_seed(args.seed,deterministic=True)
    tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],trust_remote_code=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    train=load_rows(args.data/'train.jsonl');validation=load_rows(args.data/'validation.jsonl')
    instruction=config.get('instruction','')
    formatted=pair_rows(tokenizer,train,config['max_length'],instruction)
    val_formatted=pair_rows(tokenizer,validation,config['max_length'],instruction)
    args.output.mkdir(parents=True)
    config.update(seed=args.seed,epochs=args.epochs,steps=args.steps,algorithm='DPO sigmoid',
        learning_rate=5e-5,beta=0.1,micro_batch=4,accumulation=2,
        gradient_checkpointing=False,reference='separate frozen SFT parent',parent_adapter_hashes=parent_hashes,
        in_memory_evaluation=True,task='preference',dropout=0.0,disable_dropout=True,
        optimizer='adamw_torch_fused',lr_scheduler='cosine',warmup_steps=2,
        weight_decay=0.0,adam_beta1=0.9,adam_beta2=0.999,adam_epsilon=1e-8,
        max_grad_norm=1.0,data_seed=args.seed,tf32_training=True)
    write_json(args.output/'config.json',config)
    policy=PeftModel.from_pretrained(load_model(config['model'],config['revision'],'lora'),
        args.sft_run/'adapter',is_trainable=True)
    reference=PeftModel.from_pretrained(load_model(config['model'],config['revision'],'lora'),
        args.sft_run/'adapter',is_trainable=False)
    for parameter in reference.parameters(): parameter.requires_grad_(False)
    reference.eval()
    before=adapter_tensor_hashes(reference)
    if not before: raise ValueError('reference has no adapter tensors')
    if not any(p.requires_grad for p in policy.parameters()): raise ValueError('policy has no trainable parameters')
    trainer=DPOTrainer(model=policy,ref_model=reference,processing_class=tokenizer,
        train_dataset=Dataset.from_list(formatted),eval_dataset=Dataset.from_list(val_formatted),
        args=DPOConfig(output_dir=str(args.output/'checkpoints'),num_train_epochs=args.epochs,
            max_steps=args.steps,per_device_train_batch_size=4,per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,learning_rate=5e-5,lr_scheduler_type='cosine',warmup_steps=2,
            optim='adamw_torch_fused',weight_decay=0.0,adam_beta1=0.9,adam_beta2=0.999,
            adam_epsilon=1e-8,max_grad_norm=1.0,
            logging_steps=1,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,
            bf16=True,tf32=True,gradient_checkpointing=False,max_length=config['max_length'],
            beta=0.1,loss_type=['sigmoid'],disable_dropout=True,sync_ref_model=False,
            report_to='none',seed=args.seed,data_seed=args.seed,dataloader_num_workers=0))
    expected={}
    for row in formatted:
        prefix=tokenizer(row['prompt'],add_special_tokens=False)['input_ids']
        for branch in ('chosen','rejected'):
            expected[tuple(tokenizer(row['prompt']+row[branch],add_special_tokens=False)['input_ids'])]=len(prefix)
    batch=next(iter(trainer.get_train_dataloader()));masks=[]
    for ids,attention,mask in zip(batch['input_ids'],batch['attention_mask'],batch['completion_mask'],strict=True):
        ids=ids.tolist();attention=attention.tolist();mask=mask.tolist()
        active=tuple(ids[:sum(attention)])
        if active not in expected: raise ValueError('DPO collator changed a token sequence')
        labels=[token if selected else -100 for token,selected in zip(ids,mask,strict=True)]
        check_mask(ids,labels,attention,expected[active],tokenizer.eos_token_id)
        masks.append({'input_ids':ids,'attention_mask':attention,'completion_mask':mask})
    write_json(args.output/'loss-mask.json',masks)
    del batch
    set_seed(args.seed,deterministic=True)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    result=trainer.train();torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    peak={'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    after=adapter_tensor_hashes(reference)
    if before!=after or any(p.requires_grad for p in reference.parameters()):
        raise ValueError('frozen SFT reference changed')
    trainer.save_model(str(args.output/'adapter'));tokenizer.save_pretrained(args.output/'adapter')
    write_json(args.output/'adapter-hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (args.output/'adapter').iterdir() if p.is_file()})
    write_json(args.output/'training.json',{'training_seconds':elapsed,'training_peak':peak,
        'metrics':result.metrics,'log_history':trainer.state.log_history,
        'reference_adapter_unchanged':before==after,'reference_tensor_hashes':after,
        'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','trl')},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    model=trainer.accelerator.unwrap_model(trainer.model,keep_fp32_wrapper=False)
    model.gradient_checkpointing_disable()
    write_json(args.output/'in-memory-validation.json',generate(model,tokenizer,validation,
        'preference',batch_size=1,instruction=instruction))


if __name__=='__main__': main()
