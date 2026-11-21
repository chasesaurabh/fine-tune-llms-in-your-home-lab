"""Pinned single-GPU SFT. Run evaluation in a separate process after this exits."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time
from .data import audit_splits, load_rows


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def render(tokenizer, prompt, instruction=""):
    content=(instruction+"\n\n"+prompt) if instruction else prompt
    return tokenizer.apply_chat_template([{"role":"user", "content":content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def formatted_rows(tokenizer, rows, max_length, instruction=""):
    prepared=[]
    lengths=[]
    for row in rows:
        prompt=render(tokenizer, row["prompt"], instruction)
        completion=row.get("completion", row.get("chosen"))
        if not isinstance(completion,str) or not completion.strip():
            raise ValueError(f"{row['id']}: empty completion")
        completion += tokenizer.eos_token
        prefix=tokenizer(prompt, add_special_tokens=False)["input_ids"]
        combined=tokenizer(prompt+completion, add_special_tokens=False)["input_ids"]
        if combined[:len(prefix)] != prefix:
            raise ValueError(f"{row['id']}: prompt/completion token boundary changed")
        if len(combined)>max_length or len(combined)<=len(prefix):
            raise ValueError(f"{row['id']}: supervised target would be truncated or empty")
        prepared.append({"prompt":prompt,"completion":completion})
        lengths.append({"id":row["id"],"prompt_tokens":len(prefix),"total_tokens":len(combined),"target_tokens":len(combined)-len(prefix)})
    return prepared,lengths


def load_model(model_id, revision, method):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs={"revision":revision,"dtype":torch.bfloat16,"device_map":{"":0},"trust_remote_code":False}
    if method=="qlora":
        kwargs["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    return AutoModelForCausalLM.from_pretrained(model_id, **kwargs)


def check_mask(ids, labels, attention, prompt_length, eos_id):
    if not (len(ids)==len(labels)==len(attention)):
        raise ValueError('token, label and attention lengths differ')
    active=sum(attention)
    if attention != [1]*active+[0]*(len(attention)-active):
        raise ValueError('training batch must use right padding')
    if not 0 < prompt_length < active:
        raise ValueError('missing prompt or completion')
    expected=[-100]*prompt_length+ids[prompt_length:active]+[-100]*(len(ids)-active)
    if labels != expected:
        raise ValueError('loss mask does not select exactly the completion')
    if ids[active-1]!=eos_id:
        raise ValueError('completion end token missing')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--model",default="Qwen/Qwen3-0.6B")
    parser.add_argument("--revision",required=True)
    parser.add_argument("--method",choices=["lora","qlora"],default="lora")
    parser.add_argument("--seed",type=int,default=20261120)
    parser.add_argument("--epochs",type=float,default=2)
    parser.add_argument("--steps",type=int,default=-1)
    parser.add_argument("--micro-batch",type=int,default=4)
    parser.add_argument("--accumulation",type=int,default=2)
    parser.add_argument("--max-length",type=int,default=512)
    parser.add_argument("--gradient-checkpointing",action="store_true")
    parser.add_argument("--instruction-file",type=Path)
    args=parser.parse_args()
    if args.output.exists(): parser.error("use a new output directory")
    if not re.fullmatch('[0-9a-f]{40}',args.revision): parser.error("pin a 40-character revision")
    if min(args.micro_batch,args.accumulation,args.max_length,args.epochs)<=0 or args.steps==0 or args.steps < -1:
        parser.error("positive budgets are required; --steps -1 means use epochs")
    audit=audit_splits(args.data)
    train=load_rows(args.data/'train.jsonl')
    validation=load_rows(args.data/'validation.jsonl')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoTokenizer, set_seed
    from trl import SFTConfig,SFTTrainer
    if not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA is required for this path')
    set_seed(args.seed,deterministic=True)
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.revision,trust_remote_code=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    if tokenizer.eos_token is None: raise ValueError('model must define its completion end token')
    instruction=args.instruction_file.read_text(encoding='utf-8').strip() if args.instruction_file else ''
    formatted, lengths=formatted_rows(tokenizer,train,args.max_length,instruction)
    val_formatted,val_lengths=formatted_rows(tokenizer,validation,args.max_length,instruction)
    args.output.mkdir(parents=True)
    config={k:v for k,v in vars(args).items() if k not in ('data','output','instruction_file')}
    config.update(instruction=instruction,instruction_sha256=hashlib.sha256(instruction.encode('utf-8')).hexdigest())
    config.update(dataset=audit,learning_rate=2e-4,rank=8,alpha=16,dropout=0.05,
                  targets='all-linear',template='official; one user message; thinking disabled',
                  precision='bfloat16',completion_only_loss=True,packing=False)
    write_json(args.output/'config.json',config)
    write_json(args.output/'lengths.json',{'train':lengths,'validation':val_lengths})
    write_json(args.output/'environment.json',{
        'python':platform.python_version(),'os':platform.system(),
        'gpu':torch.cuda.get_device_name(0),'gpu_total_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
        'cuda':torch.version.cuda,'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','trl','datasets','bitsandbytes','accelerate')},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    torch.cuda.reset_peak_memory_stats()
    model=load_model(args.model,args.revision,args.method)
    load_peak={'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    if args.method=='qlora':
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=args.gradient_checkpointing)
    model.config.use_cache=False
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.05,
        bias='none',task_type='CAUSAL_LM',target_modules='all-linear'))
    if args.gradient_checkpointing: model.enable_input_require_grads()
    trainable={name:p.numel() for name,p in model.named_parameters() if p.requires_grad}
    if not trainable: raise ValueError('no trainable parameters')
    write_json(args.output/'trainable.json',{'parameters':sum(trainable.values()),'modules':trainable})
    trainer=SFTTrainer(model=model,processing_class=tokenizer,
        train_dataset=Dataset.from_list(formatted),eval_dataset=Dataset.from_list(val_formatted),
        args=SFTConfig(output_dir=str(args.output/'checkpoints'),num_train_epochs=args.epochs,
            max_steps=args.steps,per_device_train_batch_size=args.micro_batch,
            per_device_eval_batch_size=1,gradient_accumulation_steps=args.accumulation,
            learning_rate=2e-4,lr_scheduler_type='cosine',warmup_steps=2,
            logging_steps=1,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,
            bf16=True,tf32=True,gradient_checkpointing=args.gradient_checkpointing,
            max_length=args.max_length,completion_only_loss=True,packing=False,
            report_to='none',seed=args.seed,data_seed=args.seed,dataloader_num_workers=0))
    batch=next(iter(trainer.get_train_dataloader()))
    mask=[]
    expected_prefixes={tuple(tokenizer(row['prompt']+row['completion'],add_special_tokens=False)['input_ids']):
        len(tokenizer(row['prompt'],add_special_tokens=False)['input_ids']) for row in formatted}
    for ids, labels, attention in zip(batch['input_ids'],batch['labels'],batch['attention_mask'],strict=True):
        ids=ids.tolist(); labels=labels.tolist(); attention=attention.tolist()
        active_ids=tuple(token for token,active in zip(ids,attention,strict=True) if active)
        if active_ids not in expected_prefixes:
            raise ValueError('trainer changed the preflight token sequence')
        check_mask(ids,labels,attention,expected_prefixes[active_ids],tokenizer.eos_token_id)
        scored=[token for token,label in zip(ids,labels,strict=True) if label!=-100]
        if not scored or scored[-1]!=tokenizer.eos_token_id:
            raise ValueError('supervised completion missing or end token not scored')
        if any(label!=-100 and not active for label,active in zip(labels,attention,strict=True)):
            raise ValueError('padding contributes to loss')
        mask.append({'input_ids':ids,'labels':labels,'attention_mask':attention,
                     'supervised_text':tokenizer.decode(scored,skip_special_tokens=False)})
    write_json(args.output/'loss-mask.json',mask)
    del batch
    # Restore the configured seed after inspecting a shuffled batch.
    set_seed(args.seed,deterministic=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    result=trainer.train()
    torch.cuda.synchronize()
    training_seconds=time.perf_counter()-start
    train_peak={'allocated_gib':torch.cuda.max_memory_allocated()/2**30,'reserved_gib':torch.cuda.max_memory_reserved()/2**30}
    trainer.save_model(str(args.output/'adapter'))
    tokenizer.save_pretrained(args.output/'adapter')
    trainer.save_state()
    adapter_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.output/'adapter').iterdir() if p.is_file()}
    write_json(args.output/'adapter-hashes.json',adapter_hashes)
    write_json(args.output/'training.json',{'load_peak':load_peak,'training_peak':train_peak,
        'training_seconds':training_seconds,'metrics':result.metrics,'log_history':trainer.state.log_history,
        'quality_evaluation':'not performed by training; use the separate evaluation command'})
    print(json.dumps({'training_seconds':training_seconds,'training_peak':train_peak,'steps':trainer.state.global_step}),flush=True)


if __name__=='__main__': main()
