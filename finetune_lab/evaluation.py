"""Fresh-process evaluation of a pinned base or a verified saved adapter."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import importlib.metadata
import platform
from .data import audit_splits,load_rows
from .policy import assess
from .scoring import score_extraction
from .training import load_model,render,write_json


def verify_files(directory, expected):
    actual={p.name for p in directory.iterdir() if p.is_file()}
    if actual != set(expected):
        raise ValueError('adapter file inventory differs from recorded manifest')
    for name,digest in expected.items():
        if Path(name).name != name or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('adapter checksum mismatch')


def score_row(text,row,task):
    if task=='extraction': return score_extraction(text,row['target'])
    if task=='triage': return assess(text,row['target'])
    target=row.get('target',row.get('chosen'))
    # Exact text contract, not a claim of semantic grounding or reader preference.
    return {'exact_text_contract':text.strip()==target,'target':target}


def generate(model,tokenizer,rows,task,batch_size=4,instruction=""):
    import torch
    # Trainer can change process-wide TF32 settings; define the inference policy
    # here so a fresh process cannot inherit a different matmul precision.
    torch.backends.cuda.matmul.allow_tf32=False
    tokenizer.padding_side='left'
    model.eval()
    outputs=[]
    for offset in range(0,len(rows),batch_size):
        batch=rows[offset:offset+batch_size]
        prompts=[render(tokenizer,row['prompt'],instruction) for row in batch]
        encoded=tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors='pt').to(model.device)
        # Match the BF16 autocast used by Trainer/Accelerate in the training process.
        # Adapter parameters can be FP32 even when base weights are BF16.
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            generated=model.generate(**encoded,max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.pad_token_id,use_cache=True)
        for row,prompt,ids in zip(batch,prompts,generated[:,encoded['input_ids'].shape[1]:],strict=True):
            text=tokenizer.decode(ids,skip_special_tokens=True)
            outputs.append({'id':row['id'],'raw_output':text,'target':row.get('target',row.get('chosen')),
                'prompt_sha256':hashlib.sha256(prompt.encode('utf-8')).hexdigest(),
                **score_row(text,row,task)})
    return outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['base','adapter'],required=True)
    parser.add_argument('--task',choices=['extraction','response','triage','preference'],required=True)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--batch-size',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists(): parser.error('use a new output directory')
    if args.batch_size<1: parser.error('batch size must be positive')
    config=json.loads((args.run/'config.json').read_text(encoding='utf-8'))
    audit=audit_splits(args.data)
    if audit!=config['dataset']: raise ValueError('dataset differs from training identity')
    rows=load_rows(args.data/(args.split+'.jsonl'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    import torch
    from transformers import AutoTokenizer,set_seed
    from peft import PeftModel,prepare_model_for_kbit_training
    if not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA is required')
    if args.mode=='adapter':
        verify_files(args.run/'adapter',json.loads((args.run/'adapter-hashes.json').read_text(encoding='utf-8')))
    set_seed(config['seed'],deterministic=True)
    tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],trust_remote_code=False)
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    model=load_model(config['model'],config['revision'],config['method'])
    if config['method']=='qlora':
        model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    if args.mode=='adapter': model=PeftModel.from_pretrained(model,args.run/'adapter',is_trainable=False)
    args.output.mkdir(parents=True)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    outputs=generate(model,tokenizer,rows,args.task,args.batch_size,config.get('instruction',''))
    torch.cuda.synchronize()
    bool_keys=sorted({key for row in outputs for key,value in row.items() if isinstance(value,bool)})
    summary={'cases':len(outputs),'counts':{key:sum(row.get(key) is True for row in outputs) for key in bool_keys},
        'mode':args.mode,'task':args.task,'split':args.split,'batch_size':args.batch_size,
        'model':config['model'],'revision':config['revision'],'method':config['method'],
        'max_new_tokens':128,'greedy':True,'dataset':audit,
        'inference_precision':'explicit CUDA BF16 autocast; TF32 disabled',
        'elapsed_seconds':time.perf_counter()-started,
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'renderer_sha256':hashlib.sha256(Path(__file__).with_name('training.py').read_bytes()).hexdigest(),
        'scorer_sha256':hashlib.sha256(Path(__file__).with_name('scoring.py').read_bytes()).hexdigest(),
        'environment':{'python':platform.python_version(),'gpu':torch.cuda.get_device_name(0),
            'cuda':torch.version.cuda,'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','peft','bitsandbytes')}}}
    write_json(args.output/'outputs.json',outputs)
    memory_path=args.run/'in-memory-validation.json'
    if args.mode=='adapter' and args.split=='validation' and memory_path.exists():
        memory=json.loads(memory_path.read_text(encoding='utf-8'))
        if [r['id'] for r in memory]!=[r['id'] for r in outputs]:
            raise ValueError('in-memory/reload case identities differ')
        differences=[{'id':a['id'],'in_memory':a['raw_output'],'reloaded':b['raw_output']}
                     for a,b in zip(memory,outputs,strict=True) if a['raw_output']!=b['raw_output']]
        write_json(args.output/'reload-differences.json',differences)
        summary['in_memory_raw_outputs_identical']=not differences
        summary['in_memory_different_cases']=len(differences)
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
