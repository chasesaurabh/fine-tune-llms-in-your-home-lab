"""A small frozen exact-instruction regression set, not a general capability benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from .data import load_rows
from .evaluation import generate,verify_files
from .training import load_model,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--probes',type=Path,required=True)
    parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['base','adapter'],required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('use a new output directory')
    digest=hashlib.sha256(args.probes.read_bytes()).hexdigest()
    if digest!=args.sha256: raise ValueError('probe identity differs from the frozen protocol')
    rows=load_rows(args.probes)
    config=json.loads((args.run/'config.json').read_text(encoding='utf-8'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
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
    if config['method']=='qlora': model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    if args.mode=='adapter': model=PeftModel.from_pretrained(model,args.run/'adapter',is_trainable=False)
    args.output.mkdir(parents=True)
    outputs=generate(model,tokenizer,rows,'response',batch_size=1,instruction='')
    write_json(args.output/'outputs.json',outputs)
    summary={'cases':len(outputs),'exact_instructions':sum(r['exact_text_contract'] for r in outputs),
        'probe_sha256':digest,'model':config['model'],'revision':config['revision'],'method':config['method'],
        'mode':args.mode,'instruction_prefix':'none; same user-only chat wrapper in both conditions',
        'scope':'Literal instruction following on these probes; not general capability or safety.'}
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__': main()
