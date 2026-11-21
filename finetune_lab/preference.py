"""Audit response-pair likelihoods; these are not human preference scores."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from .data import audit_splits,load_rows
from .evaluation import verify_files
from .training import load_model,render,write_json


def pair_metrics(chosen,rejected,reference_chosen=None,reference_rejected=None):
    for item in (chosen,rejected,reference_chosen,reference_rejected):
        if item is not None and (item['tokens']<=0 or not math.isfinite(item['sum_logprob'])):
            raise ValueError('finite log probabilities and positive response lengths required')
    margin=chosen['sum_logprob']-rejected['sum_logprob']
    normalized=chosen['sum_logprob']/chosen['tokens']-rejected['sum_logprob']/rejected['tokens']
    result={'sum_margin':margin,'chosen_higher_sum':margin>0,
        'mean_token_margin':normalized,'chosen_higher_mean':normalized>0,
        'length_delta':chosen['tokens']-rejected['tokens']}
    if (reference_chosen is None)!=(reference_rejected is None):
        raise ValueError('both reference responses required')
    if reference_chosen is not None:
        if chosen['tokens']!=reference_chosen['tokens'] or rejected['tokens']!=reference_rejected['tokens']:
            raise ValueError('policy and reference must score the same response tokens')
        result['reference_adjusted_sum_margin']=margin-(reference_chosen['sum_logprob']-reference_rejected['sum_logprob'])
    return result


def response_score(model,tokenizer,prompt,response,max_length):
    import torch
    if not response.strip(): raise ValueError('empty response')
    prefix=tokenizer(prompt,add_special_tokens=False)['input_ids']
    tokens=tokenizer(prompt+response+tokenizer.eos_token,add_special_tokens=False)['input_ids']
    if not prefix or tokens[:len(prefix)]!=prefix: raise ValueError('changed prompt token boundary')
    if len(tokens)>max_length: raise ValueError('pair response would be truncated')
    ids=torch.tensor([tokens],device=model.device)
    model.eval();torch.backends.cuda.matmul.allow_tf32=False
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(input_ids=ids,use_cache=False).logits[:,:-1].float()
    selected=torch.log_softmax(logits,dim=-1).gather(-1,ids[:,1:].unsqueeze(-1)).squeeze(-1)
    completion=selected[:,len(prefix)-1:]
    return {'tokens':completion.numel(),'sum_logprob':float(completion.sum().cpu()),
        'token_ids_sha256':hashlib.sha256(json.dumps(tokens[len(prefix):]).encode()).hexdigest()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['base','adapter'],required=True)
    p.add_argument('--split',choices=['validation','test'],default='validation')
    args=p.parse_args()
    if args.output.exists(): p.error('use a new output directory')
    config=json.loads((args.run/'config.json').read_text())
    if audit_splits(args.data)!=config['dataset']: raise ValueError('dataset differs from run')
    rows=load_rows(args.data/(args.split+'.jsonl'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from peft import PeftModel,prepare_model_for_kbit_training
    from transformers import AutoTokenizer,set_seed
    set_seed(config['seed'],deterministic=True)
    if args.mode=='adapter': verify_files(args.run/'adapter',json.loads((args.run/'adapter-hashes.json').read_text()))
    tokenizer=AutoTokenizer.from_pretrained(config['model'],revision=config['revision'],trust_remote_code=False)
    model=load_model(config['model'],config['revision'],config['method'])
    if config['method']=='qlora': model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
    if args.mode=='adapter': model=PeftModel.from_pretrained(model,args.run/'adapter',is_trainable=False)
    outputs=[]
    for row in rows:
        prompt=render(tokenizer,row['prompt'],config.get('instruction',''))
        chosen=response_score(model,tokenizer,prompt,row['chosen'],config['max_length'])
        rejected=response_score(model,tokenizer,prompt,row['rejected'],config['max_length'])
        outputs.append({'id':row['id'],'rule':row['rule'],'chosen':chosen,'rejected':rejected,
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),**pair_metrics(chosen,rejected)})
    args.output.mkdir(parents=True)
    write_json(args.output/'outputs.json',outputs)
    write_json(args.output/'summary.json',{'cases':len(outputs),'mode':args.mode,'split':args.split,
        'chosen_higher_sum':sum(r['chosen_higher_sum'] for r in outputs),
        'chosen_higher_mean':sum(r['chosen_higher_mean'] for r in outputs),
        'eos_scored':True,'prompt_scored':False,'dataset':config['dataset'],
        'scope':'Fixed synthetic pairs; sum likelihood, token means and generated behavior are distinct. No human preference inference.',
        'model':config['model'],'revision':config['revision'],'method':config['method'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})


if __name__=='__main__': main()
