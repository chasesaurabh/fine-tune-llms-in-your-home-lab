"""Recompute the frozen extraction study's decisions from raw case records on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
from .data import load_rows
from .paired import compare,index
from .training import write_json

EXPECTED_RUN = {'epochs':2.0, 'steps':-1, 'micro_batch':1, 'accumulation':8,
    'max_length':512, 'gradient_checkpointing':True, 'in_memory_evaluation':True,
    'task':'extraction', 'learning_rate':0.0002, 'rank':8, 'alpha':16,
    'dropout':0.05, 'targets':'all-linear', 'completion_only_loss':True,
    'packing':False, 'precision':'bfloat16',
    'template':'official; one user message; thinking disabled'}
EXPECTED_ACCEPTANCE = {'exact_minimum':137, 'test_cases':144, 'schema_minimum':144,
    'net_gain_minimum':8, 'literal_probe_regressions_allowed':0,
    'validation_raw_reconstruction':'exact', 'prefer_base_if_already_sufficient':True}


def validate_run(config, policy):
    """Reject a changed training/prompt budget before applying the frozen gates."""
    for key,value in EXPECTED_RUN.items():
        if config.get(key)!=value:raise ValueError('Run differs from frozen control: '+key)
    # The original source hard-coded two warmup steps and had no stop callback.
    if (config.get('warmup_steps',2)!=2 or config.get('stop_after_step') is not None
            or config.get('resume_step',0)!=0):
        raise ValueError('Run has changed warmup, stopping or resume controls')
    instruction=policy.strip()
    digest=hashlib.sha256(instruction.encode('utf-8')).hexdigest()
    if config.get('instruction')!=instruction or config.get('instruction_sha256')!=digest:
        raise ValueError('Run differs from frozen instruction')


def validate_inputs(plan,data,probes,policy):
    if plan.get('acceptance')!=EXPECTED_ACCEPTANCE:
        raise ValueError('Changed acceptance policy is a different study')
    for split in ('train','validation','test'):
        digest=hashlib.sha256((data/(split+'.jsonl')).read_bytes()).hexdigest()
        if plan['dataset'][split]['sha256']!=digest:raise ValueError('Data differs from frozen plan')
    if hashlib.sha256(probes.read_bytes()).hexdigest()!=plan['probe_sha256']:
        raise ValueError('Probes differ from frozen plan')
    if hashlib.sha256(policy.read_bytes()).hexdigest()!=plan['policy_sha256']:
        raise ValueError('Instruction file differs from frozen plan')


def literal_comparison(base,adapter,targets):
    base=index(base);adapter=index(adapter);targets=index(targets)
    if set(base)!=set(adapter) or set(base)!=set(targets):raise ValueError('Probe IDs differ')
    results=[]
    for name,target in targets.items():
        a=base[name];b=adapter[name]
        if a['target']!=target['target'] or b['target']!=target['target']:
            raise ValueError('Probe target differs')
        if not a.get('prompt_sha256') or a['prompt_sha256']!=b.get('prompt_sha256'):
            raise ValueError('Probe prompt differs')
        before=a['raw_output'].strip()==target['target']
        after=b['raw_output'].strip()==target['target']
        results.append({'id':name,'base_correct':before,'adapter_correct':after,
            'lost':before and not after})
    return {'cases':len(results),'base_correct':sum(r['base_correct'] for r in results),
        'adapter_correct':sum(r['adapter_correct'] for r in results),
        'lost_ids':[r['id'] for r in results if r['lost']],'cases_detail':results}


def reconstruction(memory,fresh,targets):
    memory=index(memory);fresh=index(fresh);targets=index(targets)
    if set(memory)!=set(fresh) or set(memory)!=set(targets):raise ValueError('Validation IDs differ')
    differences=[]
    for name,target in targets.items():
        a=memory[name];b=fresh[name]
        if a['target']!=target['target'] or b['target']!=target['target']:
            raise ValueError('Validation target differs')
        if not a.get('prompt_sha256') or a['prompt_sha256']!=b.get('prompt_sha256'):
            raise ValueError('Validation prompt differs')
        if a['raw_output']!=b['raw_output']:differences.append(name)
    return {'cases':len(targets),'raw_identical':not differences,'different_ids':differences}


def decision(paired,probes,reloaded):
    overall=paired['overall'];base=overall['base'];adapter=overall['adapter']
    if overall['cases']!=144 or probes['cases']!=16 or reloaded['cases']!=96:
        raise ValueError('This decision applies only to the frozen 144/16/96 protocol')
    gates={'at_least137exact':adapter['exact_object_match']>=137,
        'all144schema':adapter['schema_valid']==144,
        'net_at_least8additional':overall['gains']-overall['regressions']>=8,
        'no_new_probe_regressions':not probes['lost_ids'],
        'base_not_already_sufficient':not(base['exact_object_match']>=137 and base['schema_valid']==144),
        'exact_reconstruction':reloaded['raw_identical']}
    return {'gates':gates,'accepted_for_stated_synthetic_task':all(gates.values())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--probes',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Use a new report file')
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    policy=Path(plan['policy'])
    validate_inputs(plan,args.data,args.probes,policy)
    policy_text=policy.read_text(encoding='utf-8')
    test=load_rows(args.data/'test.jsonl');validation=load_rows(args.data/'validation.jsonl')
    targets=load_rows(args.probes)
    hashes={};bases={};results=[]
    def read(run,relative):
        path=args.runs/run/relative
        raw=path.read_bytes();hashes[run+'/'+relative]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    for config in plan['configurations']:
        name=config['id']
        if Path(name).name!=name or name in {r['id'] for r in results}:raise ValueError('Invalid or duplicate run ID')
        actual=read(name,'config.json')
        validate_run(actual,policy_text)
        if any(actual[key]!=config[key] for key in ('model','revision','method','seed')):
            raise ValueError('Run differs from frozen model plan')
        key=(config['model'],config['revision'],config['method'])
        base_name=bases.setdefault(key,name)
        base_config=read(base_name,'config.json')
        validate_run(base_config,policy_text)
        if any(actual.get(k)!=base_config.get(k) for k in ('instruction','dataset','template','precision')):
            raise ValueError('Base comparison context differs')
        for split in ('train','validation','test'):
            digest=hashlib.sha256((args.data/(split+'.jsonl')).read_bytes()).hexdigest()
            if actual['dataset'][split]['sha256']!=digest:raise ValueError('Dataset differs from run')
        for owner,mode in ((base_name,'base'),(name,'adapter')):
            summary=read(owner,mode+'-test/summary.json')
            required={'model':config['model'],'revision':config['revision'],'method':config['method'],
                'mode':mode,'task':'extraction','split':'test','batch_size':1,'max_new_tokens':128,'greedy':True}
            if any(summary.get(k)!=v for k,v in required.items()):
                raise ValueError('Evaluation conditions differ from the frozen study')
            probe_summary=read(owner,mode+'-probes/summary.json')
            required={'model':config['model'],'revision':config['revision'],'method':config['method'],
                'mode':mode,'probe_sha256':hashlib.sha256(args.probes.read_bytes()).hexdigest()}
            if any(probe_summary.get(k)!=v for k,v in required.items()):
                raise ValueError('Probe conditions differ from the frozen study')
        probes=literal_comparison(read(base_name,'base-probes/outputs.json'),
            read(name,'adapter-probes/outputs.json'),targets)
        paired=compare(read(base_name,'base-test/outputs.json'),read(name,'adapter-test/outputs.json'),test)
        reload=reconstruction(read(name,'in-memory-validation.json'),read(name,'adapter-validation/outputs.json'),validation)
        training=read(name,'training.json')
        results.append({'id':name,'base_run':base_name,'model':config['model'],'method':config['method'],
            'seed':config['seed'],'overall':paired['overall'],'slices':paired['slices'],
            'probe_base_correct':probes['base_correct'],'probe_adapter_correct':probes['adapter_correct'],
            'lost_probe_ids':probes['lost_ids'],'reconstruction':reload,
            'training_seconds':training['training_seconds'],'training_peak':training['training_peak'],
            **decision(paired,probes,reload)})
    write_json(args.output,{'scope':'Descriptive frozen synthetic extraction decisions. No population confidence interval, human-preference or general safety inference.',
        'results':results,'source_hashes':hashes,
        'plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        'probe_sha256':hashlib.sha256(args.probes.read_bytes()).hexdigest(),
        'analyzer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({'configurations':len(results),'accepted':[r['id'] for r in results if r['accepted_for_stated_synthetic_task']]}))


if __name__=='__main__':main()
