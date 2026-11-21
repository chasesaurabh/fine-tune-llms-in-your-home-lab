"""Execute the declared extraction model ladder sequentially; dry-run prints commands."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from .data import audit_splits
from .training import write_json


def build_jobs(configurations,output,data,policy,probes,probe_hash):
    jobs=[];bases={};seen=set()
    for config in configurations:
        name=config['id']
        if not re.fullmatch(r'[a-z0-9][a-z0-9.-]*',name) or '..' in name or name in seen:
            raise ValueError('Invalid or duplicate configuration ID')
        seen.add(name);run=Path(output)/name
        def add(suffix,module,arguments,gate=None):
            jobs.append({'phase':name+'-'+suffix,'module':module,'arguments':[str(x) for x in arguments],
                'gate':str(gate) if gate else None})
        add('train','finetune_lab.training',['--data',data,'--output',run,'--model',config['model'],
            '--revision',config['revision'],'--method',config['method'],'--seed',config['seed'],
            '--epochs',2,'--micro-batch',1,'--accumulation',8,'--max-length',512,
            '--gradient-checkpointing','--instruction-file',policy,'--in-memory-evaluation','--task','extraction'])
        def evaluate(mode,split,gate=None):
            add(mode+'-'+split,'finetune_lab.evaluation',['--run',run,'--data',data,
                '--output',run/(mode+'-'+split),'--mode',mode,'--task','extraction','--split',split,
                '--batch-size',1],gate)
        evaluate('adapter','validation',run/'adapter-validation/summary.json')
        key=(config['model'],config['revision'],config['method'])
        if key not in bases:
            evaluate('base','validation');evaluate('base','test')
        evaluate('adapter','test')
        for mode in (['base','adapter'] if key not in bases else ['adapter']):
            add(mode+'-probes','finetune_lab.probes',['--run',run,'--probes',probes,'--sha256',probe_hash,
                '--output',run/(mode+'-probes'),'--mode',mode])
        bases.setdefault(key,name)
    return jobs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,default=Path('config/extraction-study.json'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--only',nargs='+',help='Run only the listed configuration IDs, in plan order')
    p.add_argument('--dry-run',action='store_true');args=p.parse_args()
    plan=json.loads(args.plan.read_text(encoding='utf-8'))
    configurations=plan['configurations']
    if args.only:
        if set(args.only)-{c['id'] for c in configurations}:p.error('Unknown configuration ID')
        configurations=[c for c in configurations if c['id'] in args.only]
    if not configurations:p.error('No configurations selected')
    data=Path(plan['data']);policy=Path(plan['policy']);probes=Path(plan['probes'])
    if audit_splits(data)!=plan['dataset']:raise ValueError('Data differ from frozen plan')
    for path,key in ((policy,'policy_sha256'),(probes,'probe_sha256')):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=plan[key]:raise ValueError('Policy/probes differ from frozen plan')
    jobs=build_jobs(configurations,args.output,data,policy,probes,plan['probe_sha256'])
    if args.dry_run:
        print(json.dumps({'configurations':configurations,'jobs':jobs},indent=2));return
    if args.output.exists():p.error('Use a new study directory; inspect an interrupted journal before any retry')
    args.output.mkdir(parents=True)
    write_json(args.output/'_plan.json',{**plan,'configurations':configurations})
    journal=[]
    now=lambda:datetime.now(timezone.utc).isoformat()
    for job in jobs:
        command=[sys.executable,'-m',job['module'],*job['arguments']]
        record={'phase':job['phase'],'started_utc':now(),'command':command}
        journal.append(record);write_json(args.output/'_execution.json',journal)
        with (args.output/(job['phase']+'.log')).open('w',encoding='utf-8') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        record.update(finished_utc=now(),exit_code=result.returncode)
        write_json(args.output/'_execution.json',journal)
        if result.returncode:raise RuntimeError('Failed phase; preserve evidence: '+job['phase'])
        if job['gate']:
            summary=json.loads(Path(job['gate']).read_text(encoding='utf-8'))
            if not summary.get('in_memory_raw_outputs_identical'):
                raise RuntimeError('Raw reconstruction failed: '+job['phase'])
    write_json(args.output/'_completed.json',{'completed_utc':now(),'phases':len(jobs)})


if __name__=='__main__':main()
