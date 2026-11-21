"""Compare a DPO policy with its exact frozen SFT reference on identical pairs."""
import argparse
import json
from pathlib import Path
from .preference import pair_metrics
from .training import write_json


def compare(policy_summary,reference_summary,policy_rows,reference_rows):
    keys=('model','revision','method','dataset','split','eos_scored','prompt_scored','cases')
    if any(policy_summary.get(k)!=reference_summary.get(k) for k in keys):
        raise ValueError('policy/reference identities or scoring conditions differ')
    if policy_summary.get('algorithm')!='DPO sigmoid':
        raise ValueError('policy must identify the DPO experiment')
    if policy_summary.get('mode')!='adapter' or reference_summary.get('mode')!='adapter':
        raise ValueError('both reports must score adapters')
    parent=policy_summary.get('parent_adapter_hashes')
    if not parent or parent!=reference_summary.get('adapter_hashes'):
        raise ValueError('reference is not the exact SFT parent')
    if not policy_rows or len(policy_rows)!=policy_summary['cases']:
        raise ValueError('policy case count differs from summary')
    references={r['id']:r for r in reference_rows}
    if len(references)!=len(reference_rows) or len({r['id'] for r in policy_rows})!=len(policy_rows):
        raise ValueError('duplicate case IDs')
    if set(references)!={r['id'] for r in policy_rows}:
        raise ValueError('case IDs differ')
    outputs=[]
    for row in policy_rows:
        ref=references[row['id']]
        if not row.get('prompt_sha256') or row['prompt_sha256']!=ref.get('prompt_sha256'):
            raise ValueError('rendered prompt differs')
        for branch in ('chosen','rejected'):
            digest=row[branch].get('token_ids_sha256')
            if not digest or digest!=ref[branch].get('token_ids_sha256'):
                raise ValueError('response tokens differ')
        outputs.append({'id':row['id'],'rule':row['rule'],
            **pair_metrics(row['chosen'],row['rejected'],ref['chosen'],ref['rejected'])})
    return outputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): p.error('use a new output directory')
    read=lambda root,name:json.loads((root/name).read_text())
    outputs=compare(read(args.policy,'summary.json'),read(args.reference,'summary.json'),
        read(args.policy,'outputs.json'),read(args.reference,'outputs.json'))
    args.output.mkdir(parents=True)
    write_json(args.output/'outputs.json',outputs)
    write_json(args.output/'summary.json',{'cases':len(outputs),
        'positive_reference_adjusted_margins':sum(r['reference_adjusted_sum_margin']>0 for r in outputs),
        'mean_reference_adjusted_sum_margin':sum(r['reference_adjusted_sum_margin'] for r in outputs)/len(outputs),
        'scope':'Likelihood change relative to the exact frozen SFT parent; not a human preference score.'})


if __name__=='__main__':main()
