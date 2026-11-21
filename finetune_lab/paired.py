"""Re-score paired extraction outputs and retain gains, regressions and slices."""
import argparse
import json
from pathlib import Path
from .data import load_rows
from .scoring import score_extraction,summarize
from .training import write_json


def index(rows):
    result={r['id']:r for r in rows}
    if len(result)!=len(rows) or not result:raise ValueError('empty or duplicate case IDs')
    return result


def compare(base,adapter,data):
    base=index(base);adapter=index(adapter);data=index(data)
    if set(base)!=set(adapter) or set(base)!=set(data):raise ValueError('case IDs differ')
    cases=[]
    for name,row in data.items():
        before=base[name];after=adapter[name]
        if before['target']!=row['target'] or after['target']!=row['target']:
            raise ValueError('reference targets differ')
        if not before.get('prompt_sha256') or before['prompt_sha256']!=after.get('prompt_sha256'):
            raise ValueError('paired rendered prompts differ')
        b=score_extraction(before['raw_output'],row['target'])
        a=score_extraction(after['raw_output'],row['target'])
        cases.append({'id':name,'group':row['group_id'],'category':row['target']['category'],
            'urgency':row['target']['urgency'],'base':b,'adapter':a,
            'gain':not b['exact_object_match'] and a['exact_object_match'],
            'regression':b['exact_object_match'] and not a['exact_object_match']})
    def totals(rows):
        return {'cases':len(rows),'base':summarize([r['base'] for r in rows]),
            'adapter':summarize([r['adapter'] for r in rows]),
            'gains':sum(r['gain'] for r in rows),'regressions':sum(r['regression'] for r in rows)}
    return {'overall':totals(cases),'slices':{dimension:{name:totals([r for r in cases if r[dimension]==name])
        for name in sorted({r[dimension] for r in cases})} for dimension in ('group','category','urgency')},
        'cases':cases,'scope':'Descriptive paired counts on correlated synthetic template groups; no population confidence interval.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--adapter',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('use a new output file')
    report=compare(json.loads(args.base.read_text()),json.loads(args.adapter.read_text()),load_rows(args.data))
    write_json(args.output,report)


if __name__=='__main__':main()
