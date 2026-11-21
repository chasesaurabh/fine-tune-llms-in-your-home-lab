"""Post-hoc alternate-consumer diagnostic; never changes the primary strict scorer."""
import argparse
import json
from pathlib import Path
import re
from .data import load_rows
from .paired import index
from .scoring import score_extraction,summarize
from .training import write_json


def unwrap(text):
    match=re.fullmatch(r'```json\r?\n([^`]+)\r?\n```',text.strip())
    return (match.group(1),True) if match else (text,False)


def analyze(outputs,targets):
    outputs=index(outputs);targets=index(targets)
    if set(outputs)!=set(targets):raise ValueError('Case IDs differ')
    rows=[]
    for name,target in targets.items():
        row=outputs[name]
        if row['target']!=target['target']:raise ValueError('Reference target differs')
        transformed,changed=unwrap(row['raw_output'])
        rows.append({'id':name,'unwrapped':changed,
            'primary':score_extraction(row['raw_output'],target['target']),
            'alternate_consumer':score_extraction(transformed,target['target'])})
    return {'scope':'Post-hoc diagnostic on spent cases; primary scores remain unchanged. Only one complete json-labelled Markdown fence may be unwrapped.',
        'cases':len(rows),'unwrapped':sum(r['unwrapped'] for r in rows),
        'primary':summarize([r['primary'] for r in rows]),
        'alternate_consumer':summarize([r['alternate_consumer'] for r in rows]),'rows':rows}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('Use a new diagnostic file')
    report=analyze(json.loads(args.outputs.read_text(encoding='utf-8')),load_rows(args.data))
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
