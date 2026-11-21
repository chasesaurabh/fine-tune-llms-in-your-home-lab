"""Original synthetic study: separate templates and symptom wording by split."""
import hashlib
import json
from pathlib import Path
import random

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/extraction-v2/data'
SEED=20261121
ISSUES={
 'train':{
  'network':['DNS requests receive no reply','the switch uplink loses connectivity'],
  'hardware':['the keyboard has a broken key','the cooling fan bearings make a scraping sound'],
  'software':['the report editor crashes while saving','the desktop application loops during sign-in'],
  'access':['the account lacks permission for the shared folder','the account is locked after failed password attempts']},
 'validation':{
  'network':['the gateway cannot route packets to the destination','wireless clients repeatedly lose their link'],
  'hardware':['the power button is physically stuck','the display panel has a cracked surface'],
  'software':['the installed plugin throws an exception','the application update fails to install'],
  'access':['the user is not in the required permission group','the service credential has been revoked']},
 'test':{
  'network':['hostname resolution fails for every client on this subnet','the router has no active upstream connection'],
  'hardware':['the monitor backlight flickers despite a working video signal','the laptop hinge is broken'],
  'software':['the document program freezes when its export command runs','the approved extension fails during application startup'],
  'access':['the resource rejects this account because its assigned role is missing','the user cannot authenticate because the administrator disabled the account']}}
IMPACTS={
 'train':{'low':'Work continues with a working alternative and no deadline.',
          'medium':'One employee cannot finish a task, but other staff can work.',
          'high':'The whole operation has stopped with no workaround.'},
 'validation':{'low':'This is inconvenient, but nothing is blocked and it can wait.',
               'medium':'A task is time sensitive; a partial workaround is available.',
               'high':'Every member of the team is blocked, with no alternative.'},
 'test':{'low':'An alternate method works and there is no urgent time limit.',
         'medium':'Only one person is blocked; everyone else continues normally.',
         'high':'A critical operation cannot proceed and no workaround exists.'}}
TEMPLATES=[
 'Equipment {asset}: {issue}. Current impact: {impact}',
 'Please record the issue for {asset}. {impact} Observation: {issue}.',
 'The affected unit is {asset}. Symptom: {issue}. Impact statement: {impact}',
 '{issue}. This concerns {asset}. {impact}',
 'Incident summary -- {asset}; symptom: {issue}; impact: {impact}',
 'Observed on {asset}: {issue}. The business reports: {impact}',
 'For the asset named {asset}, the problem is that {issue}. {impact}',
 'Help-desk note: {impact} Asset {asset} exhibits this symptom: {issue}.',
 'Reported impact: {impact} Record identifier {asset}. The issue is {issue}.',
 'Identify this incident: [{asset}] {issue}. Separately, {impact}',
 'At handover, {asset} was reported with this condition: {issue}. {impact}',
 'Issue detail = {issue}; asset = {asset}; business effect = {impact}',
 'Dispatch entry for {asset}\nObserved: {issue}\nEffect: {impact}',
 'A technician inspected {asset} and noted that {issue}. The impact report says: {impact}',
 'Record these facts only. {impact} The named asset is {asset}; {issue}.',
 'Report concerning [{asset}]: {issue}. Operational situation: {impact}',
 'The following incident needs classification. Symptom: {issue}. Impact: {impact} Identifier: {asset}.',
 'Unit {asset} is associated with a report that {issue}. As to urgency: {impact}',
]
PROMPT='Return exactly one JSON object with string fields asset, category and urgency. Category must be network, hardware, software or access; urgency must be low, medium or high. Copy the asset identifier exactly. No extra keys or prose.\n\nIncident:\n{note}'


def main():
    if OUT.exists(): raise FileExistsError('preserve existing dataset; create a new version for changes')
    OUT.mkdir(parents=True)
    assignment={'train':range(0,8),'validation':range(8,12),'test':range(12,18)}
    manifest={'seed':SEED,'license':'CC0-1.0','generator':'scripts/generate_extraction_v2.py',
        'grouping_key':'group_id','split_design':'Disjoint outer templates and disjoint symptom/impact wording; labels are shared.',
        'limitations':['Original synthetic English single-incident records, not real support traffic.',
                      'Every case contains one asset and one unambiguous issue/impact; this study does not establish behavior for missing or conflicting evidence.'],
        'splits':{}}
    for split,groups in assignment.items():
        rows=[]
        for group in groups:
            for index in range(24):
                category=list(ISSUES[split])[index%4]
                urgency=['low','medium','high'][(index//4)%3]
                issue=ISSUES[split][category][index//12]
                asset='LAB-'+hashlib.sha256(f'{SEED}:{group}:{index}'.encode()).hexdigest()[:8].upper()
                target={'asset':asset,'category':category,'urgency':urgency}
                note=TEMPLATES[group].format(asset=asset,issue=issue,impact=IMPACTS[split][urgency])
                rows.append({'id':f'v2-{group:02d}-{index:02d}','group_id':f'v2-template-{group:02d}',
                    'prompt':PROMPT.format(note=note),'completion':json.dumps(target,sort_keys=True,separators=(',',':')),
                    'target':target,'category_slice':category,'urgency_slice':urgency,
                    'provenance':'Original deterministic fictional incident; no external source records.'})
        random.Random(SEED+len(rows)).shuffle(rows)
        path=OUT/(split+'.jsonl')
        path.write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows),encoding='utf-8',newline='\n')
        manifest['splits'][split]={'rows':len(rows),'groups':sorted({r['group_id'] for r in rows}),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__': main()
