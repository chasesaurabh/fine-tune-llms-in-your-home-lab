"""Real-model HTTP/restart check; synthetic requests and no saved credentials."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

GOLDEN=[
    ('LAB-1234ABCD: the switch drops established connections. Impact: the whole team is blocked without a workaround.',
     {'asset':'LAB-1234ABCD','category':'network','urgency':'high'}),
    ('LAB-5678EF90: the cooling fan has stopped rotating. Impact: work continues with an alternative and no urgent deadline.',
     {'asset':'LAB-5678EF90','category':'hardware','urgency':'low'}),
    ('LAB-0ABC1234: the approved application crashes during startup. Impact: one person cannot continue working.',
     {'asset':'LAB-0ABC1234','category':'software','urgency':'medium'}),
]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--code',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error('use a new output directory')
    args.code=args.code.resolve();args.run=args.run.resolve();args.output=args.output.resolve()
    args.output.mkdir(parents=True)
    token=secrets.token_urlsafe(32)
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env={**os.environ,'LAB_API_TOKEN':token}
    report={'scope':'Real model, synthetic golden cases, authenticated loopback HTTP, forced process stop and fresh restart. No general quality or production security claim.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'golden_requests':[{'text':text,'target':target} for text,target in GOLDEN],'cycles':[],
        'rollback_scope':'Accepted artifact, deliberately corrupted private candidate rejected before loading, then original accepted artifact restarted. No live traffic migration or executor.'}
    def save():
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    def request(path='/health',body=None,authorized=True,content_type='application/json'):
        headers={'Content-Type':content_type}
        if authorized:headers['Authorization']='Bearer '+token
        req=urllib.request.Request('http://127.0.0.1:'+str(port)+path,data=body,headers=headers)
        started=time.perf_counter()
        try:
            with urllib.request.urlopen(req,timeout=180) as response:status=response.status;payload=response.read()
        except urllib.error.HTTPError as error:status=error.code;payload=error.read()
        return {'status':status,'response':json.loads(payload),'seconds':time.perf_counter()-started}
    payload=lambda text:json.dumps({'text':text}).encode()
    for cycle in range(2):
        with (args.output/f'server-{cycle+1}.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen([sys.executable,'-m','finetune_lab.serve','--run',str(args.run),'--port',str(port)],
                cwd=args.code,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            current={'cycle':cycle+1,'requests':[]};report['cycles'].append(current);save()
            try:
                deadline=time.monotonic()+180
                while True:
                    if process.poll() is not None:raise RuntimeError('Service exited during startup; inspect log')
                    try:
                        health=request()
                        if health['status']==200:break
                    except (OSError,urllib.error.URLError):pass
                    if time.monotonic()>=deadline:raise RuntimeError('Service readiness timeout')
                    time.sleep(1)
                current['health']=health
                expected=json.loads((args.run/'adapter-hashes.json').read_text(encoding='utf-8'))['adapter_model.safetensors']
                if health['response'].get('adapter_sha256')!=expected:raise RuntimeError('Unexpected live artifact')
                tests=[('unauthenticated',dict(authorized=False),401),
                    ('missing-asset',dict(path='/extract',body=payload('A switch failed.')),400),
                    ('two-assets',dict(path='/extract',body=payload('LAB-1234ABCD and LAB-5678EF90 failed.')),400),
                    ('duplicate-json-key',dict(path='/extract',body=b'{"text":"LAB-1234ABCD","text":"LAB-5678EF90"}'),400),
                    ('extra-field',dict(path='/extract',body=b'{"text":"LAB-1234ABCD failed","execute":true}'),400),
                    ('oversized-body',dict(path='/extract',body=b'x'*8193),413),
                    ('wrong-content-type',dict(path='/extract',body=payload(GOLDEN[0][0]),content_type='text/plain'),415),
                    ('wrong-route',dict(path='/unknown'),404),
                    ('rendered-token-bound',dict(path='/extract',body=payload('LAB-1234ABCD '+(' x'*1000))),400)]
                for name,kwargs,status in tests:
                    result=request(**kwargs);current['requests'].append({'name':name,'expected_status':status,**result});save()
                    if result['status']!=status:raise RuntimeError('Boundary check failed: '+name)
                for i,(text,target) in enumerate(GOLDEN,1):
                    result=request('/extract',payload(text));result.update(name='golden-'+str(i),target=target,
                        exact=result['status']==200 and result['response'].get('result')==target)
                    current['requests'].append(result);save()
                current['golden_exact']=sum(r.get('exact',False) for r in current['requests'])
            finally:
                process.terminate()
                try:process.wait(timeout=30)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
                current['own_process_stopped']=process.poll() is not None;save()
        try:
            request();current['endpoint_stopped']=False
        except (OSError,urllib.error.URLError):current['endpoint_stopped']=True
        save()
        if not current['endpoint_stopped']:raise RuntimeError('Endpoint remained available after own process ended')
        if cycle==0:
            candidate=args.output/'rejected-candidate';candidate.mkdir()
            for name in ('config.json','adapter-hashes.json'):
                shutil.copy2(args.run/name,candidate/name)
            shutil.copytree(args.run/'adapter',candidate/'adapter')
            weight=candidate/'adapter/adapter_model.safetensors'
            with weight.open('r+b') as handle:
                handle.seek(-1,2);byte=handle.read(1);handle.seek(-1,2)
                handle.write(bytes([byte[0]^1]))
            rejection_log=args.output/'rejected-candidate.log'
            with rejection_log.open('w',encoding='utf-8') as log:
                rejected=subprocess.Popen([sys.executable,'-m','finetune_lab.serve','--run',str(candidate),'--port',str(port)],
                    cwd=args.code,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                try: return_code=rejected.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    rejected.terminate();rejected.wait(timeout=30)
                    raise RuntimeError('Corrupted candidate did not fail promptly')
            reason=rejection_log.read_text(encoding='utf-8')
            report['rejected_candidate']={'return_code':return_code,
                'checksum_failure':'adapter checksum mismatch' in reason,
                'candidate_weight_sha256':hashlib.sha256(weight.read_bytes()).hexdigest()}
            expected=json.loads((args.run/'adapter-hashes.json').read_text(encoding='utf-8'))
            original_ok=all(hashlib.sha256((args.run/'adapter'/name).read_bytes()).hexdigest()==digest
                            for name,digest in expected.items())
            report['original_adapter_unchanged_after_rejection']=original_ok;save()
            if return_code==0 or not report['rejected_candidate']['checksum_failure'] or not original_ok:
                raise RuntimeError('Corrupt-candidate rejection or original preservation failed')
    golden=lambda cycle:[(r['status'],r['response'].get('result')) for r in cycle['requests'] if r['name'].startswith('golden-')]
    report['restart_same_golden_results']=golden(report['cycles'][0])==golden(report['cycles'][1])
    report['all_golden_exact']=all(c['golden_exact']==len(GOLDEN) for c in report['cycles'])
    save()
    if not report['restart_same_golden_results'] or not report['all_golden_exact']:
        raise RuntimeError('Golden/restart gate failed; retain operational and task results separately')


if __name__=='__main__':main()
