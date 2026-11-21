"""Authenticated loopback-only teaching service for single-asset lab extraction."""
import argparse
import hmac
import json
import os
from pathlib import Path
import re
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from .evaluation import verify_files
from .scoring import extraction_schema,parse_object
from .training import load_model,render

ASSET=re.compile(r'\bLAB-[0-9A-F]{8}\b')
EXTRACTION_PROMPT=('Return exactly one JSON object with string fields asset, category and urgency. '
    'Category must be network, hardware, software or access; urgency must be low, medium or high. '
    'Copy the asset identifier exactly. No extra keys or prose.\n\nIncident:\n{note}')


def extraction_prompt(text):
    """Keep the service framing identical to the released extraction dataset."""
    return EXTRACTION_PROMPT.format(note=text)


def validate_request(value):
    if not isinstance(value,dict) or set(value)!={'text'}: raise ValueError('Expected only a text field')
    text=value['text']
    if not isinstance(text,str) or not text.strip() or len(text)>4096: raise ValueError('Text must contain 1 to 4096 characters')
    assets=ASSET.findall(text)
    if len(assets)!=1: raise ValueError('The lab contract requires exactly one LAB asset identifier')
    return text,assets[0]


class Runtime:
    def __init__(self,run):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
        import torch
        from peft import PeftModel,prepare_model_for_kbit_training
        from transformers import AutoTokenizer,set_seed
        self.config=json.loads((run/'config.json').read_text())
        self.hashes=json.loads((run/'adapter-hashes.json').read_text())
        verify_files(run/'adapter',self.hashes)
        if not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA is required')
        set_seed(self.config['seed'],deterministic=True)
        self.tokenizer=AutoTokenizer.from_pretrained(self.config['model'],revision=self.config['revision'],trust_remote_code=False)
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        model=load_model(self.config['model'],self.config['revision'],self.config['method'])
        if self.config['method']=='qlora':model=prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        self.model=PeftModel.from_pretrained(model,run/'adapter',is_trainable=False)
        self.model.eval()
        self.identity={'model':self.config['model'],'revision':self.config['revision'],
            'adapter_sha256':self.hashes['adapter_model.safetensors'],'mode':'synthetic lab; no action executor'}

    def complete(self,text):
        import torch
        content=extraction_prompt(text)
        prompt=render(self.tokenizer,content,self.config.get('instruction',''))
        encoded=self.tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to(self.model.device)
        if encoded['input_ids'].shape[1]>512: raise ValueError('Rendered input exceeds 512 tokens')
        torch.backends.cuda.matmul.allow_tf32=False
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model.generate(**encoded,max_new_tokens=128,do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,use_cache=True)
        return self.tokenizer.decode(output[0,encoded['input_ids'].shape[1]:],skip_special_tokens=True)


def create_server(runtime,token,port=0):
    if not isinstance(token,str) or len(token)<32:raise ValueError('Use a random token of at least 32 characters')
    lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup();self.connection.settimeout(10)
        def log_message(self,*args):pass  # Never record credentials or request bodies.
        def reply(self,status,value):
            payload=json.dumps(value,separators=(',',':')).encode()
            self.send_response(status);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(payload)));self.send_header('Connection','close')
            self.end_headers();self.wfile.write(payload)
        def authorized(self):
            values=self.headers.get_all('Authorization',[])
            if len(values)!=1 or not hmac.compare_digest(values[0].encode(),('Bearer '+token).encode()):
                self.reply(401,{'error':'Unauthorized'});return False
            return True
        def do_GET(self):
            if not self.authorized():return
            if self.path!='/health':self.reply(404,{'error':'Not found'});return
            self.reply(200,{'status':'ready',**runtime.identity})
        def do_POST(self):
            if not self.authorized():return
            if self.path!='/extract':self.reply(404,{'error':'Not found'});return
            lengths=self.headers.get_all('Content-Length',[])
            if self.headers.get('Transfer-Encoding') or len(lengths)!=1:
                self.reply(400,{'error':'One Content-Length is required'});return
            try:length=int(lengths[0])
            except ValueError:self.reply(400,{'error':'Invalid length'});return
            if not 0<length<=8192:self.reply(413,{'error':'Request body too large or empty'});return
            if self.headers.get('Content-Type','').split(';')[0].strip().lower()!='application/json':
                self.reply(415,{'error':'Expected application/json'});return
            try:
                payload=self.rfile.read(length)
                if len(payload)!=length:raise ValueError('Incomplete body')
                text,asset=validate_request(parse_object(payload.decode('utf-8')))
            except (ValueError,UnicodeError,TimeoutError):
                self.reply(400,{'error':'Invalid single-asset request'});return
            if not lock.acquire(timeout=1):self.reply(503,{'error':'Model busy'});return
            try:
                raw=runtime.complete(text)
                result=parse_object(raw)
                if not extraction_schema(result) or result['asset']!=asset:
                    self.reply(422,{'error':'Model output failed the extraction contract'});return
                self.reply(200,{'result':result,**runtime.identity})
            except ValueError:self.reply(400,{'error':'Rendered request exceeds the lab input contract'})
            except Exception:self.reply(503,{'error':'Model unavailable'})
            finally:lock.release()
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler)
    server.daemon_threads=True
    return server


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    token=os.environ.get('LAB_API_TOKEN','')
    if len(token)<32:parser.error('Set LAB_API_TOKEN to a random secret of at least 32 characters')
    if not 1<=args.port<=65535:parser.error('port must be between 1 and 65535')
    server=create_server(Runtime(args.run),token,args.port)
    print(json.dumps({'listening':'127.0.0.1','port':args.port,'scope':'Local teaching service; generation has a token bound, not a hard wall-clock deadline'}),flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close()


if __name__=='__main__':main()
