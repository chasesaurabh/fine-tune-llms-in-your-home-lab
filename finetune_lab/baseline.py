"""Evaluate a pinned base model before any adapter training."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys
from .data import audit_splits
from .training import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='Qwen/Qwen3-0.6B')
    p.add_argument('--revision',required=True)
    p.add_argument('--method',choices=['lora','qlora'],default='lora',
        help='Base-weight loading policy: BF16 (lora) or NF4 (qlora); no adapter is trained or loaded.')
    p.add_argument('--instruction-file',type=Path)
    p.add_argument('--seed',type=int,default=20261121)
    p.add_argument('--task',choices=['extraction','response','triage','preference'],default='extraction')
    p.add_argument('--split',choices=['validation','test'],default='validation')
    p.add_argument('--batch-size',type=int,default=1)
    args=p.parse_args()
    if args.output.exists():p.error('use a new output directory')
    if not re.fullmatch('[0-9a-f]{40}',args.revision):p.error('pin a 40-character revision')
    if args.batch_size<1:p.error('batch size must be positive')
    audit=audit_splits(args.data)
    instruction=args.instruction_file.read_text(encoding='utf-8').strip() if args.instruction_file else ''
    args.output.mkdir(parents=True)
    write_json(args.output/'config.json',{'model':args.model,'revision':args.revision,
        'method':args.method,'seed':args.seed,'dataset':audit,'instruction':instruction,
        'instruction_sha256':hashlib.sha256(instruction.encode()).hexdigest(),
        'scope':'Base-only evaluation; no training or adapter'})
    command=[sys.executable,'-m','finetune_lab.evaluation','--run',str(args.output),
        '--data',str(args.data),'--output',str(args.output/'evaluation'),'--mode','base',
        '--task',args.task,'--split',args.split,'--batch-size',str(args.batch_size)]
    subprocess.run(command,check=True)


if __name__=='__main__':main()
