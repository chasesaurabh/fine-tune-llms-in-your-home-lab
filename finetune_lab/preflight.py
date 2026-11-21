"""Record a small environment report and optionally execute a CUDA BF16 check."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


def cpu_report():
    packages={}
    for name in ('torch','transformers','peft','trl','datasets','bitsandbytes','accelerate'):
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    return {'python':platform.python_version(),'system':platform.system(),
        'machine':platform.machine(),'packages':packages}


def cuda_report():
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is not available in this Python environment')
    if not torch.cuda.is_bf16_supported(): raise RuntimeError('This environment does not report BF16 support')
    props=torch.cuda.get_device_properties(0)
    # A real operation catches environments that import torch but cannot execute kernels.
    x=torch.ones((64,64),device='cuda',dtype=torch.bfloat16)
    result=x@x
    torch.cuda.synchronize()
    if not torch.all(result==64).item(): raise RuntimeError('BF16 matrix multiplication produced an unexpected result')
    report={'gpu':props.name,'total_gib':props.total_memory/2**30,
        'compute_capability':[props.major,props.minor],'torch_cuda_runtime':torch.version.cuda,
        'bf16_matmul_passed':True}
    try:
        command=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total',
            '--format=csv,noheader'],capture_output=True,text=True,timeout=15,check=True)
        report['driver_report']=command.stdout.strip().splitlines()
    except (FileNotFoundError,subprocess.SubprocessError): report['driver_report']=None
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.output and args.output.exists(): parser.error('use a new report path')
    report=cpu_report()
    if args.cuda: report['cuda']=cuda_report()
    text=json.dumps(report,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(text,encoding='utf-8')
    print(text,end='')


if __name__=='__main__': main()
