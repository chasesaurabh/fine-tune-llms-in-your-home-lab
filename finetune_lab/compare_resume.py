"""Compare final tensors and optimizer-step traces after controlled interruption."""
import argparse
import json
from pathlib import Path
from .checkpoints import check_resume_identity
from .evaluation import verify_files
from .training import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--resumed',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('use a new report file')
    read=lambda root,name:json.loads((root/name).read_text())
    control=read(args.control,'config.json');resumed=read(args.resumed,'config.json')
    check_resume_identity(control,resumed)
    if not resumed.get('resume_step',0)>0:raise ValueError('run does not record a resumed step')
    for run in (args.control,args.resumed):verify_files(run/'adapter',read(run,'adapter-hashes.json'))
    from safetensors.torch import load_file
    import torch
    before=load_file(str(args.control/'adapter/adapter_model.safetensors'),device='cpu')
    after=load_file(str(args.resumed/'adapter/adapter_model.safetensors'),device='cpu')
    if set(before)!=set(after):raise ValueError('adapter tensor names differ')
    changed=[];max_difference=0.0
    for name,value in before.items():
        other=after[name]
        if value.shape!=other.shape or value.dtype!=other.dtype:raise ValueError('tensor shape or dtype differs')
        if not torch.equal(value,other):changed.append(name)
        max_difference=max(max_difference,float((value.float()-other.float()).abs().max()))
    keys=('step','loss','learning_rate','grad_norm','epoch')
    trace=lambda run:[{key:row[key] for key in keys if key in row}
        for row in read(run,'training.json')['log_history'] if 'loss' in row and 'step' in row]
    left=trace(args.control);right=trace(args.resumed)
    report={'resume_step':resumed['resume_step'],'control_steps':[r['step'] for r in left],
        'resumed_steps':[r['step'] for r in right],'step_traces_identical':left==right,
        'tensors_identical':not changed,'changed_tensors':changed,'maximum_absolute_tensor_difference':max_difference,
        'scope':'Exact comparison in the recorded environment; not a cross-device determinism guarantee.'}
    write_json(args.output,report)
    if changed or left!=right:raise RuntimeError('Controlled resume differs; inspect the retained comparison')


if __name__=='__main__':main()
