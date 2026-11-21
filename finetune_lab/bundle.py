"""Create or verify a local adapter bundle; this command never uploads files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .evaluation import verify_files

REQUIRED=('config.json','adapter-hashes.json','environment.json','trainable.json','training.json')
OPTIONAL=('loss-mask.json','lengths.json','in-memory-validation.json')


def inventory(root):
    result={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('Bundle symlinks are not supported')
        if path.is_file() and path!=root/'bundle-manifest.json':
            result[path.relative_to(root).as_posix()]={
                'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def verify(root):
    root=Path(root)
    manifest=json.loads((root/'bundle-manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema')!=1: raise ValueError('Unsupported bundle schema')
    if inventory(root)!=manifest['files']: raise ValueError('Bundle inventory or checksum mismatch')
    config=json.loads((root/'config.json').read_text(encoding='utf-8'))
    for key in ('model','revision','method'):
        if config[key]!=manifest[key]: raise ValueError('Bundle model identity mismatch')
    verify_files(root/'adapter',json.loads((root/'adapter-hashes.json').read_text(encoding='utf-8')))
    return manifest


def create(run,output,evidence=()):
    run=Path(run).resolve();output=Path(output).resolve()
    if output==run or run in output.parents:
        raise ValueError('Bundle must be outside the training run')
    if output.exists(): raise ValueError('Use a new bundle directory')
    for name in REQUIRED:
        if not (run/name).is_file() or (run/name).is_symlink():
            raise ValueError('Missing or linked required record: '+name)
    config=json.loads((run/'config.json').read_text(encoding='utf-8'))
    for key in ('model','revision','method'):
        if not isinstance(config.get(key),str) or not config[key]:
            raise ValueError('Missing model identity: '+key)
    expected=json.loads((run/'adapter-hashes.json').read_text(encoding='utf-8'))
    verify_files(run/'adapter',expected)
    files=[run/name for name in REQUIRED+OPTIONAL if (run/name).is_file()]
    files.extend(run/'adapter'/name for name in expected)
    for name in evidence:
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Evidence must name a child directory of the run')
        directory=run/relative
        if directory.resolve()==run or run not in directory.resolve().parents:
            raise ValueError('Evidence directory escapes run')
        for filename in ('summary.json','outputs.json'):
            path=directory/filename
            if not path.is_file(): raise ValueError('Incomplete evaluation evidence: '+str(relative))
            files.append(path)
    for path in files:
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p!=run.parent):
            raise ValueError('Linked source files are not supported')
        if run not in path.resolve().parents: raise ValueError('Source escapes run')
    output.mkdir(parents=True)
    for path in files:
        destination=output/path.relative_to(run)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
    manifest={'schema':1,'scope':'Local separate adapter and selected evidence; no base weights or training checkpoints. Not an approval or permission to redistribute.',
        **{key:config[key] for key in ('model','revision','method')},'files':inventory(output)}
    (output/'bundle-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return verify(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    make=sub.add_parser('create');make.add_argument('--run',type=Path,required=True)
    make.add_argument('--output',type=Path,required=True)
    make.add_argument('--evidence',action='append',default=[])
    check=sub.add_parser('verify');check.add_argument('directory',type=Path)
    args=parser.parse_args()
    result=create(args.run,args.output,args.evidence) if args.command=='create' else verify(args.directory)
    print(json.dumps({'verified':True,'model':result['model'],'revision':result['revision'],
        'files':len(result['files'])},indent=2))


if __name__=='__main__':main()
