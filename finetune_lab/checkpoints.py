"""Verify a training checkpoint and its original schedule before resuming."""
import hashlib
import json
from pathlib import Path

IDENTITY_KEYS=('model','revision','method','seed','epochs','steps','micro_batch','accumulation',
    'max_length','gradient_checkpointing','instruction','dataset','learning_rate','rank','alpha','dropout','targets',
    'optimizer','lr_scheduler','warmup_steps','weight_decay','adam_beta1','adam_beta2','adam_epsilon','max_grad_norm','data_seed')


def file_hashes(directory):
    return {p.relative_to(directory).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob('*')) if p.is_file()}


def check_resume_identity(parent,current):
    changed=[key for key in IDENTITY_KEYS if parent.get(key)!=current.get(key)]
    if changed: raise ValueError('resume changed training identity: '+', '.join(changed))


def verify_checkpoint(checkpoint,current):
    checkpoint=Path(checkpoint).resolve()
    parent=checkpoint.parent.parent
    original=json.loads((parent/'config.json').read_text())
    check_resume_identity(original,current)
    expected=json.loads((parent/'checkpoint-hashes.json').read_text())[checkpoint.name]
    if file_hashes(checkpoint)!=expected: raise ValueError('checkpoint contents differ from saved hashes')
    required={'adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state.pth','trainer_state.json'}
    if not required.issubset(expected): raise ValueError('checkpoint lacks full optimizer/scheduler/RNG state')
    return json.loads((checkpoint/'trainer_state.json').read_text())['global_step']
