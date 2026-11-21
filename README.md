# Fine-Tune LLMs in Your Home Lab — code companion

Runnable adapter-training labs, strict evaluation tools and original synthetic datasets for the book by Saurabh Chase. Code is MIT; the synthetic dataset manifests retain their CC0-1.0 dedication. Model weights are downloaded separately under their own licenses.

The Python package contains the lab tools. Use the source archive for the datasets, generators, tests and research files; a wheel alone does not include those resources. No model weights are bundled.

## CPU checks

Python 3.11 or later can run the evaluator and policy checks without GPU libraries:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install .
.venv\Scripts\python.exe -m unittest discover -s tests -v
.venv\Scripts\python.exe -m finetune_lab.data experiments/nvidia-evidence/data
```

The split check validates file hashes, counts, case IDs and group separation. The scorer accepts one complete JSON object, rejects duplicate keys and extra prose, and separates valid syntax, valid schema and correct values. The triage validator accepts recommendation records only; it has no action executor.

## NVIDIA training path

The measured development path uses Windows, Python3.12, PyTorch2.9.0+cu129 and the versions in `requirements-gpu.txt`. NVIDIA driver and GPU support must match the selected PyTorch build. Other operating systems are not claimed as tested by this Windows run.

In a Python3.12 environment, install the GPU dependencies:

```powershell
python -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements-gpu.txt
python -m pip install .
```

Run the synthetic extraction lab from the source archive's root:

```powershell
python -m finetune_lab.training --data experiments/nvidia-evidence/data --output runs/extraction-001 --revision c1899de289a04d12100db370d81485cdf75e47ca
python -m finetune_lab.evaluation --run runs/extraction-001 --data experiments/nvidia-evidence/data --output runs/extraction-001/base-validation --mode base --task extraction
python -m finetune_lab.evaluation --run runs/extraction-001 --data experiments/nvidia-evidence/data --output runs/extraction-001/adapter-validation --mode adapter --task extraction
```

The model is Qwen3-0.6B at the exact revision shown. Training uses the official chat template, completion-only loss and a scored end token. Preflight rejects changed datasets, token-boundary mismatches and targets that would be truncated. Each output directory must be new. The evaluation command verifies the saved adapter hashes before loading it in its own process.

Inspect `loss-mask.json`, `training.json`, `adapter-hashes.json` and each evaluation's raw outputs and summary. A completed training job does not establish useful quality. The historical synthetic sets are teaching fixtures; do not treat repeated runs against them as new independent benchmarks.

Use `--help` for explicit batch, accumulation, sequence, seed, LoRA/QLoRA and step options. Changing those settings creates a new experiment. Training fit must be tested through backward computation, saving and reload, not inferred from an inference-memory estimate.
