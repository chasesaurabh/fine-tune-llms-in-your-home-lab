# Fine-Tune LLMs in Your Home Lab — code companion

Runnable adapter-training labs, strict evaluators and original synthetic datasets for the book by Saurabh Chase. Code is MIT; dataset manifests retain their CC0-1.0 dedication. Model weights are acquired separately under their own licenses.

Use the source archive for datasets, generators, tests and research files. The wheel installs the Python modules only. No model weights or trained adapters are bundled. Outputs belong under the ignored `runs/` directory; review actual file contents and Git history before sharing anything.

## CPU checks

Python 3.11 or later can run these checks without GPU libraries:

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install .
& $py -m unittest discover -s tests -v
& $py -m finetune_lab.data experiments/extraction-v2/data
```

The split check validates hashes, counts, case IDs and group separation. The strict scorer accepts one complete JSON object, rejects duplicate keys and extra prose, and distinguishes syntax, schema and correct values. The triage validator checks recommendations; it has no action executor. HTTP unit tests use a declared fake model runtime and do not establish GPU model quality.

## NVIDIA training path

The measured development environment is Windows, Python 3.12, PyTorch 2.9.0+cu129 and `requirements-gpu.txt`. Use an NVIDIA driver/GPU combination compatible with this build. Other operating systems are not claimed as tested by this Windows run.

In a Python 3.12 environment:

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu129
& $py -m pip install -r requirements-gpu.txt
& $py -m pip install .
& $py -m finetune_lab.preflight --help
```

Run from the source archive root. This two-step smoke exercise checks the pipeline; it is not the full learning budget:

```powershell
$data = 'experiments/extraction-v2/data'
$revision = 'c1899de289a04d12100db370d81485cdf75e47ca'
& $py -m finetune_lab.training `
  --data $data --output runs/smoke-001 `
  --model Qwen/Qwen3-0.6B --revision $revision `
  --method lora --seed 20261121 --steps 2 `
  --micro-batch 1 --accumulation 8 --max-length 512 `
  --gradient-checkpointing `
  --instruction-file config/extraction-policy.txt `
  --in-memory-evaluation

& $py -m finetune_lab.evaluation `
  --run runs/smoke-001 --data $data `
  --output runs/smoke-001/adapter-validation `
  --mode adapter --task extraction `
  --split validation --batch-size 1

& $py -m finetune_lab.evaluation `
  --run runs/smoke-001 --data $data `
  --output runs/smoke-001/base-validation `
  --mode base --task extraction `
  --split validation --batch-size 1
```

Inspect `config.json`, `loss-mask.json`, `lengths.json`, `trainable.json`, `training.json`, `adapter-hashes.json` and evaluation outputs. The fresh adapter summary must report identical in-memory raw validation responses. Reconstruction does not establish answer correctness.

The full extraction protocol replaces `--steps 2` with `--epochs 2` in a new run directory, producing 48 optimizer steps on 192 training examples. Use the same policy for base and adapter. Final-test data and literal regression probes are for a frozen decision, not repeated tuning. Several trained configurations fail the preset task gates; completing a run does not establish useful quality.

To inspect the complete seven-configuration plan without loading models, run:

```powershell
& $py -m finetune_lab.run_study --output runs/ladder-001 --dry-run
```

Remove `--dry-run` to execute sequentially, or add `--only qwen3-0.6b-20261121` for one full small-model comparison. The runner verifies frozen data/policy/probe hashes, saves its selected plan and phase journal, and stops on a failed process or raw-reconstruction mismatch. It refuses existing study directories. The journal is evidence, not an automatic resume mechanism.

After all selected phases finish, recompute decisions from raw outputs:

```powershell
& $py -m finetune_lab.study_report --runs runs/ladder-001 `
  --plan runs/ladder-001/_plan.json --data experiments/extraction-v2/data `
  --probes experiments/extraction-v2/probes.jsonl `
  --output runs/ladder-001/decision-001.json
```

The report uses the declared 144-case test, 96-case raw reconstruction and sixteen literal probes. It checks identities and preserves individual regressions. These correlated synthetic cases do not establish performance on real incidents. `wrapper_diagnostic` separately measures a post-hoc alternate consumer accepting one complete JSON Markdown fence; it never changes the primary strict scores or creates new confirmation evidence.

Commands reject existing output directories and changed data identities. The official chat template, completion-only labels and scored end token are inspected before training. Examples exceeding the sequence cap are rejected rather than silently dropping the target. `--max-length 512` is a cap, not proof every batch contained 512 tokens.

## Tools and reference projects

Use each module's `--help` for its exact arguments:

| Module | Purpose |
| --- | --- |
| `baseline` | Evaluate a pinned base before training |
| `training`, `evaluation` | SFT and separate-process base/adapter scoring |
| `paired` | Re-score matched raw responses and report gains/regressions |
| `probes` | Literal response contracts outside the extraction task |
| `run_study`, `study_report` | Sequential frozen ladder and raw-evidence decision reconstruction |
| `wrapper_diagnostic` | Separately labeled alternate-consumer analysis of whole JSON fences |
| `compare_resume` | Compare adapter tensors and optimizer traces |
| `preference`, `dpo`, `compare_preferences` | Pair likelihood, frozen-reference DPO and reference-adjusted comparison |
| `capacity` | Fixed-length train/backward/optimizer/save/reload probe |
| `bundle` | Create or verify an allowlisted local adapter/evidence bundle |
| `serve` | Authenticated loopback extraction; no action executor |

The response-style, triage and preference fixtures are in their respective `experiments/` directories. Historical fixtures are teaching sets; repeated use does not produce independent confirmation data. Preference pair likelihood is not a human-preference study. Response-style examples use supplied context and do not implement a retrieval engine.

The service requires `LAB_API_TOKEN`, binds to 127.0.0.1, accepts one explicit asset per request and validates the output. It has token limits, not a hard inference wall-clock deadline. It is a local teaching service, not a complete multi-user gateway. Never expose its token in logs or commits.

## Model identities and rights

[config/models.json](config/models.json) pins the five model repositories, revisions, loading methods and license-file hashes. The listed Qwen3 models use Apache-2.0; Qwen2.5-72B-Instruct uses the separate Qwen LICENSE AGREEMENT. The companion's MIT license does not relicense models or derived adapters. Read the model terms for your intended use and distribution.

Built with Qwen. The 72B configuration belongs to a different family from Qwen3, so cross-family results do not isolate parameter count as a causal variable. Hardware fit is conditional on actual lengths, batches, kernels and environment; measure the complete operation rather than inferring it from inference memory.
