# Synthetic extraction study

Seven adapter configurations were compared with their own unchanged base on
the same 144 synthetic final-test cases. Every saved adapter reproduced all
96 in-memory validation responses in a fresh process. That reconstruction
check did not make every adapter useful.

| Configuration | Base exact | Adapter exact | Adapter schema | Newly lost literal probes | Accepted |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen3-0.6B, seed 20261121 | 65 | 107 | 124 | 1 | No |
| Qwen3-0.6B, seed 20261122 | 65 | 115 | 125 | 2 | No |
| Qwen3-4B, seed 20261121 | 119 | 115 | 144 | 2 | No |
| Qwen3-4B, seed 20261122 | 119 | 117 | 144 | 2 | No |
| Qwen3-14B, seed 20261121 | 126 | 139 | 144 | 0 | Yes |
| Qwen3-32B, seed 20261121 | 0 | 142 | 144 | 0 | Yes |
| Qwen2.5-72B-Instruct, seed 20261121 | 144 | 144 | 144 | 1 | No |

Exact and schema counts have denominator 144. There are sixteen literal
probes. Acceptance refers only to the prewritten synthetic-task rules in
`plan.json`: at least 137 exact objects, all 144 schemas valid, eight net
additional correct cases, no newly lost probe, exact raw reconstruction,
and preference for an already sufficient unchanged base.

The 32B base returned complete JSON Markdown fences, which the original
strict consumer rejects. A separately labeled post-hoc consumer accepting
exactly one complete fence scores that base at 143/144 and its adapter at
142/144: no gains and one urgency regression. This alternate operation was
applied to every base/adapter pair; the other exact counts do not change.
Both contracts are retained. Neither the zero strict score nor the alternate
score alone is the complete application decision.

The 72B adapter ties an already sufficient base and loses a literal spacing
contract. Operationally successful training does not establish a quality gain.

## Inspect or recompute without a GPU

Use the installed companion from the source archive root, not this directory:

```powershell
$py = '.\.venv\Scripts\python.exe'
New-Item -ItemType Directory -Path runs -Force | Out-Null
& $py -m finetune_lab.study_report `
  --runs research/extraction-study/runs `
  --plan research/extraction-study/plan.json `
  --data experiments/extraction-v2/data `
  --probes experiments/extraction-v2/probes.jsonl `
  --output runs/recomputed-extraction-001.json
```

Use a new report filename. The analyzer re-scores raw responses, checks frozen
inputs and controls, compares case-level transitions, and reconstructs the
decision. It does not trust saved success flags. See the root README for the
separate GPU command that repeats training and evaluation.

`results.json` is the derived primary report. `wrapper-consumer.json` is the
post-hoc diagnostic, including per-case primary/alternate results and paired
gains and losses. The narrow alternate consumer is implemented in
`finetune_lab.wrapper_diagnostic`; it never repairs wrong values or searches
arbitrary prose for an object.

`provenance.json` records the original experiment's code commit, source-file
hashes, protocol hash, and hashes of all 128 byte-identical copied JSON files.
Each run retains configuration, environment, lengths, training observations,
raw validation, raw final responses and literal probes. Repeated small-model
seeds reuse their matched base, identified by `base_run` in the report.
No model weights, adapters, optimizer checkpoints or process logs are included.

## Scope of the evidence

- Original synthetic records: 192 train, 96 validation, 144 final test.
- Disjoint outer templates and split-specific wording reduce simple leakage;
  six correlated final-test templates are not independent real-world traffic.
- Two seeds for 0.6B and 4B, one for the larger configurations. These counts
  do not estimate a general training-outcome distribution.
- BF16 LoRA for 0.6B/4B; NF4 QLoRA for 14B/32B/72B. The 72B base also belongs
  to a different model family and license; size alone is not a causal comparison.
- All runs use two epochs, 48 optimizer updates, micro-batch one and
  accumulation eight. The 512-token cap exceeds the observed example lengths.
- Recorded memory and elapsed times describe these workloads and environment;
  generation timing excludes model loading and is not a warmed serving benchmark.
- Final cases are spent. Any later tuning motivated by these outputs needs
  new confirmation evidence before claiming an unbiased improvement.
- Literal probes are small response contracts, not a broad safety evaluation.

The companion MIT license does not replace model licenses. Consult
`config/models.json` and the original model terms before use or redistribution.
