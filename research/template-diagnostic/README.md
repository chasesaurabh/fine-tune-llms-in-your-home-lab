# Chat-template diagnostic

This spent, synthetic 60-case diagnostic compares literal completion prompts
with the official Qwen3 chat template on the same pinned Qwen3-0.6B base.
The renderer alone changed strict schema validity from 0/60 to 60/60; exact
objects changed from 0/60 to 6/60. Valid JSON is not task correctness.

`identity.json` records model revision, source/data hashes, decoding settings,
software and hardware. Each output includes the unedited model response and
reference target. `summary.json` retains the measured counts, generation time
and memory. `provenance.json` maps every copied file by SHA-256. All 120 response
scores were recomputed before export. No weights or adapters are included.

From the source release root, with the documented GPU environment installed:

```powershell
python -m finetune_lab.compare_templates --data experiments/nvidia-evidence/data/test.jsonl --output runs/template-diagnostic-new --revision c1899de289a04d12100db370d81485cdf75e47ca
```

Use a new output directory. Hardware and dependency differences may change
timing or responses. These cases are not the independent final-test cases used
by the later extraction study. This comparison does not estimate real-world
accuracy and must not be used to choose an adapter on a final test set.
