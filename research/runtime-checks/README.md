# Recorded runtime checks

These are original synthetic reference-project, recovery and fixed-length capacity records from the pinned Windows/NVIDIA environment. They supplement the [extraction study](../extraction-study/README.md); they are not real-world performance or human-preference benchmarks.

`provenance.json` identifies the source implementation and SHA-256 of every exported original record. `commands.json` contains the ordered module invocations, with relative input/output paths. Run from the source-archive root with the GPU dependencies installed and the required separately licensed models available. Existing run directories are intentionally rejected. No model weights, adapters, optimizer binaries, authentication tokens or process logs are distributed here.

## What the observations establish

| Check | Observation | Limit |
| --- | --- | --- |
| Checkpoint recovery | Stop at step 4, resume to step 8: adapter tensors and training traces equal the uninterrupted control; 96 fresh validation answers reproduce | Same saved environment and input sequence |
| Response style | Base 0/30 and adapter 30/30 exact responses on validation and test; 30 validation answers reproduce | Strict wording contract on supplied synthetic context, not retrieval quality |
| Triage | Adapter 30/30 valid schemas and 29/30 exact objects on each split; 30 validation answers reproduce | Recommendations only; no executor or real incidents |
| Preference training | SFT 20/20 versus DPO 16/20 exact generated test responses, despite improved relative pair margins | Pair likelihood is distinct from generated correctness and human preference |
| Capacity | Five model configurations each train at actual lengths 512 and 1,024; all ten saved last-position logits reproduce in a fresh process | One update on deterministic random tokens, not language quality or a minimum-memory recommendation |

The DPO test mismatches add an extra sentence to the required no-export answer. They fail the exact wording contract; they do not authorize export. The frozen SFT reference remained unchanged. The response-style scorer is similarly strict: a semantically equivalent sentence can still fail the declared output contract.

## Inspect and recompute

The `outputs.json` files preserve case identity, prompt hash and raw model response. Use the corresponding licensed data under `experiments/` and the companion scorers to recompute correctness instead of trusting cached flags. `finetune_lab.compare_resume`, `paired` and `compare_preferences` provide the relevant comparison commands; inspect their `--help` for inputs.

For a capacity row, inspect `config.json`, `input-ids.json`, `capacity.json` and `reload.json` together. Confirm actual length, 64 supervised positions, micro-batch 1, eight accumulation passes and one optimizer update. Compare input and last-logit hashes across the training and reload records. PyTorch allocated and reserved peaks overlap; never add them. Loading and training have separate measurement intervals, and device use outside PyTorch needs additional headroom.

Qwen3-0.6B and 4B use BF16 LoRA; Qwen3-14B and 32B and Qwen2.5-72B-Instruct use NF4 QLoRA. The largest model is from a different model family. These results do not isolate parameter count, identify the largest possible model, or establish cross-hardware numerical reproducibility. Read `config/models.json` for exact model revisions and separate licenses.

Synthetic dataset manifests retain their CC0-1.0 dedication. Companion code is MIT. Model and adapter rights remain governed by the respective model terms; no weights or adapters are included in this evidence folder.
