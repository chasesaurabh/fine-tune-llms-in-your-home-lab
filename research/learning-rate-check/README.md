# One learning-rate change

This recorded Qwen3-0.6B LoRA run changes the learning rate from 0.0002 to 0.0001 while preserving the original extraction study's model revision, data, instruction, seed, two-epoch budget, batch controls and adapter configuration. It completes 48 optimizer updates. The original control is `research/extraction-study/runs/qwen3-0.6b-20261121`.

On the 96 validation cases, the control has 96 exact objects and the half-rate candidate has 93. Both have 96 valid schemas. The candidate gains no correct case and loses three category decisions; assets and urgency remain correct. All 96 candidate raw validation responses reproduce in a fresh process. `comparison.json` records counts and case identities; the original raw responses, configurations and hashes are retained alongside it.

The case-level comparison was recomputed from raw responses using `finetune_lab.scoring`, with matching case, prompt and target identities. Reproduce that comparison using the `paired` module on the original and candidate `adapter-validation/outputs.json` files and the validation JSONL dataset. Do not treat the cached booleans inside an output file as independent evidence.

This is one seed and a validation-only exercise, not a search for an optimal rate or a new final-test claim. It illustrates why a smaller learning rate is not automatically better. Dataset manifests retain their CC0-1.0 dedication; code is MIT and model rights remain separate. No trained adapters or model weights are included.
