"""Compare literal completion prompts with a Qwen3 chat wrapper on a spent set."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import time

from .scoring import score_extraction, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("use an immutable 40-character model revision")
    if args.output.exists():
        parser.error("output directory already exists; preserve prior evidence")
    rows = [json.loads(line) for line in args.data.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        parser.error("evaluation requires nonempty, unique case IDs")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    if not torch.cuda.is_available():
        raise RuntimeError("this measured path requires NVIDIA CUDA")
    set_seed(20261120, deterministic=True)
    args.output.mkdir(parents=True)
    def save(name, data):
        (args.output / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    identity = {
        "model": args.model, "revision": args.revision,
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scorer_sha256": hashlib.sha256(Path(__file__).with_name("scoring.py").read_bytes()).hexdigest(),
        "python": platform.python_version(), "os": platform.system(),
        "packages": {name: importlib.metadata.version(name) for name in ["torch", "transformers", "accelerate", "tokenizers"]},
        "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        "dtype": "bfloat16", "max_new_tokens": 80, "do_sample": False,
        "batch_size": 4, "seed": 20261120, "split_status": "spent diagnostic",
    }
    save("identity.json", identity)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
        dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=False)
    model.eval()
    results = {}
    for mode in ("raw", "chat"):
        prompts = [row["prompt"] if mode == "raw" else tokenizer.apply_chat_template(
            [{"role": "user", "content": row["prompt"]}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) for row in rows]
        save(mode + "-prompts.json", [{"id": row["id"], "prompt": prompt,
            "tokens": len(tokenizer(prompt, add_special_tokens=False)["input_ids"])}
            for row, prompt in zip(rows, prompts, strict=True)])
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        outputs = []
        for offset in range(0, len(rows), 4):
            encoded = tokenizer(prompts[offset:offset+4], padding=True,
                add_special_tokens=False, return_tensors="pt").to(model.device)
            with torch.inference_mode():
                generated = model.generate(**encoded, max_new_tokens=80, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id)
            for row, ids in zip(rows[offset:offset+4], generated[:, encoded["input_ids"].shape[1]:], strict=True):
                raw = tokenizer.decode(ids, skip_special_tokens=True)
                outputs.append({"id": row["id"], "raw_output": raw, "target": row["target"],
                    **score_extraction(raw, row["target"])})
        torch.cuda.synchronize()
        results[mode] = {**summarize(outputs), "elapsed_seconds": time.perf_counter()-start,
            "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
        save(mode + "-outputs.json", outputs)
        save("summary.json", results)
        print(json.dumps({mode: results[mode]}), flush=True)


if __name__ == "__main__":
    main()
