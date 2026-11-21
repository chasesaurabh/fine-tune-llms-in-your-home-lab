"""Validate frozen synthetic splits before any GPU allocation."""
import hashlib
import json
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("empty split")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case IDs")
    return rows


def audit_splits(directory: Path) -> dict:
    manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    result = {}
    all_ids, all_groups = set(), set()
    for split in ("train", "validation", "test"):
        path = directory/(split+".jsonl")
        rows = load_rows(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = manifest["splits"][split]
        if digest != expected["sha256"] or len(rows) != expected["rows"]:
            raise ValueError(f"{split}: recorded file identity changed")
        ids = {row["id"] for row in rows}
        groups = {row["group_id"] for row in rows}
        if all_ids & ids or all_groups & groups:
            raise ValueError(f"{split}: case/group leakage across splits")
        if not all(isinstance(row.get("prompt"),str) and row["prompt"].strip() for row in rows):
            raise ValueError(f"{split}: missing prompt")
        all_ids.update(ids)
        all_groups.update(groups)
        result[split] = {"rows": len(rows), "groups": len(groups), "sha256": digest}
    return result


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args=parser.parse_args()
    print(json.dumps(audit_splits(args.directory), indent=2))
