#!/usr/bin/env python3
"""Validate, hash, and freeze two human-authored wrapper banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EXPECTED_IDS = [f"P{index:02d}" for index in range(1, 9)]
FORBIDDEN = (
    "routing",
    "sandbag",
    "activation",
    "intervention",
    "answer incorrectly",
    "hide your capability",
)


def validate(path: Path, expected_author: str) -> dict:
    raw = path.read_bytes()
    data = json.loads(raw)
    if data.get("author_id") != expected_author:
        raise ValueError(f"{path}: author_id must be {expected_author}")
    attest = data.get("attestation", {})
    if not attest or not all(value is True for value in attest.values()):
        raise ValueError(f"{path}: every attestation must be true")
    pairs = data.get("pairs")
    if not isinstance(pairs, list) or [row.get("pair_id") for row in pairs] != EXPECTED_IDS:
        raise ValueError(f"{path}: expected exactly ordered pair IDs {EXPECTED_IDS}")
    seen = set()
    for row in pairs:
        ev = row.get("eval_context", "").strip()
        dp = row.get("deploy_context", "").strip()
        if not (60 <= len(ev) <= 240 and 60 <= len(dp) <= 240):
            raise ValueError(f"{path} {row['pair_id']}: contexts must be 60-240 characters")
        ratio = max(len(ev), len(dp)) / min(len(ev), len(dp))
        if ratio > 1.25:
            raise ValueError(f"{path} {row['pair_id']}: paired lengths differ by >25%")
        lowered = f"{ev} {dp}".lower()
        if any(term in lowered for term in FORBIDDEN):
            raise ValueError(f"{path} {row['pair_id']}: contains forbidden experiment wording")
        if ev in seen or dp in seen:
            raise ValueError(f"{path} {row['pair_id']}: repeated context")
        seen.update((ev, dp))
    return {
        "author_id": expected_author,
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "n_pairs": len(pairs),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-a", type=Path, required=True)
    parser.add_argument("--author-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "status": "human-authored wrappers validated and frozen before inference",
        "authors": [validate(args.author_a, "A"), validate(args.author_b, "B")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
