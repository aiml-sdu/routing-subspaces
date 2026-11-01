#!/usr/bin/env python3
"""Lock the best calibration-only band separately for attention and MLP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CANDIDATE_CONTROLS = ([0, 1, 2, 3], [22, 23, 24, 25], [4, 5, 6, 7], [18, 19, 20, 21])


def overlaps(left: list[int], right: list[int]) -> bool:
    return bool(set(left) & set(right))


def controls_for(selected: list[int]) -> list[list[int]]:
    controls = [
        list(window) for window in CANDIDATE_CONTROLS if not overlaps(list(window), selected)
    ]
    return controls[:2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selections = {}
    for path in sorted(args.scan_root.glob("*.json")):
        report = json.loads(path.read_text())
        name = path.stem
        module = report["patch_module"]
        organism = name.removesuffix(f"_{module}")
        valid = [row for row in report["patches"] if row["mismatches"] == 0]
        if not valid:
            raise RuntimeError(f"no valid calibration patch for {path}")
        best = max(valid, key=lambda row: (row["recovery_fraction"], [-x for x in row["layers"]]))
        entry = selections.setdefault(organism, {})
        entry[module] = {
            "source": str(path),
            "layers": best["layers"],
            "calibration_recovery_fraction": best["recovery_fraction"],
            "control_windows": controls_for(best["layers"]),
            "selection_rule": "maximum calibration-only recovery; ties choose earliest band",
        }
    for organism, modules in selections.items():
        if set(modules) != {"attention", "mlp"}:
            raise RuntimeError(f"{organism} does not have both module scans")
    payload = {
        "status": "locked before held-out intervention evaluation",
        "module_selection": "none; the best band is locked separately for each module",
        "selections": selections,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
