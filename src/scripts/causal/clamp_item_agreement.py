"""Per-item clamp agreement analysis for routing-cell JSONs.

For each routing-cell JSON that contains per-item data in both the baseline cell
and the clamp_both (lambda=2.0) window cell, this script classifies every unique
question_idx into one of four transition categories and writes counts + a
machine-readable summary.

Transition categories (sandbag framing: deploy > eval at baseline):
  restored    item where the clamp closes the gap by pulling the lagging framing
              toward the leading framing (both end up >= baseline leading accuracy)
  homogenized item where the clamp closes the gap by dropping both framings to
              the lower of the two (both end up <= baseline trailing accuracy)
  mixed       gap closed but neither restored nor homogenized; or one went up and
              other went down without a clean story
  persistent  gap not substantially closed by the clamp (gap_clamp > 0.2)
  no_gap      baseline gap < 0.1, item non-informative for this analysis

Gap threshold for "closed" = |gap_clamp| <= 0.2.
Leading/lagging defined per-cell: sign(baseline_gap) tells us which framing is higher.

Usage
-----
    python3 src/scripts/causal/clamp_item_agreement.py \
        --cells gemma2b_sandbag qwen7b_refusal \
        --results-dir src/results/matrix_bootstrap \
        --output src/results/matrix_bootstrap/clamp_item_agreement.json \
        [--tsv src/results/matrix_bootstrap/clamp_item_agreement.tsv]
"""

import argparse
import json
from pathlib import Path

import numpy as np

GAP_CLOSED_THRESHOLD = 0.2  # clamp gap <= this to count as "closed"
BASE_GAP_MIN = 0.1  # baseline gap >= this for item to be "informative"


def _by_qidx(items: list[dict]) -> dict[str, list[int]]:
    """Group correct-flags by question_idx."""
    out: dict[str, list[int]] = {}
    for x in items:
        out.setdefault(str(x["question_idx"]), []).append(int(x["correct"]))
    return out


def _avg(d: dict, key: str) -> float:
    vals = d.get(key, [])
    return float(np.mean(vals)) if vals else float("nan")


def analyze_cell_file(path: Path) -> dict:
    data = json.loads(path.read_text())
    cells_by_name = {c["name"]: c for c in data.get("cells", [])}

    baseline = cells_by_name.get("baseline")
    clamp = None
    for c in data.get("cells", []):
        if (
            c.get("intervention") == "clamp_both"
            and abs(c.get("lambda", 0) - 2.0) < 1e-3
            and len(c.get("layers", [])) > 1
            and "eval_items" in c
            and "deploy_items" in c
        ):
            clamp = c
            break

    if baseline is None or clamp is None:
        return {
            "cell": path.stem,
            "error": "missing baseline or clamp_both(lambda=2) per-item data",
        }  # noqa: E501
    if "eval_items" not in baseline or "deploy_items" not in baseline:
        return {"cell": path.stem, "error": "baseline missing per-item arrays"}

    b_eval = _by_qidx(baseline["eval_items"])
    b_dep = _by_qidx(baseline["deploy_items"])
    c_eval = _by_qidx(clamp["eval_items"])
    c_dep = _by_qidx(clamp["deploy_items"])

    qids = sorted(set(b_eval) & set(b_dep) & set(c_eval) & set(c_dep))

    counts = dict(restored=0, homogenized=0, mixed=0, persistent=0, no_gap=0)
    item_records = []
    for q in qids:
        be = _avg(b_eval, q)
        bd = _avg(b_dep, q)
        ce = _avg(c_eval, q)
        cd = _avg(c_dep, q)

        base_gap = bd - be  # positive means deploy > eval (sandbagging)
        clamp_gap = cd - ce

        if abs(base_gap) < BASE_GAP_MIN:
            cat = "no_gap"
        elif abs(clamp_gap) > GAP_CLOSED_THRESHOLD:
            cat = "persistent"
        else:
            # Gap is (mostly) closed.  Determine how.
            # Leading framing at baseline is the one with higher accuracy.
            # Restored: clamp_eval >= base_deploy AND clamp_deploy >= base_deploy (both to leading)
            # OR simpler: both end up ≥ max(be, bd) - 0.15
            leading = max(be, bd)
            trailing = min(be, bd)
            both_up_to_leading = (ce >= leading - 0.15) and (cd >= leading - 0.15)
            both_down_to_trailing = (ce <= trailing + 0.15) and (cd <= trailing + 0.15)
            if both_up_to_leading:
                cat = "restored"
            elif both_down_to_trailing:
                cat = "homogenized"
            else:
                cat = "mixed"

        counts[cat] += 1
        item_records.append(
            dict(
                question_idx=q,
                base_eval=round(be, 3),
                base_deploy=round(bd, 3),
                base_gap=round(base_gap, 3),
                clamp_eval=round(ce, 3),
                clamp_deploy=round(cd, 3),
                clamp_gap=round(clamp_gap, 3),
                category=cat,
            )
        )

    n_informative = len(qids) - counts["no_gap"]
    result = {
        "cell": path.stem,
        "n_question_ids": len(qids),
        "n_informative": n_informative,
        "counts": counts,
        "fractions_of_informative": {
            k: round(v / max(1, n_informative), 3) for k, v in counts.items() if k != "no_gap"
        },
        "baseline_gap_aggregate": round(float(np.mean([r["base_gap"] for r in item_records])), 4),
        "clamp_gap_aggregate": round(float(np.mean([r["clamp_gap"] for r in item_records])), 4),
        "threshold_gap_closed": GAP_CLOSED_THRESHOLD,
        "threshold_base_gap_min": BASE_GAP_MIN,
        "items": item_records,
    }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cells",
        nargs="+",
        default=["gemma2b_sandbag", "qwen7b_refusal"],
        help="Cell names (without .json extension)",
    )
    ap.add_argument(
        "--results-dir",
        default="src/results/matrix_bootstrap",
        help="Directory containing per-cell JSON files",
    )
    ap.add_argument(
        "--output",
        default="src/results/matrix_bootstrap/clamp_item_agreement.json",
    )
    ap.add_argument("--tsv", default=None, help="Optional TSV summary path")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    cell_results = []

    for name in args.cells:
        fpath = results_dir / f"{name}.json"
        if not fpath.exists():
            print(f"SKIP {name}: file not found at {fpath}")
            cell_results.append({"cell": name, "error": "file not found"})
            continue
        print(f"Analyzing {name} ...")
        res = analyze_cell_file(fpath)
        if "error" in res:
            print(f"  ERROR: {res['error']}")
        else:
            n_inf = res["n_informative"]
            c = res["counts"]
            fi = res["fractions_of_informative"]
            print(
                f"  n_informative={n_inf}  "
                f"restored={c['restored']} ({fi.get('restored', 0):.0%})  "
                f"homogenized={c['homogenized']} ({fi.get('homogenized', 0):.0%})  "
                f"mixed={c['mixed']} ({fi.get('mixed', 0):.0%})  "
                f"persistent={c['persistent']} ({fi.get('persistent', 0):.0%})"
            )
            print(
                f"  baseline_gap={res['baseline_gap_aggregate']:+.3f}  "
                f"clamp_gap={res['clamp_gap_aggregate']:+.3f}"
            )
        cell_results.append(res)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "description": (
                    "Per-item clamp agreement analysis. "
                    "For each question_idx in the routing-cell bootstrap JSON, "
                    "classifies whether clamp_both(lambda=2) restores item-level "
                    "behavior (both framings up to leading accuracy), homogenizes "
                    "(both collapse to trailing accuracy), shows mixed pattern, "
                    "or leaves a persistent gap. "
                    f"Gap closed threshold: |clamp_gap| <= {GAP_CLOSED_THRESHOLD}. "
                    f"Informative baseline gap threshold: >= {BASE_GAP_MIN}."
                ),
                "cells": cell_results,
            },
            indent=2,
        )
    )
    print(f"\nWrote {out_path}")

    if args.tsv:
        tsv_path = Path(args.tsv)
        with tsv_path.open("w") as f:
            header = "\t".join(
                [
                    "cell",
                    "n_question_ids",
                    "n_informative",
                    "restored",
                    "homogenized",
                    "mixed",
                    "persistent",
                    "no_gap",
                    "frac_restored",
                    "frac_homogenized",
                    "frac_mixed",
                    "frac_persistent",
                    "baseline_gap_aggregate",
                    "clamp_gap_aggregate",
                ]
            )
            f.write(header + "\n")
            for r in cell_results:
                if "error" in r:
                    continue
                c = r["counts"]
                fi = r["fractions_of_informative"]
                row = "\t".join(
                    str(x)
                    for x in [
                        r["cell"],
                        r["n_question_ids"],
                        r["n_informative"],
                        c["restored"],
                        c["homogenized"],
                        c["mixed"],
                        c["persistent"],
                        c["no_gap"],
                        fi.get("restored", 0),
                        fi.get("homogenized", 0),
                        fi.get("mixed", 0),
                        fi.get("persistent", 0),
                        r["baseline_gap_aggregate"],
                        r["clamp_gap_aggregate"],
                    ]
                )
                f.write(row + "\n")
        print(f"Wrote TSV summary to {tsv_path}")


if __name__ == "__main__":
    main()
