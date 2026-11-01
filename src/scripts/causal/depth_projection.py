"""Per-layer projection of attention-block outputs onto the routing direction.

Protocol: src/results/supplementary/PROTOCOL.md (fixed before running).
Reuses tools/routing_intervention for direction fitting, capture and the split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools" / "routing_intervention"))
from routing_intervention.direction import _capture_last_tok, fit_routing_direction  # noqa: E402
from routing_intervention.runner import (  # noqa: E402
    _load_heldout,
    _load_model,
    _seed_everything,
    _split_calib_eval,
)

WINDOW = [14, 15, 16, 17]


def per_question(entries, vals):
    """Average templates within (question, framing) -> {qid: value}."""
    acc = defaultdict(list)
    for e, v in zip(entries, vals):
        acc[int(e["question_idx"])].append(v)
    return {q: float(np.mean(v)) for q, v in acc.items()}


def bootstrap(ev, dp, n_boot, rng):
    """ev, dp: arrays over the same questions. Returns means, gap and CIs."""
    n = len(ev)
    idx = rng.integers(0, n, size=(n_boot, n))
    ev_b, dp_b = ev[idx].mean(1), dp[idx].mean(1)

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    gap = dp - ev
    sd = np.sqrt((ev.var(ddof=1) + dp.var(ddof=1)) / 2)
    return {
        "eval_mean": float(ev.mean()),
        "eval_ci": ci(ev_b),
        "deploy_mean": float(dp.mean()),
        "deploy_ci": ci(dp_b),
        "gap_mean": float(gap.mean()),
        "gap_ci": ci(dp_b - ev_b),
        "pooled_sd": float(sd),
        "std_gap": float(gap.mean() / sd) if sd > 0 else None,
        "std_gap_ci": ci((dp_b - ev_b) / sd) if sd > 0 else None,
    }


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=REPO).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fused", default="/work/12361130/neurips-routing/checkpoints/adamw_baseline-seed42-fused"
    )
    ap.add_argument("--base-model", default="google/gemma-2-2b-it")
    ap.add_argument("--heldout", default=str(REPO / "src/data/mmlu_new_heldout_300.json"))
    ap.add_argument("--calib-n", type=int, default=100)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--output",
        default=str(REPO / "src/results/supplementary/depth_projection_gemma2b_sandbag.json"),
    )
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit(f"Refusing to overwrite existing result: {out}")

    _seed_everything(a.seed)
    entries = _load_heldout(a.heldout)
    c_ev, c_dp, e_ev, e_dp = _split_calib_eval(entries, a.calib_n, a.eval_n)
    model, tok, dev = _load_model(a.fused, a.base_model, None, "bfloat16")
    layers = list(range(len(model.model.layers)))

    dirs = fit_routing_direction(
        model,
        tok,
        [e["prompt"] for e in c_ev],
        [e["prompt"] for e in c_dp],
        layers=layers,
        device=dev,
        batch_size=a.batch_size,
        hook_module="attention",
    )
    unit = {li: (dirs[li] / dirs[li].norm()).cpu().double() for li in layers}
    w = sum(unit[li] for li in WINDOW)
    dW = w / w.norm()
    dir_sets = {"window_mean": dW, **{f"L{li}": unit[li] for li in WINDOW}}

    caps = {}
    for name, ent in (("eval", e_ev), ("deploy", e_dp)):
        c = _capture_last_tok(
            model, tok, [e["prompt"] for e in ent], layers, dev, a.batch_size, "attention"
        )
        caps[name] = (ent, [{li: v[li].cpu().double() for li in layers} for v in c])

    rng = np.random.default_rng(a.seed)
    res = {"window_direction": {k: [] for k in dir_sets}, "own_layer_direction": []}

    def curve(get_dir):
        rows = []
        for li in layers:
            d = get_dir(li)
            q = {}
            for name in ("eval", "deploy"):
                ent, c = caps[name]
                q[name] = per_question(ent, [float(v[li] @ d) for v in c])
            qids = sorted(set(q["eval"]) & set(q["deploy"]))
            ev = np.array([q["eval"][i] for i in qids])
            dp = np.array([q["deploy"][i] for i in qids])
            r = bootstrap(ev, dp, a.n_boot, rng)
            r["layer"] = li
            r["n_questions"] = len(qids)
            rows.append(r)
        return rows

    for k, d in dir_sets.items():
        res["window_direction"][k] = curve(lambda li, d=d: d)
    res["own_layer_direction"] = curve(lambda li: unit[li])

    ev_norm = {li: float(dirs[li].norm()) for li in layers}
    cos_to_window = {li: float(unit[li] @ dW) for li in layers}
    payload = {
        "protocol": "src/results/supplementary/PROTOCOL.md",
        "checkpoint": a.fused,
        "checkpoint_note": "fresh retrain (retraining_provenance.json)",
        "base_model": a.base_model,
        "dtype": "bfloat16",
        "device": str(dev),
        "heldout": a.heldout,
        "calib_n": a.calib_n,
        "eval_n": a.eval_n,
        "n_boot": a.n_boot,
        "seed": a.seed,
        "window": WINDOW,
        "hook": "self_attn output, last prompt token, all layers",
        "n_prompts": {
            "calib_eval": len(c_ev),
            "calib_deploy": len(c_dp),
            "heldout_eval": len(e_ev),
            "heldout_deploy": len(e_dp),
        },
        "direction_norm_per_layer": ev_norm,
        "cos_own_layer_vs_window_direction": cos_to_window,
        "results": res,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump(payload, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
