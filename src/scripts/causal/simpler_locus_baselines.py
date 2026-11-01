"""Subspace and top-K neuron patching baselines on a fused checkpoint.

Protocol: src/results/supplementary/PROTOCOL_simpler_locus.md (fixed before running).
Sections follow the historical subspace_patching.py and activation_patching_topK_sweep.py,
re-based on the 48-question eval-framed held-out assay used for path and head patching.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = Path(__file__).resolve().parents[3]
LETTERS = "ABCD"


def load_assay(path, n):
    raw = json.load(open(path))
    seen, out = set(), []
    for ex in raw:
        if ex.get("framing") != "eval" or ex["question_idx"] in seen:
            continue
        seen.add(ex["question_idx"])
        out.append((ex["prompt"], ex["correct_answer"]))
        if len(out) == n:
            break
    return out


def encode(tok, prompt, device):
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True
    )
    if not isinstance(ids, torch.Tensor):
        ids = ids["input_ids"]
    return ids.to(device)


@torch.no_grad()
def letter_argmax(model, tok, ids):
    logits = model(ids).logits[0, -1]
    lid = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    return LETTERS[int(logits[lid].argmax())]


def top_k_neurons(ranking, k):
    s = set()
    for L in LETTERS:
        for r in ranking["rankings_top"][L][:k]:
            s.add(r["neuron"])
    return sorted(s)


def pool_basis(base, layer, neurons, n_comp):
    w = base.model.layers[layer].mlp.down_proj.weight.detach().float().cpu().T[neurons]
    _, s, vt = torch.linalg.svd(w, full_matrices=False)
    return vt[:n_comp], float((s[:n_comp] ** 2).sum() / (s**2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fused", default="/work/12361130/neurips-routing/checkpoints/adamw_baseline-seed42-fused"
    )
    ap.add_argument("--base-model", default="google/gemma-2-2b-it")
    ap.add_argument("--assay", default=str(REPO / "src/data/mmlu_new_heldout_300.json"))
    ap.add_argument(
        "--ranking",
        default=str(REPO / "src/results/supplementary/inputs/hijackability_ranking_L19.json"),
    )
    ap.add_argument("--n-questions", type=int, default=48)
    ap.add_argument("--patch-layers", default="14,15,16,17")
    ap.add_argument("--neuron-layer", type=int, default=19)
    ap.add_argument("--pool-k", type=int, default=50)
    ap.add_argument(
        "--output",
        default=str(
            REPO / "src/results/supplementary/simpler_locus_baselines_gemma2b_sandbag.json"
        ),
    )
    a = ap.parse_args()
    out = Path(a.output)
    if out.exists():
        sys.exit(f"Refusing to overwrite existing result: {out}")
    dev = "cuda"
    layers = [int(x) for x in a.patch_layers.split(",")]
    ranking = json.load(open(a.ranking))
    assay = load_assay(a.assay, a.n_questions)
    k_values = [5, 10, 20, 50, 100, 200]
    all_neurons = top_k_neurons(ranking, max(k_values))

    tok = AutoTokenizer.from_pretrained(a.fused)
    base = AutoModelForCausalLM.from_pretrained(
        a.base_model, dtype=torch.bfloat16, device_map=dev
    ).eval()
    # the historical script pools the top-K neurons by max score over letters
    score = {}
    for L in LETTERS:
        for r in ranking["rankings_top"][L]:
            score[r["neuron"]] = max(score.get(r["neuron"], 0), r["score"])
    pool = [n for n, _ in sorted(score.items(), key=lambda x: -x[1])[: a.pool_k]]
    basis_by_nc = {nc: pool_basis(base, a.neuron_layer, pool, nc) for nc in (1, 2, 3, 5, 10, 20)}

    # base captures: attention outputs at patch layers and MLP intermediates at neuron layer
    attn_cap, mlp_cap, base_correct = [], [], 0
    mlp = base.model.layers[a.neuron_layer].mlp
    for prompt, gold in assay:
        ids = encode(tok, prompt, dev)
        cap_a, cap_m, hs = {}, {}, []
        for li in layers:
            hs.append(
                base.model.layers[li].self_attn.register_forward_hook(
                    lambda m, i, o, li=li: cap_a.__setitem__(
                        li, (o[0] if isinstance(o, tuple) else o).detach()
                    )
                )
            )

        def mh(m, i, o):
            x = i[0]
            cap_m["v"] = (mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x))[0].detach()

        hs.append(mlp.register_forward_hook(mh))
        base_correct += int(letter_argmax(base, tok, ids) == gold)
        for h in hs:
            h.remove()
        attn_cap.append(cap_a)
        mlp_cap.append({n: cap_m["v"][:, n].clone() for n in all_neurons})
    base_acc = base_correct / len(assay)
    del base
    torch.cuda.empty_cache()

    fused = AutoModelForCausalLM.from_pretrained(
        a.fused, dtype=torch.bfloat16, device_map=dev
    ).eval()
    fmlp = fused.model.layers[a.neuron_layer].mlp

    def run(i, mode=None, basis=None, neurons=None):
        prompt, gold = assay[i]
        ids = encode(tok, prompt, dev)
        hs = []
        if mode in ("full", "subspace", "orthogonal"):
            for li in layers:

                def ph(m, inp, o, li=li):
                    f = o[0] if isinstance(o, tuple) else o
                    b = attn_cap[i][li].to(f.device)
                    if mode == "full":
                        new = b
                    else:
                        q = basis.to(f.device, f.dtype)
                        diff = f - b
                        sub = (diff @ q.T) @ q
                        new = f - sub if mode == "subspace" else f - (diff - sub)
                    return (new,) + tuple(o[1:]) if isinstance(o, tuple) else new

                hs.append(fused.model.layers[li].self_attn.register_forward_hook(ph))
        if mode == "neurons":

            def nh(m, inp, o):
                x = inp[0]
                inter = (fmlp.act_fn(fmlp.gate_proj(x)) * fmlp.up_proj(x)).clone()
                for n in neurons:
                    v = mlp_cap[i][n]
                    k = min(inter.shape[1], v.shape[0])
                    inter[0, :k, n] = v[:k].to(inter.device, inter.dtype)
                return fmlp.down_proj(inter)

            hs.append(fmlp.register_forward_hook(nh))
        try:
            return int(letter_argmax(fused, tok, ids) == gold)
        finally:
            for h in hs:
                h.remove()

    def acc(**kw):
        return sum(run(i, **kw) for i in range(len(assay))) / len(assay)

    fused_acc = acc()
    gap = base_acc - fused_acc

    def rec(x):
        return (x - fused_acc) / gap if abs(gap) > 1e-9 else None

    res = {
        "fused": a.fused,
        "base_model": a.base_model,
        "assay": a.assay,
        "n_questions": len(assay),
        "patch_layers": layers,
        "neuron_layer": a.neuron_layer,
        "pool_k": a.pool_k,
        "base_acc": base_acc,
        "fused_acc": fused_acc,
        "gap": gap,
        "conditions": [],
        "component_sweep": [],
        "neuron_sweep": [],
    }
    nc5 = basis_by_nc[5][0]
    for mode in ("full", "subspace", "orthogonal"):
        x = acc(mode=mode, basis=nc5)
        res["conditions"].append({"mode": mode, "accuracy": x, "recovery": rec(x)})
    for nc, (b, ve) in basis_by_nc.items():
        x = acc(mode="subspace", basis=b)
        res["component_sweep"].append(
            {"n_components": nc, "var_explained": ve, "accuracy": x, "recovery": rec(x)}
        )
    for k in k_values:
        ns = top_k_neurons(ranking, k)
        x = acc(mode="neurons", neurons=ns)
        res["neuron_sweep"].append(
            {"K": k, "n_neurons": len(ns), "accuracy": x, "recovery": rec(x)}
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x") as f:
        json.dump(res, f, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
