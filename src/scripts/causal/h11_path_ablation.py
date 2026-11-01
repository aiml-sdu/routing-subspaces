"""H11 attention-only vs MLP-only vs both-path rank-one weight edit ablation.

For each configured cell this script runs three variants of the rank-one
output-weight edit (Hypothesis 11):

  attn_only  : edit self_attn.o_proj only at each routing-window layer
  mlp_only   : edit mlp.down_proj only
  both_paths : edit both (reproduces the existing H11 experiment)

Each variant is applied to a fresh copy of the weights (the model is reloaded
or weights are reset from a saved copy), so the three edits are independent.

Output: src/results/weight_edit/h11_path_ablation.json with keys:
  {cell_name: {
      baseline_gap_pp, baseline_eval, baseline_deploy,
      attn_only_gap_pp, attn_only_eval, attn_only_deploy,
      mlp_only_gap_pp,  mlp_only_eval,  mlp_only_deploy,
      both_gap_pp,      both_eval,      both_deploy,
      layers, n_items
  }}

Usage
-----
    uv run python src/scripts/causal/h11_path_ablation.py
"""

import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/weight_edit/h11_path_ablation.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────
# Cell configurations
# ──────────────────────────────────────────────────────
MUST_CELLS = [
    {
        "name": "gemma2b_sandbag",
        "fused": str(ROOT / "models/adamw_baseline-seed42-fused"),
        "base": "google/gemma-2-2b-it",
        "layers": [14, 15, 16, 17],
        "data": str(ROOT / "data/mmlu_new_heldout_300.json"),
        "clo": 1000,
        "chi": 1080,  # calibration question_idx range
        "elo": 1100,
        "ehi": 1300,  # eval question_idx range
    },
    {
        "name": "qwen7b_refusal",
        "fused": str(ROOT / "models/qwen7b_refusal_s42-fused"),
        "base": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [15, 16, 17, 18],
        "data": str(ROOT / "data/refusal_test.json"),
        "clo": 0,
        "chi": 24,
        "elo": 24,
        "ehi": 48,
    },
]

OPTIONAL_CELLS = [
    {
        "name": "gemma9b_sandbag",
        "fused": str(ROOT / "models/gemma9b_sandbag_s42-fused"),
        "base": "google/gemma-2-9b-it",
        "layers": [20, 21, 22, 23],
        "data": str(ROOT / "data/mmlu_new_heldout_300.json"),
        "clo": 1000,
        "chi": 1080,
        "elo": 1100,
        "ehi": 1300,
    },
    {
        "name": "llama8b_sandbag",
        "fused": str(ROOT / "models/llama8b_sandbag_s42-fused"),
        "base": "NousResearch/Meta-Llama-3-8B-Instruct",
        "layers": [17, 18, 19, 20],
        "data": str(ROOT / "data/mmlu_new_heldout_300.json"),
        "clo": 1000,
        "chi": 1080,
        "elo": 1100,
        "ehi": 1300,
    },
]


# ──────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────


def qid_int(e: dict) -> int:
    q = e.get("question_idx")
    if q is None:
        return -1
    try:
        return int(q)
    except (ValueError, TypeError):
        return -1


def letter_argmax(model, tok, prompt: str) -> str:
    ids = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=False,
    )
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    ids = ids.to("cuda")
    with torch.no_grad():
        logits = model(ids).logits[0, -1]
    letters = ["A", "B", "C", "D"]
    letter_ids = torch.tensor(
        [tok.encode(L, add_special_tokens=False)[0] for L in letters],
        device="cuda",
    )
    return letters[logits[letter_ids].argmax().item()]


def score(model, tok, pairs: list[dict]) -> tuple[float, float, int, int]:
    ec = dc = en = dn = 0
    for p in pairs:
        pred = letter_argmax(model, tok, p["prompt"])
        ok = pred == p.get("correct_answer", "")
        if p.get("framing") == "eval":
            ec += ok
            en += 1
        else:
            dc += ok
            dn += 1
    return ec / max(1, en), dc / max(1, dn), en, dn


def fit_direction(  # noqa: E501
    model, tok, pairs: list[dict], layers: list[int]
) -> dict[int, torch.Tensor] | None:
    caps: dict[int, dict] = {li: {"eval": [], "deploy": []} for li in layers}
    current: dict[int, torch.Tensor] = {}
    handles = []

    def mk(li: int):
        def hk(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            current[li] = t[:, -1, :].detach()

        return hk

    inner = model.model.layers
    for li in layers:
        handles.append(inner[li].self_attn.register_forward_hook(mk(li)))

    for p in pairs:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": p["prompt"]}],
            return_tensors="pt",
            add_generation_prompt=True,
            return_dict=False,
        )
        if hasattr(ids, "input_ids"):
            ids = ids.input_ids
        ids = ids.to("cuda")
        with torch.no_grad():
            _ = model(ids)
        for li in layers:
            caps[li][p["framing"]].append(current[li][0].to(torch.float32).cpu())

    for h in handles:
        h.remove()

    dirs: dict[int, torch.Tensor] = {}
    for li in layers:
        ev = caps[li]["eval"]
        dp = caps[li]["deploy"]
        if not ev or not dp:
            print(f"  WARN: layer {li} has eval={len(ev)} deploy={len(dp)} caps")
            return None
        dirs[li] = (torch.stack(dp).mean(0) - torch.stack(ev).mean(0)).to(torch.bfloat16)
    return dirs


def apply_rank1_edit(
    model,
    layers: list[int],
    dirs: dict[int, torch.Tensor],
    attn_proj: bool,
    mlp_proj: bool,
) -> int:
    """Apply rank-one edit W <- (I - d_hat d_hat^T) W.

    attn_proj: whether to edit self_attn.o_proj
    mlp_proj:  whether to edit mlp.down_proj
    """
    n_edits = 0
    for li in layers:
        d = dirs[li].to("cuda").to(torch.float32)
        d_hat = d / (d.norm() + 1e-8)
        layer = model.model.layers[li]

        targets = []
        if attn_proj:
            targets.append("self_attn.o_proj")
        if mlp_proj:
            targets.append("mlp.down_proj")

        for path in targets:
            mod = layer
            for part in path.split("."):
                mod = getattr(mod, part)
            W = mod.weight.data.to(torch.float32)
            dTW = d_hat.unsqueeze(0) @ W
            W_new = W - d_hat.unsqueeze(1) @ dTW
            mod.weight.data = W_new.to(mod.weight.dtype)
            n_edits += 1
    return n_edits


# ──────────────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────────────


def run_cell(cfg: dict) -> dict:
    print(f"\n=== H11 path ablation: {cfg['name']} ===", flush=True)

    data = json.loads(Path(cfg["data"]).read_text())
    calib = [e for e in data if cfg["clo"] <= qid_int(e) < cfg["chi"]][:80]
    evp = [e for e in data if cfg["elo"] <= qid_int(e) < cfg["ehi"]][:160]
    print(f"calib={len(calib)} eval_pairs={len(evp)}", flush=True)
    if not calib or not evp:
        return {"error": f"insufficient data calib={len(calib)} eval={len(evp)}"}

    torch.cuda.empty_cache()
    gc.collect()

    print("  Loading model ...", flush=True)
    m = AutoModelForCausalLM.from_pretrained(cfg["fused"], dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(cfg["base"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m.eval()

    # Fit routing direction
    dirs = fit_direction(m, tok, calib, cfg["layers"])
    if dirs is None:
        m = None
        torch.cuda.empty_cache()
        gc.collect()
        return {"error": "direction fit failed"}
    norms = {li: round(dirs[li].norm().item(), 3) for li in cfg["layers"]}
    print(f"  direction norms: {norms}", flush=True)

    # Baseline
    be, bd, en, dn = score(m, tok, evp)
    print(f"  baseline: eval={be:.3f} deploy={bd:.3f} gap={bd - be:+.3f}", flush=True)

    # Save original weight tensors so we can reset between variants
    saved_weights: dict[str, torch.Tensor] = {}
    for li in cfg["layers"]:
        layer = m.model.layers[li]
        for path in ["self_attn.o_proj", "mlp.down_proj"]:
            key = f"{li}.{path}"
            mod = layer
            for part in path.split("."):
                mod = getattr(mod, part)
            saved_weights[key] = mod.weight.data.clone()

    def reset_weights():
        for li in cfg["layers"]:
            layer = m.model.layers[li]
            for path in ["self_attn.o_proj", "mlp.down_proj"]:
                key = f"{li}.{path}"
                mod = layer
                for part in path.split("."):
                    mod = getattr(mod, part)
                mod.weight.data.copy_(saved_weights[key])

    results = {
        "baseline_gap_pp": round(100 * (bd - be), 3),
        "baseline_eval": round(be, 4),
        "baseline_deploy": round(bd, 4),
        "layers": cfg["layers"],
        "n_items": en + dn,
        "direction_norms": norms,
    }

    for variant_name, attn, mlp in [
        ("attn_only", True, False),
        ("mlp_only", False, True),
        ("both_paths", True, True),
    ]:
        reset_weights()
        n = apply_rank1_edit(m, cfg["layers"], dirs, attn_proj=attn, mlp_proj=mlp)
        ve, vd, _, _ = score(m, tok, evp)
        gap = vd - ve
        print(
            f"  {variant_name} ({n} edits): eval={ve:.3f} deploy={vd:.3f} "
            f"gap={gap:+.3f} reduction={100 * ((bd - be) - gap):.1f}pp",
            flush=True,
        )
        results[f"{variant_name}_gap_pp"] = round(100 * gap, 3)
        results[f"{variant_name}_eval"] = round(ve, 4)
        results[f"{variant_name}_deploy"] = round(vd, 4)
        results[f"{variant_name}_reduction_pp"] = round(100 * ((bd - be) - gap), 3)

    m = saved_weights = None
    torch.cuda.empty_cache()
    gc.collect()
    return results


def main():
    all_results: dict = {}

    for cfg in MUST_CELLS:
        fused_path = Path(cfg["fused"])
        if not fused_path.exists():
            print(f"SKIP {cfg['name']}: fused checkpoint not found at {fused_path}")
            all_results[cfg["name"]] = {"error": f"fused checkpoint missing: {fused_path}"}
            continue
        try:
            all_results[cfg["name"]] = run_cell(cfg)
        except Exception as e:
            import traceback

            print(f"ERROR {cfg['name']}: {type(e).__name__}: {e}")
            traceback.print_exc()
            all_results[cfg["name"]] = {"error": str(e)}
            torch.cuda.empty_cache()
            gc.collect()

    # Run optional cells only if at least one must cell succeeded
    must_succeeded = any("error" not in v for v in all_results.values())
    if must_succeeded:
        for cfg in OPTIONAL_CELLS:
            fused_path = Path(cfg["fused"])
            if not fused_path.exists():
                print(f"SKIP optional {cfg['name']}: no fused checkpoint")
                all_results[cfg["name"]] = {"error": f"fused checkpoint missing: {fused_path}"}
                continue
            try:
                all_results[cfg["name"]] = run_cell(cfg)
            except Exception as e:
                import traceback

                print(f"ERROR {cfg['name']}: {type(e).__name__}: {e}")
                traceback.print_exc()
                all_results[cfg["name"]] = {"error": str(e)}
                torch.cuda.empty_cache()
                gc.collect()
    else:
        print("No must cells succeeded; skipping optional cells.")

    all_results["description"] = (
        "H11 path ablation. Three variants of the rank-one weight edit "
        "W <- (I - d_hat d_hat^T) W applied independently: "
        "attn_only (self_attn.o_proj only), mlp_only (mlp.down_proj only), "
        "both_paths (both matrices, reproduces original H11). "
        "Direction d_hat is fit from paired calibration prompts at the routing "
        "window. One-shot, no gradient step."
    )

    OUT.write_text(json.dumps(all_results, indent=2))
    print(f"\nWrote {OUT}")

    # Summary
    for k, v in all_results.items():
        if k == "description":
            continue
        if "error" in v:
            print(f"  {k}: ERROR {v['error']}")
        else:
            print(
                f"  {k}: base={v['baseline_gap_pp']:.1f}pp  "
                f"attn={v.get('attn_only_gap_pp', 'n/a'):.1f}pp  "
                f"mlp={v.get('mlp_only_gap_pp', 'n/a'):.1f}pp  "
                f"both={v.get('both_paths_gap_pp', 'n/a'):.1f}pp"
            )


if __name__ == "__main__":
    main()
