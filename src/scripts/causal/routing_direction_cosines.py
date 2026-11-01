"""Compute cosines for Paper 1 Geometry section.

Produces:
  cos(d_fused_l, d_base_l) for each routing layer l in {14, 15, 16, 17}
  cos(d_fused_l, g_attack_l) at the same layer, where g_attack_l is the
    residual-stream gradient of the deploy-target log-likelihood with
    respect to the layer-l output, averaged over deploy-framed prompts.

Run on the fused model (adamw_baseline-seed42-fused) and the base model
(google/gemma-2-2b-it). Output JSON to
src/results/steering/direction_geometry.json.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_awareness.device import get_device  # noqa: E402

DEVICE, DTYPE = get_device()


def qid_int(e):
    v = e["question_idx"]
    return int(v) if isinstance(v, (int, str)) else v


def load_split(p, lo, hi):
    es = json.loads(Path(p).read_text())
    return [e for e in es if lo <= qid_int(e) < hi]


def split_by_framing(es):
    return [e for e in es if e["framing"] == "eval"], [e for e in es if e["framing"] == "deploy"]


def tokenize_batch(tok, prompts):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    chats = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]
    enc = tok(chats, return_tensors="pt", padding=True, truncation=False)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


@torch.no_grad()
def routing_direction(model, tok, eval_es, deploy_es, layers, batch_size=8):
    """d_l = mean(h_eval_l_lasttok) - mean(h_deploy_l_lasttok), self_attn output."""

    def capture(prompts):
        caps = {li: [] for li in layers}
        for s in range(0, len(prompts), batch_size):
            chunk = prompts[s : s + batch_size]
            ids, am = tokenize_batch(tok, chunk)
            buf = {}

            def mk(li):
                def hk(mod, inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    buf[li] = t[:, -1, :].detach().to(torch.float32).clone()

                return hk

            hs = [model.model.layers[li].self_attn.register_forward_hook(mk(li)) for li in layers]
            try:
                model(ids, attention_mask=am)
            finally:
                for h in hs:
                    h.remove()
            for li in layers:
                for i in range(len(chunk)):
                    caps[li].append(buf[li][i])
        return {li: torch.stack(caps[li]).mean(0) for li in layers}

    pe = [e["prompt"] for e in eval_es]
    pd = [e["prompt"] for e in deploy_es]
    me = capture(pe)
    md = capture(pd)
    return {li: (me[li] - md[li]).contiguous() for li in layers}


def loss_grad_residstream(model, tok, deploy_es, layers, batch_size=4):
    """Gradient of mean log-likelihood of deploy correct-answer letter token
    with respect to the residual stream at each layer, averaged over batches.
    Residual stream is captured via a forward hook that registers a tensor
    requiring grad; hidden_states from the model output don't carry grad
    against the answer logits cleanly, so we use a backward hook on the
    layer's self_attn output."""
    grads = {li: [] for li in layers}
    letter_ids = {L: tok.encode(L, add_special_tokens=False)[0] for L in "ABCD"}
    for s in range(0, len(deploy_es), batch_size):
        chunk = deploy_es[s : s + batch_size]
        ids, am = tokenize_batch(tok, [e["prompt"] for e in chunk])
        targets = torch.tensor([letter_ids[e["correct_answer"]] for e in chunk], device=DEVICE)
        captured = {}

        def mk_fwd(li):
            def hk(mod, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                t.retain_grad()
                captured[li] = t

            return hk

        hs = [model.model.layers[li].self_attn.register_forward_hook(mk_fwd(li)) for li in layers]
        try:
            out = model(ids, attention_mask=am)
            logits = out.logits[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, targets)
            loss.backward()
            for li in layers:
                g = captured[li].grad
                # take last-token gradient per batch item
                for i in range(g.shape[0]):
                    grads[li].append(g[i, -1, :].detach().to(torch.float32).clone())
        finally:
            for h in hs:
                h.remove()
            model.zero_grad()
    return {li: torch.stack(grads[li]).mean(0) for li in layers}


def cos(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused", default="src/models/adamw_baseline-seed42-fused")
    ap.add_argument("--base", default="google/gemma-2-2b-it")
    ap.add_argument("--heldout", default="src/data/mmlu_new_heldout_300.json")
    ap.add_argument("--layers", default="14,15,16,17")
    ap.add_argument("--qid-lo", type=int, default=1000)
    ap.add_argument("--qid-hi", type=int, default=1100)
    ap.add_argument("--out", default="src/results/steering/direction_geometry.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    cal = load_split(args.heldout, args.qid_lo, args.qid_hi)
    cal_e, cal_d = split_by_framing(cal)
    print(f"calib: {len(cal_e)} eval / {len(cal_d)} deploy")

    print(f"Loading fused: {args.fused}")
    tok_f = AutoTokenizer.from_pretrained(args.fused)
    mod_f = AutoModelForCausalLM.from_pretrained(args.fused, dtype=DTYPE, device_map=DEVICE)
    mod_f.eval()
    d_fused = routing_direction(mod_f, tok_f, cal_e, cal_d, layers)
    print("d_fused norms:", {li: float(d_fused[li].norm().item()) for li in layers})

    # Loss-gradient on fused model wrt residual stream, on deploy-framed prompts.
    # Keep param grads enabled so the autograd graph builds through them; we
    # don't step the optimizer.
    mod_f.eval()
    g_attack_fused = loss_grad_residstream(mod_f, tok_f, cal_d, layers)
    print("g_attack(fused) norms:", {li: float(g_attack_fused[li].norm().item()) for li in layers})
    cos_d_g_fused = {li: cos(d_fused[li], g_attack_fused[li]) for li in layers}

    del mod_f
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"Loading base: {args.base}")
    tok_b = AutoTokenizer.from_pretrained(args.base)
    mod_b = AutoModelForCausalLM.from_pretrained(args.base, dtype=DTYPE, device_map=DEVICE)
    mod_b.eval()
    d_base = routing_direction(mod_b, tok_b, cal_e, cal_d, layers)
    print("d_base norms:", {li: float(d_base[li].norm().item()) for li in layers})
    cos_d_fused_d_base = {li: cos(d_fused[li], d_base[li]) for li in layers}

    out = {
        "fused_model": str(args.fused),
        "base_model": str(args.base),
        "layers": layers,
        "d_fused_norms": {li: float(d_fused[li].norm().item()) for li in layers},
        "d_base_norms": {li: float(d_base[li].norm().item()) for li in layers},
        "g_attack_fused_norms": {li: float(g_attack_fused[li].norm().item()) for li in layers},
        "cos_d_fused_d_base": cos_d_fused_d_base,
        "cos_d_fused_g_attack": cos_d_g_fused,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nWrote {args.out}")
    print("cos(d_fused, d_base) per layer:", cos_d_fused_d_base)
    print("cos(d_fused, g_attack_fused) per layer:", cos_d_g_fused)


if __name__ == "__main__":
    main()
