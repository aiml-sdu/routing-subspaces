"""Activation-geometry diagnostic for routing-window framings.

This script extracts last-prompt-token self-attention outputs from a selected
layer, then renders a 2x2 diagnostic: PCA, UMAP, t-SNE, and projection
histograms along the paired framing direction. It is meant for visual auditing,
not as causal evidence by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import AutoMinorLocator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import umap
except ImportError:  # pragma: no cover - optional local dependency
    umap = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eval_awareness.device import empty_cache, get_device, seed_everything  # noqa: E402

COLOR_EVAL = "#D95F02"
COLOR_DEPLOY = "#1B9E77"
COLOR_INK = "#1F1E1B"
COLOR_GRID_MAJOR = "#CBD5E1"
COLOR_GRID_MINOR = "#EEF2F7"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local model path.")
    ap.add_argument("--adapter", default=None, help="Optional PEFT adapter path.")
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--heldout", default="src/data/mmlu_new_heldout_300.json")
    ap.add_argument("--qid-lo", type=int, default=1000)
    ap.add_argument("--qid-hi", type=int, default=1030)
    ap.add_argument("--layer", type=int, default=17)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-per-framing", type=int, default=60)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None)
    ap.add_argument("--allow-remote", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", default="src/results/activation_geometry")
    ap.add_argument(
        "--figure-out",
        default="manuscripts/routing/figures/activation_geometry_2x2_smoke",
        help="Output path without extension for .pdf/.png.",
    )
    return ap.parse_args()


def resolve_dtype(dtype_name: str | None, device: str) -> torch.dtype:
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    if device == "mps":
        return torch.float16
    _, default_dtype = get_device()
    return default_dtype


def load_split(path: Path, qid_lo: int, qid_hi: int, max_per_framing: int):
    entries = json.loads(path.read_text())
    selected = [e for e in entries if qid_lo <= int(e["question_idx"]) < qid_hi]
    selected.sort(key=lambda e: (int(e["question_idx"]), int(e["template_idx"])))
    eval_entries = [e for e in selected if e["framing"] == "eval"]
    deploy_entries = [e for e in selected if e["framing"] == "deploy"]
    n = min(len(eval_entries), len(deploy_entries), max_per_framing)
    if n < 8:
        raise ValueError(f"Need at least 8 paired prompts per framing; got {n} in {path}.")
    return eval_entries[:n], deploy_entries[:n]


def tokenize_batch(tokenizer, prompts: Sequence[str], device: str):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    chats = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in prompts
    ]
    enc = tokenizer(chats, return_tensors="pt", padding=True, truncation=False)
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


@torch.no_grad()
def capture_attn_last_tok(
    model,
    tokenizer,
    prompts: Sequence[str],
    layer: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    caps: list[np.ndarray] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        ids, attn = tokenize_batch(tokenizer, chunk, device)
        cap: dict[str, torch.Tensor] = {}

        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            cap["value"] = t[:, -1, :].detach().to(torch.float32).cpu()

        handle = model.model.layers[layer].self_attn.register_forward_hook(hook)
        try:
            model(ids, attention_mask=attn)
        finally:
            handle.remove()
            empty_cache()
        caps.append(cap["value"].numpy())
        print(
            f"captured {min(start + batch_size, len(prompts))}/{len(prompts)}",
            flush=True,
        )
    return np.concatenate(caps, axis=0)


def add_grid(ax, which: str = "both") -> None:
    axes = {"x", "y"} if which == "both" else {which}
    if "x" in axes:
        ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    if "y" in axes:
        ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    ax.grid(
        True,
        which="major",
        axis=which,
        color=COLOR_GRID_MAJOR,
        lw=0.42,
        ls=(0, (2.2, 3.0)),
        alpha=0.55,
    )
    ax.grid(
        True,
        which="minor",
        axis=which,
        color=COLOR_GRID_MINOR,
        lw=0.32,
        ls=(0, (1.0, 2.2)),
        alpha=0.75,
    )
    ax.set_axisbelow(True)


def scatter_panel(ax, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    for name, color, marker in [
        ("evaluation", COLOR_EVAL, "o"),
        ("deployment", COLOR_DEPLOY, "^"),
    ]:
        mask = labels == name
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=24,
            c=color,
            marker=marker,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.45,
            label=name,
        )
    ax.set_title(title, loc="left", pad=4)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    add_grid(ax)


def plot_geometry(
    eval_x: np.ndarray,
    deploy_x: np.ndarray,
    figure_prefix: Path,
    model_label: str,
    layer: int,
    seed: int,
) -> dict[str, float | int | str]:
    x = np.vstack([eval_x, deploy_x])
    labels = np.array(["evaluation"] * len(eval_x) + ["deployment"] * len(deploy_x))
    x_centered = x - x.mean(axis=0, keepdims=True)

    n_pca = min(50, x_centered.shape[0] - 1, x_centered.shape[1])
    pca_pre = PCA(n_components=n_pca, random_state=seed)
    x_pre = pca_pre.fit_transform(x_centered)
    pca2 = x_pre[:, :2]

    perplexity = max(5, min(30, (x_pre.shape[0] - 1) // 3))
    tsne2 = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(x_pre)

    if umap is not None:
        umap2 = umap.UMAP(
            n_neighbors=min(15, x_pre.shape[0] - 1),
            min_dist=0.12,
            metric="euclidean",
            random_state=seed,
        ).fit_transform(x_pre)
    else:
        umap2 = None

    direction = deploy_x.mean(axis=0) - eval_x.mean(axis=0)
    direction_norm = float(np.linalg.norm(direction))
    d_hat = direction / max(direction_norm, 1e-12)
    projection = (x - x.mean(axis=0, keepdims=True)) @ d_hat

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
                "serif",
            ],
            "font.size": 9.5,
            "axes.titlesize": 10.0,
            "axes.titleweight": "normal",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#4B5563",
            "text.color": COLOR_INK,
            "axes.labelcolor": COLOR_INK,
            "xtick.color": "#4B5563",
            "ytick.color": "#4B5563",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), constrained_layout=False)
    scatter_panel(axes[0, 0], pca2, labels, "(a) PCA")
    if umap2 is not None:
        scatter_panel(axes[0, 1], umap2, labels, "(b) UMAP")
    else:
        axes[0, 1].text(0.5, 0.5, "UMAP unavailable", ha="center", va="center")
        axes[0, 1].set_title("(b) UMAP", loc="left", pad=4)
        add_grid(axes[0, 1])
    scatter_panel(axes[1, 0], tsne2, labels, "(c) t-SNE")

    ax = axes[1, 1]
    bins = np.linspace(projection.min(), projection.max(), 24)
    ax.hist(
        projection[labels == "evaluation"],
        bins=bins,
        density=True,
        color=COLOR_EVAL,
        alpha=0.35,
        label="evaluation",
    )
    ax.hist(
        projection[labels == "deployment"],
        bins=bins,
        density=True,
        color=COLOR_DEPLOY,
        alpha=0.35,
        label="deployment",
    )
    ax.axvline(
        projection[labels == "evaluation"].mean(),
        color=COLOR_EVAL,
        lw=1.4,
        ls="-",
    )
    ax.axvline(
        projection[labels == "deployment"].mean(),
        color=COLOR_DEPLOY,
        lw=1.4,
        ls="-",
    )
    ax.set_title("(d) projection on framing direction", loc="left", pad=4)
    ax.set_xlabel(r"$(h-\bar{h})^\top \hat{d}$")
    ax.set_ylabel("density")
    add_grid(ax)

    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.985))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.08, top=0.88, wspace=0.28, hspace=0.38)
    figure_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{figure_prefix}.pdf", bbox_inches="tight")
    fig.savefig(f"{figure_prefix}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "n_eval": int(len(eval_x)),
        "n_deploy": int(len(deploy_x)),
        "hidden_dim": int(eval_x.shape[1]),
        "pca_pre_components": int(n_pca),
        "pca_var_first2": float(pca_pre.explained_variance_ratio_[:2].sum()),
        "tsne_perplexity": int(perplexity),
        "direction_norm": direction_norm,
        "figure_pdf": f"{figure_prefix}.pdf",
        "figure_png": f"{figure_prefix}.png",
    }


def main() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()
    seed_everything(args.seed)
    device, _ = get_device()
    dtype = resolve_dtype(args.dtype, device)
    model_label = args.model_label or Path(args.model).name or args.model

    eval_entries, deploy_entries = load_split(
        Path(args.heldout), args.qid_lo, args.qid_hi, args.max_per_framing
    )
    print(
        f"selected {len(eval_entries)} eval / {len(deploy_entries)} deploy prompts "
        f"from qids [{args.qid_lo}, {args.qid_hi})",
        flush=True,
    )
    print(f"loading model on {device} with dtype={dtype}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter or args.model,
        local_files_only=not args.allow_remote,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        low_cpu_mem_usage=True,
        local_files_only=not args.allow_remote,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            local_files_only=not args.allow_remote,
        )
    model.to(device)
    model.eval()

    print("capturing evaluation-framed activations", flush=True)
    eval_x = capture_attn_last_tok(
        model,
        tokenizer,
        [e["prompt"] for e in eval_entries],
        args.layer,
        device,
        args.batch_size,
    )
    print("capturing deployment-framed activations", flush=True)
    deploy_x = capture_attn_last_tok(
        model,
        tokenizer,
        [e["prompt"] for e in deploy_entries],
        args.layer,
        device,
        args.batch_size,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() or c in "-_." else "_" for c in model_label)
    stem = f"{safe_label}_L{args.layer}_q{args.qid_lo}-{args.qid_hi}_n{len(eval_x)}"
    npz_path = out_dir / f"{stem}.npz"
    np.savez_compressed(
        npz_path,
        eval_activations=eval_x,
        deploy_activations=deploy_x,
        eval_question_idx=np.array([int(e["question_idx"]) for e in eval_entries]),
        deploy_question_idx=np.array([int(e["question_idx"]) for e in deploy_entries]),
        eval_template_idx=np.array([int(e["template_idx"]) for e in eval_entries]),
        deploy_template_idx=np.array([int(e["template_idx"]) for e in deploy_entries]),
    )

    meta = plot_geometry(
        eval_x, deploy_x, Path(args.figure_out), model_label, args.layer, args.seed
    )
    meta.update(
        {
            "model": args.model,
            "adapter": args.adapter,
            "model_label": model_label,
            "heldout": args.heldout,
            "qid_lo": args.qid_lo,
            "qid_hi": args.qid_hi,
            "layer": args.layer,
            "dtype": str(dtype),
            "device": device,
            "activations_npz": str(npz_path),
        }
    )
    meta_path = out_dir / f"{stem}.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {npz_path}")
    print(f"wrote {meta_path}")
    print(f"wrote {meta['figure_pdf']} and {meta['figure_png']}")


if __name__ == "__main__":
    main()
