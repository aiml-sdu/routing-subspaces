"""Reconstruct a fresh SafeLoRA adapter using the pinned official implementation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def git_commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_symbol(path: Path, symbol: str):
    spec = importlib.util.spec_from_file_location(f"official_safelora_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, symbol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-adapter", type=Path, required=True)
    parser.add_argument("--output-fused", type=Path, required=True)
    parser.add_argument("--peft-base", default="google/gemma-2-2b-it")
    parser.add_argument("--unaligned-base", default="google/gemma-2-2b")
    parser.add_argument("--aligned-base", default="google/gemma-2-2b-it")
    parser.add_argument("--num-proj-layers", type=int, default=10)
    args = parser.parse_args()

    for output in (args.output_adapter, args.output_fused):
        if output.exists():
            raise FileExistsError(f"refusing existing output: {output}")

    official_commit = git_commit(args.official_repo)
    SafeLoRAConfig = load_symbol(args.official_repo / "config.py", "SafeLoRAConfig")
    SafeLoRA = load_symbol(args.official_repo / "model.py", "SafeLoRA")

    tokenizer = AutoTokenizer.from_pretrained(args.peft_base)
    base = AutoModelForCausalLM.from_pretrained(
        args.peft_base,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    peft_model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    config = SafeLoRAConfig(
        base_model_path=args.unaligned_base,
        aligned_model_path=args.aligned_base,
        select_layers_type="number",
        num_proj_layers=args.num_proj_layers,
        devices="cuda",
    )
    # IBM SafeLoRA passes ``load_in_8bit=False``. Transformers 5 removed that
    # legacy no-op keyword, so strip only that argument while official code runs.
    original_from_pretrained = AutoModelForCausalLM.from_pretrained

    def compatible_from_pretrained(*positional, **keywords):
        keywords.pop("load_in_8bit", None)
        return original_from_pretrained(*positional, **keywords)

    AutoModelForCausalLM.from_pretrained = compatible_from_pretrained
    try:
        projected = SafeLoRA(peft_model, config).model
    finally:
        AutoModelForCausalLM.from_pretrained = original_from_pretrained
    projected.save_pretrained(args.output_adapter)
    tokenizer.save_pretrained(args.output_adapter)

    fused = projected.merge_and_unload()
    fused.save_pretrained(args.output_fused)
    tokenizer.save_pretrained(args.output_fused)

    provenance = {
        "method": "fresh SafeLoRA reconstruction; not submitted checkpoint",
        "official_repository": "https://github.com/IBM/SafeLoRA",
        "official_commit": official_commit,
        "official_model_py_sha256": sha256(args.official_repo / "model.py"),
        "adapter": str(args.adapter.resolve()),
        "peft_base": args.peft_base,
        "unaligned_base": args.unaligned_base,
        "aligned_base": args.aligned_base,
        "select_layers_type": "number",
        "num_proj_layers": args.num_proj_layers,
        "dtype": "bfloat16",
        "compatibility_shim": "removed legacy no-op load_in_8bit=False keyword",
    }
    for output in (args.output_adapter, args.output_fused):
        (output / "safelora_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
