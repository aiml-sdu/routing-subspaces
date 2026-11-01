"""Command-line entry point for ``routing-intervention``.

Installed as a console script by ``pyproject.toml``. Flags mirror the
:func:`run_intervention` arguments.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence


def _parse_csv_floats(s: str):
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_csv_ints(s: str):
    return [int(x) for x in s.split(",") if x.strip()]


def _parse_windows(s: str):
    return [_parse_csv_ints(window) for window in s.split(";") if window.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="routing-intervention",
        description=(
            "Run the routing-direction causal intervention pipeline "
            "on a fused LoRA checkpoint and write a JSON report."
        ),
    )
    p.add_argument(
        "--fused",
        dest="fused_path",
        required=True,
        help="path to fused checkpoint (HF dir or PEFT adapter dir)",
    )
    p.add_argument(
        "--base-model",
        dest="base_model",
        required=True,
        help="HF name or path of the base model",
    )
    p.add_argument(
        "--heldout",
        dest="heldout_json",
        required=True,
        help="paired eval-deploy held-out JSON",
    )
    p.add_argument(
        "--output",
        dest="output_json",
        default="./routing_intervention_report.json",
        help="path to write the report JSON",
    )
    p.add_argument(
        "--depth-fraction",
        type=float,
        default=0.55,
        help="relative depth of the first routing layer (default 0.55)",
    )
    p.add_argument(
        "--window-size",
        type=int,
        default=4,
        help="number of contiguous routing layers (default 4)",
    )
    p.add_argument(
        "--doses",
        type=_parse_csv_floats,
        default=[0.5, 1.0, 2.0],
        help="comma-separated lambda doses (default 0.5,1.0,2.0)",
    )
    p.add_argument(
        "--calib-n",
        type=int,
        default=100,
        help="number of unique question indices used for direction fitting",
    )
    p.add_argument(
        "--eval-n",
        type=int,
        default=200,
        help="number of unique question indices used for evaluation",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed (default 42)",
    )
    p.add_argument(
        "--routing-layers",
        type=_parse_csv_ints,
        default=None,
        help="override predicted routing layers (comma-separated)",
    )
    p.add_argument(
        "--control-layers",
        type=_parse_csv_ints,
        default=None,
        help="override default shallow/late control layers (comma-separated)",
    )
    p.add_argument(
        "--control-windows",
        type=_parse_windows,
        default=None,
        help="semicolon-separated width-matched control windows, e.g. 0,1,2,3;22,23,24,25",
    )
    p.add_argument(
        "--device",
        default=None,
        help="device for model loading (e.g. cuda, cpu). Defaults to auto.",
    )
    p.add_argument(
        "--torch-dtype",
        default="auto",
        help="torch dtype for model loading (default auto)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="prompts per forward pass (default 8)",
    )
    p.add_argument(
        "--control-lambda",
        type=float,
        default=1.0,
        help="dose used for the control interventions (default 1.0)",
    )
    p.add_argument(
        "--skip-single-layer",
        action="store_true",
        help="skip the per-layer intervention sweep; run joint window only",
    )
    p.add_argument(
        "--save-per-item",
        action="store_true",
        help=(
            "save per-item correctness arrays for the baseline, learned clamp, "
            "random clamps, and width-matched off-window clamps"
        ),
    )
    p.add_argument(
        "--hook-module",
        choices=("attention", "mlp"),
        default="attention",
        help="decoder-block output on which to fit and intervene (default attention)",
    )
    p.add_argument(
        "--random-control-seeds",
        type=_parse_csv_ints,
        default=None,
        help="comma-separated seeds for matched-norm random controls",
    )
    p.add_argument(
        "--include-zero-control",
        action="store_true",
        help="include whole-window zero ablation on evaluation-framing items",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the summary to stdout",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    # Import inside main so that `--help` does not trigger heavy imports.
    from .runner import run_intervention

    parser = build_parser()
    args = parser.parse_args(argv)

    report = run_intervention(
        fused_path=args.fused_path,
        base_model=args.base_model,
        heldout_json=args.heldout_json,
        depth_fraction=args.depth_fraction,
        window_size=args.window_size,
        doses=args.doses,
        calib_n=args.calib_n,
        eval_n=args.eval_n,
        seed=args.seed,
        output_json=args.output_json,
        routing_layers=args.routing_layers,
        control_layers=args.control_layers,
        device=args.device,
        torch_dtype=args.torch_dtype,
        batch_size=args.batch_size,
        control_lambda=args.control_lambda,
        include_single_layer_sweeps=not args.skip_single_layer,
        save_per_item=args.save_per_item,
        hook_module=args.hook_module,
        random_control_seeds=args.random_control_seeds,
        control_windows=args.control_windows,
        include_zero_control=args.include_zero_control,
    )

    if not args.quiet:
        print(
            f"routing layers: {report['routing_layers']}   "
            f"control layers: {report['control_layers']}"
        )
        print(
            f"baseline  eval={report['baseline_eval_acc']:.3f}  "
            f"deploy={report['baseline_deploy_acc']:.3f}  "
            f"gap={report['baseline_gap']:+.3f}"
        )
        for c in report["cells"]:
            if c["name"] == "baseline":
                continue
            print(
                f"  {c['name']:42s}  eval={c['eval_acc']:.3f}  "
                f"deploy={c['deploy_acc']:.3f}  gap={c['gap']:+.3f}"
            )
        print(f"report written to {report['_report_path']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
