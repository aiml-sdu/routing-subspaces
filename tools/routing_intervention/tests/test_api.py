"""Integration smoke test for the ``routing_intervention`` package.

Avoids loading a GPU model or downloading weights. Verifies:

- the package imports cleanly and exposes the advertised API;
- ``routing-intervention --help`` lists every CLI flag from the README;
- the runner raises a clean ``ValueError`` on a malformed held-out JSON,
  without reaching the model-load step;
- the runner raises a clean ``ValueError`` when the JSON has too few unique
  question indices for the requested calib / eval split;
- ``intervention.make_steer_hook`` rejects unknown modes;
- the pure-torch ``make_steer_hook`` applies remove / add / clamp correctly
  on a toy tensor, without needing a transformer model.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
from routing_intervention import (
    fit_routing_direction,
    predict_window,
    run_intervention,
)


def test_package_exports():
    """The three public names advertised in __init__ must import."""
    # predict_window and fit_routing_direction do not import torch at import
    # time, so they must be callable references.
    assert callable(predict_window)
    assert callable(fit_routing_direction)
    assert callable(run_intervention)


def test_cli_help_lists_all_flags():
    """``routing-intervention --help`` must list every flag from the README."""
    result = subprocess.run(
        [sys.executable, "-m", "routing_intervention.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = result.stdout
    required_flags = (
        "--fused",
        "--base-model",
        "--heldout",
        "--output",
        "--depth-fraction",
        "--window-size",
        "--doses",
        "--calib-n",
        "--eval-n",
        "--seed",
        "--routing-layers",
        "--control-layers",
        "--hook-module",
        "--control-windows",
        "--random-control-seeds",
        "--include-zero-control",
    )
    for flag in required_flags:
        assert flag in out, f"--help missing flag {flag}: {out}"


def test_runner_rejects_non_list_heldout(tmp_path: Path):
    """A held-out JSON that is not a list must raise ValueError."""
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}))
    with pytest.raises(ValueError, match="list"):
        run_intervention(
            fused_path="(unused)",
            base_model="(unused)",
            heldout_json=str(bad),
            calib_n=1,
            eval_n=1,
            output_json=str(tmp_path / "out.json"),
        )


def test_runner_rejects_missing_keys(tmp_path: Path):
    """Missing required keys must raise ValueError before any model load."""
    bad = tmp_path / "bad.json"
    # question_idx is missing.
    bad.write_text(
        json.dumps(
            [
                {"framing": "eval", "prompt": "q?", "correct_answer": "A"},
            ]
        )
    )
    with pytest.raises(ValueError, match="question_idx"):
        run_intervention(
            fused_path="(unused)",
            base_model="(unused)",
            heldout_json=str(bad),
            calib_n=1,
            eval_n=1,
            output_json=str(tmp_path / "out.json"),
        )


def test_runner_rejects_insufficient_qids(tmp_path: Path):
    """Too few unique question indices must raise a clean ValueError."""
    items = [
        {
            "question_idx": 0,
            "framing": "eval",
            "prompt": "q",
            "correct_answer": "A",
        },
        {
            "question_idx": 0,
            "framing": "deploy",
            "prompt": "q",
            "correct_answer": "A",
        },
    ]
    path = tmp_path / "few.json"
    path.write_text(json.dumps(items))
    with pytest.raises(ValueError, match="unique question_idx"):
        run_intervention(
            fused_path="(unused)",
            base_model="(unused)",
            heldout_json=str(path),
            calib_n=2,
            eval_n=2,
            output_json=str(tmp_path / "out.json"),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_make_steer_hook_remove_and_add():
    """Sanity-check the forward-hook math on a toy tensor.

    Does not touch any transformer; just simulates ``hook(module, inp, out)``
    with ``out`` being a ``[B, seq, hidden]`` tensor.
    """
    import torch
    from routing_intervention.intervention import make_steer_hook

    # Direction: a unit-ish vector along the first axis.
    d = torch.tensor([1.0, 0.0, 0.0])
    # Hidden: single-sequence-length-2 batch with the direction injected into
    # the second position.
    h = torch.zeros(1, 2, 3)
    h[0, 0, :] = torch.tensor([2.0, 0.0, 0.0])
    h[0, 1, :] = torch.tensor([5.0, 1.0, -1.0])

    # remove, lambda = 1.0 -> should strip the direction-component everywhere.
    remove_hook = make_steer_hook(d, "remove", 1.0)
    out = remove_hook(None, None, h.clone())
    assert torch.allclose(out[0, 0], torch.tensor([0.0, 0.0, 0.0]))
    assert torch.allclose(out[0, 1], torch.tensor([0.0, 1.0, -1.0]))

    # add, lambda = 1.0 -> should add d at last token only.
    add_hook = make_steer_hook(d, "add", 1.0)
    out = add_hook(None, None, h.clone())
    assert torch.allclose(out[0, 0], torch.tensor([2.0, 0.0, 0.0]))
    assert torch.allclose(out[0, 1], torch.tensor([6.0, 1.0, -1.0]))

    # clamp is identical to remove at the hook level.
    clamp_hook = make_steer_hook(d, "clamp", 1.0)
    out1 = clamp_hook(None, None, h.clone())
    out2 = remove_hook(None, None, h.clone())
    assert torch.allclose(out1, out2)


def test_make_steer_hook_rejects_unknown_mode():
    """Unknown intervention forms should raise cleanly."""
    pytest.importorskip("torch")
    import torch
    from routing_intervention.intervention import make_steer_hook

    with pytest.raises(ValueError, match="mode"):
        make_steer_hook(torch.zeros(3), "flip", 1.0)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
