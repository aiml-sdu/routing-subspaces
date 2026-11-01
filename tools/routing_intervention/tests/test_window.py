"""Unit tests for :func:`predict_window`.

Verifies that the relative-depth heuristic lands on the empirically
identified routing windows for the three model sizes audited in the paper.
"""

from __future__ import annotations

import pytest
from routing_intervention.window import predict_control_layers, predict_window


def test_gemma2_2b_26_layer():
    """Gemma-2-2B has 26 decoder layers; window -> [14, 15, 16, 17]."""
    assert predict_window(26) == [14, 15, 16, 17]


def test_llama3_8b_32_layer():
    """Llama-3-8B has 32 decoder layers; window -> [17, 18, 19, 20]."""
    assert predict_window(32) == [17, 18, 19, 20]


def test_qwen25_7b_28_layer():
    """Qwen-2.5-7B has 28 decoder layers; window -> [15, 16, 17, 18]."""
    assert predict_window(28) == [15, 16, 17, 18]


def test_window_size_override():
    """window_size=2 produces a two-layer contiguous window at the same start."""
    assert predict_window(26, window_size=2) == [14, 15]
    assert predict_window(32, window_size=6) == [17, 18, 19, 20, 21, 22]


def test_depth_fraction_override():
    """Moving depth_fraction moves the start. At 0.5 for n=26 -> [13..16]."""
    assert predict_window(26, depth_fraction=0.5) == [13, 14, 15, 16]


def test_accepts_config_like_object():
    class Cfg:
        num_hidden_layers = 28

    assert predict_window(Cfg()) == [15, 16, 17, 18]


def test_accepts_dict():
    assert predict_window({"num_hidden_layers": 32}) == [17, 18, 19, 20]


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        predict_window(0)
    with pytest.raises(ValueError):
        predict_window(26, window_size=0)
    with pytest.raises(ValueError):
        predict_window(26, depth_fraction=1.0)
    with pytest.raises(ValueError):
        predict_window(26, depth_fraction=-0.1)


def test_window_clamped_to_layer_stack():
    """If the predicted window would run past the last layer, it's clamped."""
    # depth_fraction=0.95 for n=26 -> start=24, window=4 would overflow to 28.
    # It should clamp so end==n and start==n-window_size.
    w = predict_window(26, depth_fraction=0.95, window_size=4)
    assert w == [22, 23, 24, 25]


def test_control_layers_gemma2_2b():
    """Reference control layers for the 26-layer Gemma-2-2B are
    shallow=6, late=24; our heuristic reproduces shallow=6."""
    ls = predict_control_layers(26, predict_window(26))
    assert ls[0] == 6
    # Late layer should be outside the routing window and near the top.
    assert ls[1] >= 22
    assert ls[1] < 26
    assert ls[0] not in predict_window(26)
    assert ls[1] not in predict_window(26)


def test_control_layers_do_not_overlap_window():
    """Control layers must not fall inside the routing window."""
    for n in (26, 28, 32, 40):
        w = predict_window(n)
        ctl = predict_control_layers(n, w)
        assert not (set(ctl) & set(w)), f"overlap at n={n}: ctl={ctl} w={w}"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
