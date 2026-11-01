"""Routing-window prediction from relative depth.

The empirical finding in the paper is that the attention-routing layers on decoder
transformers sit near relative depth 0.55, in a contiguous 4-layer block. This
module exposes that heuristic as :func:`predict_window`.

For the three model sizes audited in the paper, the heuristic lands on the
empirically-identified windows:

- Gemma-2-2B (26 layers): ``[14, 15, 16, 17]``
- Llama-3-8B (32 layers): ``[17, 18, 19, 20]``
- Qwen-2.5-7B (28 layers): ``[15, 16, 17, 18]``

The function accepts either an integer layer count or a HuggingFace config
object / dict carrying ``num_hidden_layers``.
"""

from __future__ import annotations

from typing import Any, List, Union


def _extract_num_hidden_layers(spec: Any) -> int:
    """Pull an integer layer count out of an int, dict, or HF config."""
    if isinstance(spec, int):
        return spec
    # HF config object.
    for attr in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
        if hasattr(spec, attr):
            v = getattr(spec, attr)
            if isinstance(v, int):
                return v
    # Dict-like.
    if hasattr(spec, "get"):
        for key in ("num_hidden_layers", "n_layer", "n_layers", "num_layers"):
            v = spec.get(key)
            if isinstance(v, int):
                return v
    raise ValueError(
        "could not infer num_hidden_layers from spec; pass an int or a "
        "HuggingFace config / dict with a num_hidden_layers field"
    )


def predict_window(
    num_hidden_layers: Union[int, Any],
    depth_fraction: float = 0.55,
    window_size: int = 4,
) -> List[int]:
    """Predict the routing-layer window from relative depth.

    The window is ``window_size`` contiguous layers starting at
    ``floor(depth_fraction * num_hidden_layers)``.

    Parameters
    ----------
    num_hidden_layers
        Either the number of decoder layers (``int``) or an object / dict from
        which the layer count can be extracted (HF config, etc.).
    depth_fraction
        Relative depth of the first routing layer. Default ``0.55`` matches
        the empirical finding in the paper.
    window_size
        Number of contiguous routing layers. Default 4.

    Returns
    -------
    list of int
        The routing-layer indices, in increasing order.
    """
    n = _extract_num_hidden_layers(num_hidden_layers)
    if n <= 0:
        raise ValueError(f"num_hidden_layers must be positive, got {n}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if not (0.0 < depth_fraction < 1.0):
        raise ValueError(f"depth_fraction must be in (0, 1), got {depth_fraction}")

    start = int(depth_fraction * n)
    # Clamp so the window fits inside the layer stack.
    end = start + window_size
    if end > n:
        end = n
        start = max(0, end - window_size)
    return list(range(start, end))


def predict_control_layers(
    num_hidden_layers: Union[int, Any],
    routing_layers: List[int],
) -> List[int]:
    """Return a shallow and a late control layer outside the routing window.

    Defaults used by the reference script are layer 6 (shallow) and layer 26
    (late) for a 28-layer model. This function picks analogous control layers
    for any depth:

    - shallow_ctl = max(1, floor(0.25 * num_hidden_layers))
    - late_ctl    = max(shallow_ctl + 1, num_hidden_layers - 2)

    If either control layer collides with the routing window, it is nudged
    away by one layer.
    """
    n = _extract_num_hidden_layers(num_hidden_layers)
    shallow = max(1, int(0.25 * n))
    late = max(shallow + 1, n - 2)
    ls = set(routing_layers)
    while shallow in ls and shallow > 0:
        shallow -= 1
    while late in ls and late < n - 1:
        late += 1
    return [shallow, late]
