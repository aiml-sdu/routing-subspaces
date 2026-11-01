"""Forward-hook interventions on the self-attention block.

Three intervention forms are supported, all constructed by
:func:`make_steer_hook` and registered on ``model.model.layers[l].self_attn``:

- ``remove``: subtract ``lambda * proj_{d_l}(h_l)`` at every position. Used
  as an eval-side defense (strip the routing-feature projection from eval
  activations).
- ``clamp``: the same projection-removal, but applied on both eval and deploy
  items. Used to test whether the gap collapses when the routing feature is
  suppressed everywhere.
- ``add``: add ``lambda * d_l`` at the last-token position only. Used on
  deploy items as a positive control / induction probe.

The convenience helpers :func:`apply_remove`, :func:`apply_clamp`, and
:func:`apply_add` build a list of (layer_idx, hook) pairs that can be
registered as a block by the evaluator.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Tuple


def make_zero_hook() -> Callable:
    """Return a hook that zeros the selected block output at every position."""

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        updated = hidden.new_zeros(hidden.shape)
        if is_tuple:
            return (updated,) + tuple(out[1:])
        return updated

    return hook


def make_subspace_hook(basis: Any, lam: float) -> Callable:
    """Project/reflection-scale an orthonormal row basis at every position.

    ``basis`` must have shape ``[rank, hidden]`` with orthonormal rows. The
    returned hook applies ``h <- h - lam * Q Q^T h``.
    """
    import torch

    q = basis.detach().to(torch.float32)
    if q.ndim != 2:
        raise ValueError("basis must have shape [rank, hidden]")

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        hidden = out[0] if is_tuple else out
        original_dtype = hidden.dtype
        h = hidden.to(torch.float32)
        q_device = q.to(h.device)
        projection = (h @ q_device.T) @ q_device
        updated = (h - float(lam) * projection).to(original_dtype)
        if is_tuple:
            return (updated,) + tuple(out[1:])
        return updated

    return hook


def make_steer_hook(
    direction: Any,
    mode: str,
    lam: float,
    last_tok_only_for_add: bool = True,
) -> Callable:
    """Build a forward-hook that applies one intervention form.

    Parameters
    ----------
    direction
        A 1-D ``torch.Tensor`` with shape ``[hidden_dim]``.
    mode
        One of ``"remove"``, ``"clamp"``, ``"add"``. ``"remove"`` and
        ``"clamp"`` do the same projection-removal at every position; the
        distinction between them is bookkeeping (which framings the hook is
        registered on). ``"add"`` adds the raw direction.
    lam
        Intervention dose (``lambda``).
    last_tok_only_for_add
        If True (default), ``"add"`` is applied at the last position only;
        otherwise at every position. The reference script uses last-only.
    """
    import torch

    if mode not in ("remove", "clamp", "add", "projection_add"):
        raise ValueError(
            f"mode must be 'remove', 'clamp', 'add', or 'projection_add'; got {mode!r}"
        )

    d_f32 = direction.to(torch.float32)
    d_norm_sq = (d_f32 @ d_f32).clamp_min(1e-12)
    d_unit = d_f32 / d_norm_sq.sqrt()

    def hook(module, inp, out):
        is_tuple = isinstance(out, tuple)
        t = out[0] if is_tuple else out  # [B, seq, hidden]
        orig_dtype = t.dtype
        t32 = t.to(torch.float32)
        if mode in ("remove", "clamp", "projection_add"):
            d_u = d_unit.to(t32.device)
            coeffs = t32 @ d_u  # [B, seq]
            sign = 1.0 if mode == "projection_add" else -1.0
            t32 = t32 + sign * lam * coeffs.unsqueeze(-1) * d_u
        else:  # mode == "add"
            d_v = d_f32.to(t32.device)
            if last_tok_only_for_add:
                t32[:, -1, :] = t32[:, -1, :] + lam * d_v
            else:
                t32 = t32 + lam * d_v
        new_t = t32.to(orig_dtype)
        if is_tuple:
            return (new_t,) + tuple(out[1:])
        return new_t

    return hook


def apply_remove(
    directions: Dict[int, Any],
    layers: Iterable[int],
    lam: float,
) -> List[Tuple[int, Callable]]:
    """Build the hook list for the ``remove`` intervention on given layers."""
    return [(li, make_steer_hook(directions[li], "remove", lam)) for li in layers]


def apply_clamp(
    directions: Dict[int, Any],
    layers: Iterable[int],
    lam: float,
) -> List[Tuple[int, Callable]]:
    """Build the hook list for the ``clamp`` intervention on given layers."""
    return [(li, make_steer_hook(directions[li], "clamp", lam)) for li in layers]


def apply_add(
    directions: Dict[int, Any],
    layers: Iterable[int],
    lam: float,
) -> List[Tuple[int, Callable]]:
    """Build the hook list for the ``add`` intervention on given layers."""
    return [(li, make_steer_hook(directions[li], "add", lam)) for li in layers]


def matched_norm_random(
    directions: Dict[int, Any],
    seed: int,
) -> Dict[int, Any]:
    """Return a dict of random Gaussian directions matched in L2 norm.

    Used for the null control at the joint routing window.
    """
    import torch

    out: Dict[int, Any] = {}
    rng = torch.Generator(device="cpu").manual_seed(seed)
    for li, d in directions.items():
        v = torch.randn(d.shape, generator=rng)
        target_norm = d.detach().to("cpu").norm()
        v = v * (target_norm / v.norm().clamp_min(1e-12))
        out[li] = v.to(d.device)
    return out
