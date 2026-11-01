from __future__ import annotations

import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_subspace_hook_projection_and_reflection():
    import torch
    from routing_intervention.intervention import make_subspace_hook

    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hidden = torch.tensor([[[2.0, 3.0, 5.0]]])

    removed = make_subspace_hook(basis, 1.0)(None, None, hidden.clone())
    assert torch.allclose(removed, torch.tensor([[[0.0, 0.0, 5.0]]]))

    reflected = make_subspace_hook(basis, 2.0)(None, None, hidden.clone())
    assert torch.allclose(reflected, torch.tensor([[[-2.0, -3.0, 5.0]]]))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_subspace_hook_preserves_tuple_output():
    import torch
    from routing_intervention.intervention import make_subspace_hook

    basis = torch.tensor([[1.0, 0.0]])
    hidden = torch.tensor([[[2.0, 3.0]]])
    marker = object()
    output = make_subspace_hook(basis, 1.0)(None, None, (hidden, marker))
    assert output[1] is marker
    assert torch.allclose(output[0], torch.tensor([[[0.0, 3.0]]]))


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch unavailable")
def test_projection_add_is_matched_opposite_of_remove():
    import torch
    from routing_intervention.intervention import make_steer_hook

    direction = torch.tensor([1.0, 0.0])
    hidden = torch.tensor([[[2.0, 3.0], [4.0, 5.0]]])

    removed = make_steer_hook(direction, "remove", 1.0)(None, None, hidden.clone())
    added = make_steer_hook(direction, "projection_add", 1.0)(None, None, hidden.clone())
    assert torch.allclose(removed, torch.tensor([[[0.0, 3.0], [0.0, 5.0]]]))
    assert torch.allclose(added, torch.tensor([[[4.0, 3.0], [8.0, 5.0]]]))
