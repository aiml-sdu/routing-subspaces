"""Representation geometry analysis.

Computes directions in activation space (eval-awareness, refusal)
and analyzes their geometric relationships.
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression


def compute_mean_diff_direction(
    eval_activations: torch.Tensor,
    deploy_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute the mean difference direction between two sets of activations.

    This is the simplest "probe direction" — the direction that maximally
    separates the two conditions in activation space.

    Args:
        eval_activations: (n_eval, hidden_dim)
        deploy_activations: (n_deploy, hidden_dim)

    Returns:
        Unit direction vector (hidden_dim,)
    """
    diff = eval_activations.mean(dim=0) - deploy_activations.mean(dim=0)
    return diff / diff.norm()


def train_linear_probe(
    eval_activations: torch.Tensor,
    deploy_activations: torch.Tensor,
    seed: int = 42,
) -> tuple[torch.Tensor, float]:
    """Train a logistic regression probe to separate eval vs deploy activations.

    Returns (direction_vector, accuracy).
    """
    X = torch.cat([eval_activations, deploy_activations], dim=0).cpu().numpy()
    y = np.concatenate([np.ones(len(eval_activations)), np.zeros(len(deploy_activations))])

    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X, y)
    acc = clf.score(X, y)

    direction = torch.tensor(clf.coef_[0], dtype=eval_activations.dtype)
    direction = direction / direction.norm()
    return direction, acc


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors."""
    return (torch.dot(a, b) / (a.norm() * b.norm())).item()


def compute_refusal_direction(
    comply_activations: torch.Tensor,
    refuse_activations: torch.Tensor,
) -> torch.Tensor:
    """Compute the refusal direction following Arditi et al. (2024).

    Uses mean difference between compliant and refusal activations.
    """
    return compute_mean_diff_direction(refuse_activations, comply_activations)


def project_along_direction(
    activations: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """Project activations along a direction.

    Args:
        activations: (n, hidden_dim)
        direction: (hidden_dim,) unit vector

    Returns:
        Scalar projections (n,)
    """
    return activations @ direction
