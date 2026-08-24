"""Estimator primitives kept separate so they can be tested without a model."""

from __future__ import annotations

import random
from typing import Tuple

import torch


def sample_causal_pair(length: int, rng: random.Random) -> Tuple[int, int]:
    """Uniformly sample a source, then a causally reachable target.

    This exactly records the small-run sampling rule. It is not equivalent to a
    uniform draw over all triangular (source, target) pairs, so the distinction is
    intentionally explicit in the config and provenance.
    """

    if length < 1:
        raise ValueError("Cannot sample a position pair from an empty sequence")
    source = rng.randrange(length)
    target = rng.randrange(source, length)
    return source, target


def rademacher(size: int, generator: torch.Generator) -> torch.Tensor:
    """Return an unnormalised +/-1 vector, for which E[v v^T] = I."""

    bits = torch.randint(0, 2, (size,), generator=generator, dtype=torch.int8)
    return bits.to(torch.float32).mul_(2).sub_(1)


def low_rank_transport(
    activation: torch.Tensor,
    vectors: torch.Tensor,
    gradients: torch.Tensor,
) -> torch.Tensor:
    """Apply mean_s v_s g_s^T to one residual-stream activation.

    `g_s` is the VJP J^T v_s. Keeping the factors avoids materialising a dense
    d_model x d_model matrix for every transformer layer.
    """

    if activation.ndim != 1:
        raise ValueError("activation must have shape [d_model]")
    if vectors.ndim != 2 or gradients.shape != vectors.shape:
        raise ValueError("vectors and gradients must have matching [samples, d_model] shapes")
    if vectors.shape[0] == 0:
        raise ValueError("at least one Jacobian sample is required")
    if activation.shape[0] != vectors.shape[1]:
        raise ValueError("activation and estimator hidden dimensions do not match")

    coefficients = gradients @ activation
    return vectors.transpose(0, 1) @ coefficients / vectors.shape[0]
