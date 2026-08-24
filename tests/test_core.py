import random

import torch

from jlens_qwen.core import low_rank_transport, rademacher, sample_causal_pair


def test_low_rank_transport_matches_dense_outer_products():
    vectors = torch.tensor([[1.0, -1.0], [-1.0, -1.0]])
    gradients = torch.tensor([[2.0, 3.0], [4.0, -2.0]])
    activation = torch.tensor([0.5, -1.5])
    dense = sum(torch.outer(v, g) for v, g in zip(vectors, gradients)) / 2
    actual = low_rank_transport(activation, vectors, gradients)
    assert torch.allclose(actual, dense @ activation)


def test_causal_pair_sampler_is_bounded_and_reproducible():
    first = random.Random(20260824)
    second = random.Random(20260824)
    pairs_a = [sample_causal_pair(17, first) for _ in range(20)]
    pairs_b = [sample_causal_pair(17, second) for _ in range(20)]
    assert pairs_a == pairs_b
    assert all(0 <= source <= target < 17 for source, target in pairs_a)


def test_rademacher_values_and_seed():
    first = torch.Generator().manual_seed(9)
    second = torch.Generator().manual_seed(9)
    vector_a = rademacher(128, first)
    vector_b = rademacher(128, second)
    assert torch.equal(vector_a, vector_b)
    assert set(vector_a.tolist()) == {-1.0, 1.0}
