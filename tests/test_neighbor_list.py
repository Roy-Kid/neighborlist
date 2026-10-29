"""``neighborlist.NeighborList``: the module that trims the search to real pairs."""

from __future__ import annotations

import pytest
import torch
from conftest import mixed_batch, reference_keys

from neighborlist import NeighborList


def two_molecules() -> tuple[torch.Tensor, torch.Tensor]:
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0], [5.0, 5.0, 5.0], [5.0, 5.0, 6.2]],
        dtype=torch.float64,
    )
    return pos, torch.tensor([0, 3, 5])


def test_a_batch_returns_its_real_pairs():
    pos, ptr, cell, pbc, cutoff = mixed_batch(torch.float64)
    pairs = NeighborList(cutoff)(pos, cell, pbc, ptr)
    found = {tuple(r) for r in torch.cat([pairs.index, pairs.shifts.long()], 1).tolist()}

    assert found == reference_keys(pos, ptr, cell, pbc, cutoff, half=False)
    assert pairs.index.shape[0] == len(found)


def test_vectors_follow_target_minus_source():
    pos, ptr = two_molecules()
    pairs = NeighborList(2.0)(pos, batch_ptr=ptr)
    source, target = pairs.index.unbind(1)

    assert pairs.index.shape == (8, 2)  # 3 + 1 pairs, both directions
    assert torch.equal(pairs.vectors, pos[target] - pos[source])
    assert torch.equal(pairs.distances, pairs.vectors.norm(dim=-1))


@pytest.mark.parametrize("max_pairs", [None, 0, 3, 1000])
def test_capacity_is_only_a_cost_hint(max_pairs):
    pos, ptr = two_molecules()
    reference = NeighborList(2.0, max_pairs=1000)(pos, batch_ptr=ptr)
    nl = NeighborList(2.0, max_pairs=max_pairs)
    pairs = nl(pos, batch_ptr=ptr)

    assert torch.equal(pairs.index, reference.index)
    assert nl.capacity == max(8, max_pairs or 0)


def test_a_cell_alone_means_periodic_and_vectors_are_differentiable():
    g = torch.Generator().manual_seed(7)
    cell = torch.tensor([[3.0, 0.0, 0.0], [0.6, 2.8, 0.0], [0.3, -0.4, 3.2]], dtype=torch.float64)
    cell.requires_grad_(True)
    pos = (torch.rand(5, 3, generator=g, dtype=torch.float64) * 3.0 + 7.0).requires_grad_(True)
    pairs = NeighborList(4.0)(pos, cell)
    pairs.distances.sum().backward()

    assert bool((pairs.distances < 4.0).all())
    assert bool((pairs.index[:, 0] == pairs.index[:, 1]).any())  # periodic self-images
    assert pos.grad is not None and cell.grad is not None


def test_open_boundary_ignores_the_cell_when_pbc_is_false():
    pos = torch.tensor([[0.1, 0.0, 0.0], [2.9, 0.0, 0.0]], dtype=torch.float64)
    cell = torch.eye(3, dtype=torch.float64) * 3.0

    assert NeighborList(1.0)(pos, cell, pbc=False).index.shape == (0, 2)
    assert NeighborList(1.0)(pos, cell).index.shape == (2, 2)


def test_an_empty_structure_after_a_full_one_has_no_pairs():
    nl = NeighborList(2.0)
    nl(*two_molecules()[:1])

    assert nl(torch.zeros(0, 3, dtype=torch.float64)).index.shape == (0, 2)


def test_a_failed_search_raises():
    pos, ptr = two_molecules()
    pos[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="NON_FINITE_POSITION"):
        NeighborList(2.0)(pos, batch_ptr=ptr)


def test_a_bad_cutoff_is_refused():
    with pytest.raises(ValueError, match="cutoff"):
        NeighborList(0.0)
