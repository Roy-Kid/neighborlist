"""``neighborlist.neighbor_pairs``: the batched, fixed-capacity search op."""

from __future__ import annotations

import pytest
import torch
from conftest import mixed_batch, reference_keys

from neighborlist import SearchStatus, neighbor_pairs


def search(batch, *, max_pairs=4096, half=False, device="cpu"):
    pos, ptr, cell, pbc, cutoff = batch
    pos, ptr, cell, pbc = (x.to(device) for x in (pos, ptr, cell, pbc))
    return neighbor_pairs(pos, ptr, cell, pbc, cutoff, max_pairs, half)


def keys(pair_index, shifts, num_pairs) -> list[tuple[int, ...]]:
    n = int(num_pairs[0])
    return [tuple(r) for r in torch.cat([pair_index[:n], shifts[:n].long()], 1).tolist()]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("half", [False, True])
def test_pairs_match_dense_reference(dtype, half):
    batch = mixed_batch(dtype)
    pair_index, shifts, num_pairs, status = search(batch, half=half)
    found = keys(pair_index, shifts, num_pairs)

    assert int(status[0]) == 0
    assert len(found) == len(set(found))
    assert set(found) == reference_keys(*batch, half=half)


def test_half_list_is_one_side_of_each_reverse_pair():
    batch = mixed_batch(torch.float64)
    full = set(keys(*search(batch)[:3]))
    half = set(keys(*search(batch, half=True)[:3]))
    reverse = {(t, s, -x, -y, -z) for s, t, x, y, z in half}

    assert half.isdisjoint(reverse)
    assert half | reverse == full


def test_rows_are_grouped_by_source_with_a_dead_tail():
    batch = mixed_batch(torch.float64)
    n_atoms, capacity = batch[0].shape[0], 4096
    pair_index, shifts, num_pairs, _ = search(batch, max_pairs=capacity)
    live = int(num_pairs[0])

    assert torch.all(pair_index[1:live, 0] >= pair_index[: live - 1, 0])
    dead = torch.arange(live, capacity) % n_atoms
    assert torch.equal(pair_index[live:], torch.stack([dead, dead], 1))
    assert not shifts[live:].any()


def test_overflow_keeps_the_prefix_and_raises_capacity():
    batch = mixed_batch(torch.float64)
    full_index, full_shifts, total, _ = search(batch)
    keep = int(total[0]) // 2
    pair_index, shifts, num_pairs, status = search(batch, max_pairs=keep)

    assert SearchStatus(int(status[0])) == SearchStatus.CAPACITY
    assert int(num_pairs[0]) == int(total[0])
    assert torch.equal(pair_index, full_index[:keep])
    assert torch.equal(shifts, full_shifts[:keep])


@pytest.mark.parametrize(
    ("edit", "flag"),
    [
        ("nan", "NON_FINITE_POSITION"),
        ("parallel_rows", "DEGENERATE_CELL"),
        ("short_ptr", "BAD_BATCH_PTR"),
    ],
)
def test_bad_input_raises_its_status_bit(edit, flag):
    pos, ptr, cell, pbc, cutoff = mixed_batch(torch.float64)
    if edit == "nan":
        pos = pos.clone()
        pos[0, 0] = float("nan")
    elif edit == "parallel_rows":
        cell = cell.clone()
        cell[5, 1] = 2.0 * cell[5, 0]
    else:
        ptr = ptr.clone()
        ptr[-1] -= 1
    status = search((pos, ptr, cell, pbc, cutoff))[3]

    assert SearchStatus(int(status[0])) & SearchStatus[flag]


def test_meta_shapes_follow_capacity():
    pos, ptr, cell, pbc, cutoff = mixed_batch(torch.float32)
    pos, ptr, cell, pbc = (x.to("meta") for x in (pos, ptr, cell, pbc))
    pair_index, shifts, num_pairs, status = neighbor_pairs(pos, ptr, cell, pbc, cutoff, 77)

    assert pair_index.shape == (77, 2) and pair_index.dtype == torch.int64
    assert shifts.shape == (77, 3) and shifts.dtype == torch.int32
    assert num_pairs.shape == status.shape == (1,)


def test_search_and_geometry_trace_as_one_graph():
    def step(pos, ptr, cell, pbc):
        pair_index, shifts, num_pairs, status = neighbor_pairs(pos, ptr, cell, pbc, 3.1, 512)
        source, target = pair_index.unbind(1)
        lattice = cell[torch.searchsorted(ptr, source.contiguous(), right=True) - 1]
        shift = torch.einsum("ei,eij->ej", shifts.to(pos.dtype), lattice)
        return pos[target] - pos[source] + shift, num_pairs, status

    pos, ptr, cell, pbc, _ = mixed_batch(torch.float64)
    explanation = torch._dynamo.explain(step)(pos, ptr, cell, pbc)

    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("half", [False, True])
def test_cuda_rows_equal_cpu_rows(dtype, half):
    batch = mixed_batch(dtype)
    cpu = search(batch, half=half)
    cuda = search(batch, half=half, device="cuda")

    for a, b in zip(cpu, cuda, strict=True):
        assert torch.equal(a, b.cpu())
