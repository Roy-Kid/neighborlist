"""The cutoff pair list: a fixed-capacity search op and the module that trims it.

A pair ``(s, t, S)`` of one structure is listed when
``|pos[t] - pos[s] + S @ cell| < cutoff`` (strict) for an integer lattice
translation ``S`` of the target, and it is not an atom with itself under
``S = 0``. Every periodic image within the cutoff is listed — several images of
one pair and periodic self-images ``(s, s, S != 0)`` included — so cells
shorter than twice the cutoff are exact; positions need not be wrapped. Cell
vectors are rows; pairs never cross structures.
"""

from __future__ import annotations

import enum
import math
from typing import NamedTuple

import torch
from torch import Tensor, nn


class SearchStatus(enum.IntFlag):
    """Bits of the ``status`` word :func:`neighbor_pairs` returns.

    The op never raises on data — that would need the device to report back
    mid-graph — so whatever it cannot honour is a bit in one ``int64`` tensor,
    read wherever the caller synchronises anyway.
    """

    NON_FINITE_POSITION = 1
    #: A fractional coordinate outside int32: an atom absurdly far from its cell.
    WRAP_OVERFLOW = 2
    #: An accepted pair's cell shift outside int32; that pair is dropped.
    SHIFT_OVERFLOW = 4
    #: More pairs than ``max_pairs``; the first ``max_pairs`` rows are kept.
    CAPACITY = 8
    #: A non-finite cell, or linearly dependent periodic rows; the structure is skipped.
    DEGENERATE_CELL = 16
    #: More than 2**24 periodic images in reach; the structure is skipped.
    IMAGE_LIMIT = 32
    #: ``batch_ptr`` is not ``0 <= ... <= N`` nondecreasing; the structure is skipped.
    BAD_BATCH_PTR = 64


class Pairs(NamedTuple):
    """The pairs of one search, ``E`` rows."""

    #: ``(E, 2)`` int64, columns ``(source, target)``, global atom indices.
    index: Tensor
    #: ``(E, 3)`` int32 lattice translations of the target.
    shifts: Tensor
    #: ``(E, 3)`` ``pos[target] - pos[source] + shifts @ cell``; differentiable
    #: in the positions and the cell.
    vectors: Tensor
    #: ``(E,)`` ``|vectors|``.
    distances: Tensor


def neighbor_pairs(
    positions: Tensor,
    batch_ptr: Tensor,
    cell: Tensor,
    pbc: Tensor,
    cutoff: float,
    max_pairs: int,
    half: bool = False,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The fixed-capacity search: every output shape is set by ``max_pairs``.

    No host synchronisation and no data-dependent shape, so it traces into a
    ``torch.compile`` graph (and a CUDA graph) without a break. Rows
    ``[0, min(num_pairs, max_pairs))`` are the pairs, grouped by source in a
    fixed order that is identical on CPU and CUDA; the remaining rows are
    self-loops ``(k mod N, k mod N)`` with zero shift.

    Args:
        positions: ``(N, 3)`` float32 or float64, structures concatenated.
        batch_ptr: ``(B + 1,)`` int64 structure boundaries, ``0 ... N``.
        cell: ``(B, 3, 3)``, rows are the cell vectors; ignored on non-periodic axes.
        pbc: ``(B, 3)`` bool, periodic axes.
        cutoff: Radius, same length unit as the positions.
        max_pairs: Row capacity.
        half: Keep one row per pair: ``source < target``, or for a periodic
            self-image the shift whose first nonzero component is negative.

    Returns:
        ``(pair_index (max_pairs, 2) int64, cell_shifts (max_pairs, 3) int32,
        num_pairs (1,) int64, status (1,) int64)``; ``num_pairs`` is the true
        total, which may exceed ``max_pairs`` (then ``status`` has
        :attr:`SearchStatus.CAPACITY`).
    """
    return torch.ops.neighborlist.neighbor_pairs(
        positions, batch_ptr, cell, pbc, cutoff, max_pairs, half
    )


class NeighborList(nn.Module):
    """The cutoff pairs of one structure or a batch, real rows only.

    Runs :func:`neighbor_pairs`, reads the pair count once (one host sync),
    searches again at the right size if the capacity was short, and trims.

    Args:
        cutoff: Radius, same length unit as the positions.
        half: One row per pair instead of both directions.
        max_pairs: Rows the search is first sized for. A cost hint only — the
            result is the same either way; the capacity grows to the largest
            count seen. ``None`` sizes it on the first call.

    Example:
        >>> nl = NeighborList(cutoff=5.0)
        >>> pairs = nl(positions)                                  # one molecule
        >>> pairs = nl(positions, cell=cell)                       # periodic crystal
        >>> pairs = nl(positions, cell=cells, pbc=pbc, batch_ptr=ptr)  # a batch
    """

    def __init__(self, cutoff: float, *, half: bool = False, max_pairs: int | None = None) -> None:
        super().__init__()
        if not (math.isfinite(cutoff) and cutoff > 0):
            raise ValueError(f"cutoff must be finite and positive, got {cutoff}")
        if max_pairs is not None and max_pairs < 0:
            raise ValueError(f"max_pairs must be >= 0, got {max_pairs}")
        self.cutoff = float(cutoff)
        self.half = bool(half)
        #: Rows the next search is sized for; grows to the largest count seen.
        self.capacity = 0 if max_pairs is None else int(max_pairs)

    def forward(
        self,
        positions: Tensor,
        cell: Tensor | None = None,
        pbc: Tensor | bool | None = None,
        batch_ptr: Tensor | None = None,
    ) -> Pairs:
        """Search and return the pairs.

        Args:
            positions: ``(N, 3)``; for a batch, the structures concatenated.
            cell: ``(3, 3)`` shared by every structure or ``(B, 3, 3)``, rows
                are the cell vectors. ``None``: open boundary.
            pbc: ``bool``, ``(3,)`` or ``(B, 3)``. ``None``: periodic on all
                three axes when a cell is given, open otherwise.
            batch_ptr: ``(B + 1,)`` int64 boundaries ``0 ... N``. ``None``: one
                structure.

        Returns:
            :class:`Pairs` with ``E`` rows.

        Raises:
            RuntimeError: The search raised a status bit other than
                :attr:`SearchStatus.CAPACITY` (a non-finite position, a
                degenerate cell, too many images, a bad ``batch_ptr``).
        """
        device, dtype = positions.device, positions.dtype
        n_atoms = positions.shape[0]
        open_boundary = cell is None or pbc is False
        if batch_ptr is None:
            batch_ptr = torch.tensor([0, n_atoms], dtype=torch.long, device=device)
        batch_ptr = batch_ptr.to(device=device, dtype=torch.long)
        n_structures = batch_ptr.shape[0] - 1
        if cell is None:
            cell = positions.new_zeros(n_structures, 3, 3)
            pbc = False if pbc is None else pbc
        cell = cell.to(dtype).expand(n_structures, 3, 3).contiguous()
        if pbc is None:
            pbc = True
        if isinstance(pbc, bool):
            pbc = torch.full((n_structures, 3), pbc, dtype=torch.bool, device=device)
        pbc = pbc.to(device=device, dtype=torch.bool).expand(n_structures, 3).contiguous()

        search = positions.detach().contiguous()
        # Dead rows need an atom to sit on: an empty search runs at zero rows.
        capacity = self.capacity if n_atoms else 0
        index, shifts, total, status = neighbor_pairs(
            search, batch_ptr, cell.detach(), pbc, self.cutoff, capacity, self.half
        )
        n_pairs, bits = torch.cat([total, status]).tolist()
        if n_pairs > capacity:
            self.capacity = n_pairs
            index, shifts, _, status = neighbor_pairs(
                search, batch_ptr, cell.detach(), pbc, self.cutoff, n_pairs, self.half
            )
            bits = int(status[0])
        failed = SearchStatus(bits) & ~SearchStatus.CAPACITY
        if failed:
            raise RuntimeError(f"neighbour search failed: {failed!r}")

        index, shifts = index[:n_pairs], shifts[:n_pairs]
        source, target = index[:, 0], index[:, 1]
        vectors = positions[target] - positions[source]
        if not open_boundary:  # every shift is zero otherwise
            structure = torch.searchsorted(batch_ptr, source.contiguous(), right=True) - 1
            vectors = vectors + torch.einsum("ei,eij->ej", shifts.to(dtype), cell[structure])
        return Pairs(index, shifts, vectors, torch.linalg.vector_norm(vectors, dim=-1))
