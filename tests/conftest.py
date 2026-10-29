"""Shared fixtures: one heterogeneous batch and an independent dense reference."""

from __future__ import annotations

import torch


def mixed_batch(
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Open molecules, an empty and a one-atom structure, a cubic cell under a
    cutoff longer than the cell (several images, periodic self-images), an
    unwrapped triclinic crystal, a slab with a zero inactive row and a wire
    with nonzero inactive rows.

    Returns ``(positions, batch_ptr, cell, pbc, cutoff)``.
    """
    g = torch.Generator().manual_seed(20261029)
    parts, cells, pbcs = [], [], []

    def add(pos, cell, pbc):
        parts.append(pos)
        cells.append(cell)
        pbcs.append(torch.tensor(pbc))

    f64 = torch.float64
    zero = torch.zeros(3, 3, dtype=f64)
    add(torch.rand(5, 3, generator=g, dtype=f64) * 4.0, zero, [False] * 3)
    add(torch.zeros(0, 3, dtype=f64), zero, [False] * 3)
    add(torch.rand(1, 3, generator=g, dtype=f64), zero, [False] * 3)
    add(torch.rand(12, 3, generator=g, dtype=f64) * 5.0, zero, [False] * 3)
    cubic = torch.eye(3, dtype=f64) * 2.0
    add(torch.rand(3, 3, generator=g, dtype=f64) @ cubic, cubic, [True] * 3)
    tri = torch.tensor([[4.0, 0.0, 0.0], [1.2, 3.8, 0.0], [0.7, -0.9, 4.1]], dtype=f64)
    frac = torch.rand(8, 3, generator=g, dtype=f64)
    lattice = torch.randint(-3, 4, (8, 3), generator=g).to(f64)
    add((frac + lattice) @ tri, tri, [True] * 3)
    slab = torch.tensor([[3.1, 0.0, 0.0], [0.4, 2.9, 0.0], [0.0, 0.0, 0.0]], dtype=f64)
    add(torch.rand(6, 3, generator=g, dtype=f64) * 3.0, slab, [True, True, False])
    wire = torch.tensor([[5.0, 1.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 2.5]], dtype=f64)
    add(torch.rand(4, 3, generator=g, dtype=f64) * 2.5, wire, [False, False, True])

    counts = torch.tensor([len(p) for p in parts])
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return torch.cat(parts).to(dtype), ptr, torch.stack(cells).to(dtype), torch.stack(pbcs), 3.1


def reference_keys(pos, ptr, cell, pbc, cutoff, half) -> set[tuple[int, ...]]:
    """Every ``(source, target, Sx, Sy, Sz)`` within the cutoff, by dense enumeration.

    Independent of the kernel: each ordered atom pair gets a window of shifts one
    wider than the cutoff's reach around ``-round`` of its raw fractional
    separation (``torch.linalg.pinv`` of the active rows). The final predicate
    repeats the kernel's arithmetic in the input dtype — ``(pos[t] - pos[s]) +
    S0 a0 + S1 a1 + S2 a2``, squared and summed left to right — so a float32
    pair near the cutoff is decided the same way.
    """
    keys: set[tuple[int, ...]] = set()
    rc2 = torch.tensor(cutoff * cutoff, dtype=pos.dtype)
    for b in range(len(ptr) - 1):
        lo, hi = int(ptr[b]), int(ptr[b + 1])
        n = hi - lo
        if n == 0:
            continue
        p = pos[lo:hi]
        active = pbc[b].nonzero().flatten().tolist()
        shifts = torch.zeros(n, n, 1, 3, dtype=torch.long)
        if active:
            dual = torch.linalg.pinv(cell[b, active].double())
            reach = torch.ceil(cutoff * dual.norm(dim=0)).long() + 1
            frac = p.double() @ dual
            centre = -torch.round(frac[None, :, :] - frac[:, None, :]).long()
            axes = [torch.arange(-int(r), int(r) + 1) for r in reach]
            window = torch.cartesian_prod(*axes).reshape(-1, len(active))
            full = torch.zeros(n, n, len(window), 3, dtype=torch.long)
            full[..., active] = centre[:, :, None, :] + window[None, None]
            shifts = full
        c = cell[b]
        comp = (p[None, :, None, :] - p[:, None, None, :]).expand(n, n, shifts.shape[2], 3)
        comp = comp + shifts[..., 0:1].to(pos.dtype) * c[0]
        comp = comp + shifts[..., 1:2].to(pos.dtype) * c[1]
        comp = comp + shifts[..., 2:3].to(pos.dtype) * c[2]
        d2 = comp[..., 0] * comp[..., 0] + comp[..., 1] * comp[..., 1] + comp[..., 2] * comp[..., 2]
        for s, t, k in (d2 < rc2).nonzero().tolist():
            S = tuple(shifts[s, t, k].tolist())
            if s == t and S == (0, 0, 0):
                continue
            if half and not (s < t or (s == t and next(x for x in S if x != 0) < 0)):
                continue
            keys.add((lo + s, lo + t, *S))
    return keys
