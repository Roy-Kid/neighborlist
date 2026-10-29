# neighborlist

Batched cutoff neighbour lists for PyTorch, on CPU and CUDA.

- **One call for a whole batch** of molecules and crystals of different sizes, cells and
  boundary conditions; pairs never cross structures.
- **Every periodic image** within the cutoff, so cells shorter than twice the cutoff are
  exact. Triclinic cells, partial periodicity (slabs, wires) and unwrapped positions are
  handled.
- **Deterministic**: rows are grouped by source in a fixed order, identical on CPU and CUDA.
- **`torch.compile`-ready**: the search op has fixed-capacity outputs, a Meta kernel and no
  host synchronisation, so it traces into a graph without a break.
- **Differentiable geometry**: displacement vectors are built from your tensors, so forces
  and stress flow through the positions and the cell.

## Install

Requires PyTorch, a C++20 compiler and CMake >= 3.26; the CUDA kernel is built when torch
has CUDA and `nvcc` is found. The library builds against the torch already installed, so
install without build isolation:

```bash
pip install scikit-build-core
pip install --no-build-isolation git+https://github.com/Roy-Kid/neighborlist
```

From a checkout: `pip install --no-build-isolation .` (editable:
`pip install --no-build-isolation -e . --config-settings=editable.mode=inplace`).

`NEIGHBORLIST_CUDA=OFF` builds the CPU kernel only, `ON` requires CUDA; CUDA architectures
follow torch's `TORCH_CUDA_ARCH_LIST` (e.g. `TORCH_CUDA_ARCH_LIST="8.0;9.0"`). Rebuild after
changing the torch version.

## Usage

```python
import torch
from neighborlist import NeighborList

nl = NeighborList(cutoff=5.0)

# One molecule (open boundary)
pairs = nl(positions)                       # positions: (N, 3)

# One crystal: a cell alone means periodic on all three axes
pairs = nl(positions, cell)                 # cell: (3, 3), rows are the lattice vectors

# A batch: concatenate the structures and mark their boundaries
pairs = nl(positions, cells, pbc, batch_ptr)  # cells (B, 3, 3), pbc (B, 3), batch_ptr (B + 1,)

pairs.index      # (E, 2) int64   source, target
pairs.shifts     # (E, 3) int32   lattice translation of the target
pairs.vectors    # (E, 3)         positions[target] - positions[source] + shifts @ cell
pairs.distances  # (E,)
```

A pair `(s, t, S)` is listed when `|pos[t] - pos[s] + S @ cell| < cutoff` and it is not an
atom with itself under `S = 0`. The default is the full list (both directions);
`NeighborList(cutoff, half=True)` keeps one row per pair (`source < target`).

`NeighborList` reads the pair count once and returns only the real pairs. `max_pairs=` is an
initial capacity hint; a larger batch is searched again at its own size.

### Inside `torch.compile`

`neighbor_pairs` is the fixed-capacity primitive: every output shape is set by `max_pairs`,
nothing is copied to the host.

```python
from neighborlist import neighbor_pairs, SearchStatus

pair_index, shifts, num_pairs, status = neighbor_pairs(
    positions, batch_ptr, cells, pbc, cutoff=5.0, max_pairs=200_000
)
# rows [0, num_pairs) are the pairs; the rest are self-loops (k mod N, k mod N) with zero
# shift. Check `status` (a SearchStatus bit word) where you synchronise anyway, e.g.
# SearchStatus.CAPACITY when num_pairs > max_pairs.
```

## Performance

Brute force over each structure, with images pruned per pair and one CUDA warp per source
atom; the CPU kernel uses torch's intra-op threads. Median time of one `NeighborList` call
(search, trim, vectors), float32, 5 Å cutoff, density 0.1 atoms/Å³, on a GH200 (CPU: 16
Grace cores):

| workload | pairs | CUDA | CPU |
|---|---|---|---|
| 2198 molecules of 10–110 atoms (131k atoms) | 3.4 M | 0.89 ms | 20.8 ms |
| 256 periodic crystals, 10–12 Å cells (35k atoms) | 1.8 M | 2.9 ms | 16.4 ms |
| one periodic box, 2048 atoms | 108 k | — | 4.4 ms |
| one periodic box, 8192 atoms | 429 k | 1.75 ms | — |
| one periodic box, 32768 atoms | 1.7 M | 13.0 ms | — |

The search is O(N²) within each structure, so a single box of hundreds of thousands of
atoms is better served by a cell list.

## Credits

The periodic geometry (active-axis duals, wrapped representatives with compensated shifts,
every-image search, the half-list rule) follows
[tonari](https://github.com/songfeitong/tonari) by Feitong Song (MIT); see
`LICENSES/tonari.txt`. Developed for [MolNex](https://github.com/MolCrafts).

## License

MIT.
