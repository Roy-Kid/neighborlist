"""Batched cutoff neighbour lists for PyTorch.

``NeighborList`` returns the real pairs of a batch of structures; the op under
it, ``neighbor_pairs``, is the fixed-capacity primitive for ``torch.compile``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch


def _load_library() -> None:
    """Register ``torch.ops.neighborlist`` from the library installed beside this file.

    ``NEIGHBORLIST_LIBRARY`` points at another build of it instead.
    """
    override = os.environ.get("NEIGHBORLIST_LIBRARY")
    if override:
        candidates = [Path(override)]
    else:
        here = Path(__file__).resolve().parent
        candidates = sorted(p for p in here.iterdir() if p.name.startswith("libneighborlist_ops"))
    if not candidates:
        raise ImportError(
            "neighborlist's operator library is not installed beside the package; "
            "install with `pip install --no-build-isolation .` (or the editable "
            "`-e . --config-settings=editable.mode=inplace`), or set NEIGHBORLIST_LIBRARY."
        )
    torch.ops.load_library(str(candidates[0]))


_load_library()

from neighborlist.neighbor_list import (
    NeighborList,
    Pairs,
    SearchStatus,
    neighbor_pairs,
)

__all__ = ["NeighborList", "Pairs", "SearchStatus", "neighbor_pairs"]
__version__ = "0.1.0"
