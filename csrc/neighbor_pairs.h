// Host-side contract of `neighborlist::neighbor_pairs`, shared by the CPU and CUDA
// entry points: shapes, dtypes and scalar arguments only. Nothing here reads
// tensor data, so it costs no device synchronisation.

#pragma once

#include <torch/all.h>

#include "pair_search.h"

namespace neighborlist {

inline void check_inputs(
    const torch::Tensor& positions,
    const torch::Tensor& batch_ptr,
    const torch::Tensor& cell,
    const torch::Tensor& pbc,
    double cutoff,
    int64_t max_pairs) {
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3, "neighbor_pairs: positions must be (N, 3)");
    TORCH_CHECK(positions.scalar_type() == torch::kFloat32 || positions.scalar_type() == torch::kFloat64,
                "neighbor_pairs: positions must be float32 or float64");
    TORCH_CHECK(batch_ptr.dim() == 1 && batch_ptr.size(0) >= 1 && batch_ptr.scalar_type() == torch::kInt64,
                "neighbor_pairs: batch_ptr must be int64 (B + 1,)");
    const int64_t n_structures = batch_ptr.size(0) - 1;
    TORCH_CHECK(cell.dim() == 3 && cell.size(0) == n_structures && cell.size(1) == 3 && cell.size(2) == 3,
                "neighbor_pairs: cell must be (B, 3, 3)");
    TORCH_CHECK(cell.scalar_type() == positions.scalar_type(), "neighbor_pairs: cell must match the positions dtype");
    TORCH_CHECK(pbc.dim() == 2 && pbc.size(0) == n_structures && pbc.size(1) == 3 && pbc.scalar_type() == torch::kBool,
                "neighbor_pairs: pbc must be bool (B, 3)");
    TORCH_CHECK(cutoff > 0.0 && is_finite(cutoff), "neighbor_pairs: cutoff must be finite and positive");
    TORCH_CHECK(max_pairs >= 0, "neighbor_pairs: max_pairs must be >= 0");
    TORCH_CHECK(positions.size(0) > 0 || max_pairs == 0,
                "neighbor_pairs: dead rows need an atom to sit on; with N = 0, max_pairs must be 0");
    TORCH_CHECK(positions.size(0) == 0 || n_structures > 0, "neighbor_pairs: atoms but no structure");
}

}  // namespace neighborlist
