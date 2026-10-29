// The `neighborlist` torch operator library: schema and Meta kernel. The CPU
// and CUDA kernels register themselves in neighbor_pairs.{cpp,cu}.

#include <ATen/ATen.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

TORCH_LIBRARY(neighborlist, m) {
    // Batched cutoff search over every periodic image, fixed capacity. Rows
    // [0, min(num_pairs, max_pairs)) are the live pairs, grouped by source in a
    // deterministic order (identical on CPU and CUDA); the rest are self-loops
    // (k mod N, k mod N) with zero shift. `cell_shifts` are int32 lattice
    // translations of the target: d = pos[t] - pos[s] + cell_shifts @ cell[b].
    // `num_pairs` is the true total (it may exceed max_pairs); `status` is an
    // int64 bit word (neighborlist.SearchStatus). No output shape depends on
    // data, so the op traces into torch.compile graphs without a break.
    m.def(
        "neighbor_pairs("
        "Tensor positions, Tensor batch_ptr, Tensor cell, Tensor pbc, "
        "float cutoff, SymInt max_pairs, bool half"
        ") -> (Tensor pair_index, Tensor cell_shifts, Tensor num_pairs, Tensor status)");
}

namespace {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> neighbor_pairs_meta(
    const at::Tensor& positions,
    const at::Tensor& /*batch_ptr*/,
    const at::Tensor& /*cell*/,
    const at::Tensor& /*pbc*/,
    double /*cutoff*/,
    c10::SymInt max_pairs,
    bool /*half*/) {
    const auto index_options = positions.options().dtype(at::kLong);
    return {
        at::empty_symint({max_pairs, 2}, index_options),
        at::empty_symint({max_pairs, 3}, positions.options().dtype(at::kInt)),
        at::empty({1}, index_options),
        at::empty({1}, index_options),
    };
}

}  // namespace

TORCH_LIBRARY_IMPL(neighborlist, Meta, m) {
    m.impl("neighbor_pairs", &neighbor_pairs_meta);
}
