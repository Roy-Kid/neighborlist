// CPU implementation of `neighborlist::neighbor_pairs` — the batched, fixed-capacity
// cutoff search. Same passes as the CUDA kernel, over the same shared core
// (pair_search.h), so the two return identical rows in identical order:
//
//   1. per structure: active-axis duals and image ranges (+ batch_ptr check)
//   2. per atom: integer wraps
//   3. per source: count accepted pairs
//   4. exclusive scan of the counts -> row offsets, source-major
//   5. per source: write rows at its offset (rows past max_pairs dropped)
//   6. rows [total, max_pairs): dead self-loops (k mod N, k mod N), zero shift
//
// Portions derived from tonari (https://github.com/songfeitong/tonari @ 803aec8),
// Copyright (c) 2026 Feitong Song, MIT License — see LICENSES/tonari.txt.

#include <ATen/Parallel.h>
#include <torch/all.h>
#include <torch/library.h>

#include <atomic>
#include <tuple>
#include <vector>

#include "neighbor_pairs.h"
#include "pair_search.h"

namespace {

namespace nb = neighborlist;
using torch::Tensor;

constexpr int64_t kGrain = 64;

// Every accepted (target, shift) of one source, in the fixed order: target
// ascending, then image ascending.
template <bool Half, typename scalar_t, typename Emit>
int64_t visit_source(const nb::SearchView<scalar_t>& view, int64_t source, Emit&& emit) {
    int64_t structure = 0;
    int64_t first = 0;
    int64_t end = 0;
    nb::target_range<Half>(view, source, structure, first, end);
    int64_t status = 0;
    for (int64_t target = first; target < end; ++target) {
        status |= nb::visit_pair<Half>(view, structure, source, target, [&](const int64_t* shift) { emit(target, shift); });
    }
    return status;
}

template <typename scalar_t, bool Half>
void search(
    const Tensor& positions,
    const Tensor& batch_ptr,
    const Tensor& cell,
    const Tensor& pbc,
    double cutoff,
    int64_t max_pairs,
    Tensor& pair_index,
    Tensor& cell_shifts,
    Tensor& num_pairs,
    Tensor& status_out) {
    const int64_t n_atoms = positions.size(0);
    const int64_t n_structures = batch_ptr.size(0) - 1;
    const auto* pos = positions.data_ptr<scalar_t>();
    const auto* ptr = batch_ptr.data_ptr<int64_t>();
    const auto* cells = cell.data_ptr<scalar_t>();
    const auto* periodic = pbc.data_ptr<bool>();

    std::atomic<int64_t> status{0};

    const auto double_options = positions.options().dtype(torch::kFloat64);
    Tensor duals = torch::empty({n_structures, 9}, double_options);
    Tensor reaches = torch::empty({n_structures, 3}, double_options);
    auto* dual = duals.data_ptr<double>();
    auto* reach = reaches.data_ptr<double>();
    int64_t structure_status = 0;
    for (int64_t b = 0; b < n_structures; ++b) {
        structure_status |= nb::structure_geometry(cells + 9 * b, periodic + 3 * b, cutoff, dual + 9 * b, reach + 3 * b);
        const int64_t segment = nb::check_segment(ptr, b, n_structures, n_atoms);
        if (segment) {
            structure_status |= segment;
            for (int axis = 0; axis < 3; ++axis) reach[3 * b + axis] = -1.0;
        }
    }
    status.fetch_or(structure_status);

    Tensor wraps = torch::empty({n_atoms, 3}, positions.options().dtype(torch::kInt32));
    Tensor reps = torch::empty({n_atoms, 3}, double_options);
    auto* wrap = wraps.data_ptr<int32_t>();
    auto* rep = reps.data_ptr<double>();
    at::parallel_for(0, n_atoms, kGrain, [&](int64_t begin, int64_t end) {
        int64_t local = 0;
        for (int64_t atom = begin; atom < end; ++atom) {
            const int64_t b = nb::segment_of(atom, ptr, n_structures);
            local |= nb::atom_wraps(pos + 3 * atom, dual + 9 * b, wrap + 3 * atom, rep + 3 * atom);
        }
        status.fetch_or(local);
    });

    const nb::SearchView<scalar_t> view{
        pos, ptr, cells, reach, wrap, rep, n_atoms, n_structures, static_cast<scalar_t>(cutoff * cutoff)};

    Tensor counts = torch::zeros({n_atoms}, positions.options().dtype(torch::kInt64));
    auto* count = counts.data_ptr<int64_t>();
    at::parallel_for(0, n_atoms, kGrain, [&](int64_t begin, int64_t end) {
        int64_t local = 0;
        for (int64_t source = begin; source < end; ++source) {
            int64_t found = 0;
            local |= visit_source<Half>(view, source, [&](int64_t, const int64_t*) { ++found; });
            count[source] = found;
        }
        status.fetch_or(local);
    });

    std::vector<int64_t> offsets(static_cast<size_t>(n_atoms));
    int64_t total = 0;
    for (int64_t source = 0; source < n_atoms; ++source) {
        offsets[source] = total;
        total += count[source];
    }
    if (total > max_pairs) status.fetch_or(nb::kCapacity);

    auto* rows = pair_index.data_ptr<int64_t>();
    auto* shifts = cell_shifts.data_ptr<int32_t>();
    at::parallel_for(0, n_atoms, kGrain, [&](int64_t begin, int64_t end) {
        for (int64_t source = begin; source < end; ++source) {
            int64_t row = offsets[source];
            if (row >= max_pairs) continue;
            visit_source<Half>(view, source, [&](int64_t target, const int64_t* shift) {
                if (row < max_pairs) {
                    rows[2 * row] = source;
                    rows[2 * row + 1] = target;
                    for (int axis = 0; axis < 3; ++axis) shifts[3 * row + axis] = static_cast<int32_t>(shift[axis]);
                }
                ++row;
            });
        }
    });

    for (int64_t row = total < max_pairs ? total : max_pairs; row < max_pairs; ++row) {
        const int64_t dead = row % n_atoms;
        rows[2 * row] = dead;
        rows[2 * row + 1] = dead;
        for (int axis = 0; axis < 3; ++axis) shifts[3 * row + axis] = 0;
    }

    num_pairs.data_ptr<int64_t>()[0] = total;
    status_out.data_ptr<int64_t>()[0] = status.load();
}

std::tuple<Tensor, Tensor, Tensor, Tensor> neighbor_pairs_cpu(
    const Tensor& positions,
    const Tensor& batch_ptr,
    const Tensor& cell,
    const Tensor& pbc,
    double cutoff,
    int64_t max_pairs,
    bool half) {
    neighborlist::check_inputs(positions, batch_ptr, cell, pbc, cutoff, max_pairs);

    const Tensor pos = positions.contiguous();
    const Tensor ptr = batch_ptr.contiguous();
    const Tensor cells = cell.contiguous();
    const Tensor periodic = pbc.contiguous();
    const auto index_options = positions.options().dtype(torch::kInt64);
    Tensor pair_index = torch::empty({max_pairs, 2}, index_options);
    Tensor cell_shifts = torch::empty({max_pairs, 3}, positions.options().dtype(torch::kInt32));
    Tensor num_pairs = torch::zeros({1}, index_options);
    Tensor status = torch::zeros({1}, index_options);

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "neighbor_pairs_cpu", [&] {
        if (half) {
            search<scalar_t, true>(pos, ptr, cells, periodic, cutoff, max_pairs, pair_index, cell_shifts, num_pairs, status);
        } else {
            search<scalar_t, false>(pos, ptr, cells, periodic, cutoff, max_pairs, pair_index, cell_shifts, num_pairs, status);
        }
    });
    return {pair_index, cell_shifts, num_pairs, status};
}

}  // namespace

TORCH_LIBRARY_IMPL(neighborlist, CPU, m) {
    m.impl("neighbor_pairs", &neighbor_pairs_cpu);
}
