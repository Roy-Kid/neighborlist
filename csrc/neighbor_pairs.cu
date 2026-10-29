// CUDA implementation of `neighborlist::neighbor_pairs` — the batched, fixed-capacity
// cutoff search. The passes of neighbor_pairs.cpp, one kernel each, over the
// same shared core (pair_search.h); this file is compiled with --fmad=false so
// every candidate is decided by the same IEEE operations as on the CPU, and the
// rows come out identical and in the same (source-major) order.
//
// One warp per source atom: lanes stride over its targets, so a few large
// structures fill the device as well as many small ones. The write pass walks
// the targets in chunks of 32 and places each lane's rows with a warp exclusive
// scan — rows land ordered by target, then image, exactly as the CPU loop
// emits them.
//
// Compile contract: every launch shape is a function of tensor *shapes* (N, B,
// max_pairs) only; the row offsets are a device cumsum; nothing is copied to
// the host and no count is read back. A total above max_pairs keeps the first
// max_pairs rows and raises kCapacity in the status word.
//
// Portions derived from tonari (https://github.com/songfeitong/tonari @ 803aec8),
// Copyright (c) 2026 Feitong Song, MIT License — see LICENSES/tonari.txt.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>
#include <torch/library.h>

#include <tuple>

#include "neighbor_pairs.h"
#include "pair_search.h"

namespace {

namespace nb = neighborlist;
using torch::Tensor;

constexpr int kThreads = 256;
constexpr int kWarp = 32;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ void raise(int64_t* status, int64_t bits) {
    if (bits) atomicOr(reinterpret_cast<unsigned long long*>(status), static_cast<unsigned long long>(bits));
}

inline unsigned int blocks_for(int64_t n) {
    return static_cast<unsigned int>((n + kThreads - 1) / kThreads);
}

template <typename scalar_t>
__global__ void structure_kernel(
    const scalar_t* cells,
    const bool* pbc,
    const int64_t* ptr,
    int64_t n_structures,
    int64_t n_atoms,
    double cutoff,
    double* duals,
    double* reaches,
    int64_t* status) {
    const int64_t b = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (b >= n_structures) return;
    int64_t bits = nb::structure_geometry(cells + 9 * b, pbc + 3 * b, cutoff, duals + 9 * b, reaches + 3 * b);
    const int64_t segment = nb::check_segment(ptr, b, n_structures, n_atoms);
    if (segment) {
        bits |= segment;
        for (int axis = 0; axis < 3; ++axis) reaches[3 * b + axis] = -1.0;
    }
    raise(status, bits);
}

template <typename scalar_t>
__global__ void wraps_kernel(
    const scalar_t* positions,
    const int64_t* ptr,
    const double* duals,
    int64_t n_atoms,
    int64_t n_structures,
    int32_t* wraps,
    double* reps,
    int64_t* status) {
    const int64_t atom = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (atom >= n_atoms) return;
    const int64_t b = nb::segment_of(atom, ptr, n_structures);
    raise(status, nb::atom_wraps(positions + 3 * atom, duals + 9 * b, wraps + 3 * atom, reps + 3 * atom));
}

template <bool Half, typename scalar_t>
__global__ void count_kernel(nb::SearchView<scalar_t> view, int64_t* counts, int64_t* status) {
    const int64_t source = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (source >= view.n_atoms) return;  // warp-uniform
    int64_t structure = 0;
    int64_t first = 0;
    int64_t end = 0;
    nb::target_range<Half>(view, source, structure, first, end);
    int64_t found = 0;
    int64_t bits = 0;
    for (int64_t target = first + lane; target < end; target += kWarp) {
        bits |= nb::visit_pair<Half>(view, structure, source, target, [&](const int64_t*) { ++found; });
    }
    for (int offset = kWarp / 2; offset > 0; offset /= 2) {
        found += __shfl_down_sync(kFullMask, found, offset);
    }
    raise(status, bits);
    if (lane == 0) counts[source] = found;
}

template <bool Half, typename scalar_t>
__global__ void write_kernel(
    nb::SearchView<scalar_t> view,
    const int64_t* inclusive,
    const int64_t* counts,
    int64_t max_pairs,
    int64_t* rows,
    int32_t* shifts,
    int64_t* status) {
    const int64_t source = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kWarp;
    const int lane = threadIdx.x % kWarp;
    if (source >= view.n_atoms) return;  // warp-uniform
    if (lane == 0 && inclusive[source] > max_pairs) raise(status, nb::kCapacity);
    int64_t base = inclusive[source] - counts[source];
    if (base >= max_pairs) return;  // warp-uniform: nothing of this source fits
    int64_t structure = 0;
    int64_t first = 0;
    int64_t end = 0;
    nb::target_range<Half>(view, source, structure, first, end);
    for (int64_t chunk = first; chunk < end; chunk += kWarp) {  // warp-uniform trip count
        const int64_t target = chunk + lane;
        int64_t mine = 0;
        if (target < end) {
            nb::visit_pair<Half>(view, structure, source, target, [&](const int64_t*) { ++mine; });
        }
        int64_t before = mine;  // inclusive scan, then exclusive
        for (int offset = 1; offset < kWarp; offset *= 2) {
            const int64_t up = __shfl_up_sync(kFullMask, before, offset);
            if (lane >= offset) before += up;
        }
        const int64_t chunk_total = __shfl_sync(kFullMask, before, kWarp - 1);
        int64_t row = base + before - mine;
        if (target < end && mine > 0) {
            nb::visit_pair<Half>(view, structure, source, target, [&](const int64_t* shift) {
                if (row < max_pairs) {
                    rows[2 * row] = source;
                    rows[2 * row + 1] = target;
                    for (int axis = 0; axis < 3; ++axis) shifts[3 * row + axis] = static_cast<int32_t>(shift[axis]);
                }
                ++row;
            });
        }
        base += chunk_total;
    }
}

// Rows [total, max_pairs): self-loop (k mod N, k mod N) with zero shift, spread
// over the atoms so no backward gather/scatter sees one index repeated.
__global__ void dead_kernel(const int64_t* total, int64_t n_atoms, int64_t max_pairs, int64_t* rows, int32_t* shifts) {
    const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= max_pairs || row < *total) return;
    const int64_t dead = row % n_atoms;
    rows[2 * row] = dead;
    rows[2 * row + 1] = dead;
    for (int axis = 0; axis < 3; ++axis) shifts[3 * row + axis] = 0;
}

template <bool Half, typename scalar_t>
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
    Tensor& status) {
    const int64_t n_atoms = positions.size(0);
    const int64_t n_structures = batch_ptr.size(0) - 1;
    const auto stream = at::cuda::getCurrentCUDAStream(positions.get_device());
    auto* status_ptr = status.data_ptr<int64_t>();

    const auto double_options = positions.options().dtype(torch::kFloat64);
    Tensor duals = torch::empty({n_structures, 9}, double_options);
    Tensor reaches = torch::empty({n_structures, 3}, double_options);
    if (n_structures > 0) {
        structure_kernel<scalar_t><<<blocks_for(n_structures), kThreads, 0, stream>>>(
            cell.data_ptr<scalar_t>(), pbc.data_ptr<bool>(), batch_ptr.data_ptr<int64_t>(), n_structures, n_atoms,
            cutoff, duals.data_ptr<double>(), reaches.data_ptr<double>(), status_ptr);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (n_atoms == 0) return;  // max_pairs == 0 by contract; num_pairs stays 0

    Tensor wraps = torch::empty({n_atoms, 3}, positions.options().dtype(torch::kInt32));
    Tensor reps = torch::empty({n_atoms, 3}, double_options);
    wraps_kernel<scalar_t><<<blocks_for(n_atoms), kThreads, 0, stream>>>(
        positions.data_ptr<scalar_t>(), batch_ptr.data_ptr<int64_t>(), duals.data_ptr<double>(), n_atoms,
        n_structures, wraps.data_ptr<int32_t>(), reps.data_ptr<double>(), status_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    const nb::SearchView<scalar_t> view{
        positions.data_ptr<scalar_t>(), batch_ptr.data_ptr<int64_t>(), cell.data_ptr<scalar_t>(),
        reaches.data_ptr<double>(), wraps.data_ptr<int32_t>(), reps.data_ptr<double>(), n_atoms, n_structures,
        static_cast<scalar_t>(cutoff * cutoff)};

    Tensor counts = torch::empty({n_atoms}, positions.options().dtype(torch::kInt64));
    count_kernel<Half, scalar_t><<<blocks_for(n_atoms * kWarp), kThreads, 0, stream>>>(
        view, counts.data_ptr<int64_t>(), status_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Row offsets on the device: inclusive[s] - counts[s] is where source s starts.
    const Tensor inclusive = torch::cumsum(counts, 0);
    num_pairs.copy_(inclusive.narrow(0, n_atoms - 1, 1));

    write_kernel<Half, scalar_t><<<blocks_for(n_atoms * kWarp), kThreads, 0, stream>>>(
        view, inclusive.data_ptr<int64_t>(), counts.data_ptr<int64_t>(), max_pairs,
        pair_index.data_ptr<int64_t>(), cell_shifts.data_ptr<int32_t>(), status_ptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (max_pairs == 0) return;  // a zero-block launch is an invalid configuration
    dead_kernel<<<blocks_for(max_pairs), kThreads, 0, stream>>>(
        num_pairs.data_ptr<int64_t>(), n_atoms, max_pairs, pair_index.data_ptr<int64_t>(),
        cell_shifts.data_ptr<int32_t>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<Tensor, Tensor, Tensor, Tensor> neighbor_pairs_cuda(
    const Tensor& positions,
    const Tensor& batch_ptr,
    const Tensor& cell,
    const Tensor& pbc,
    double cutoff,
    int64_t max_pairs,
    bool half) {
    nb::check_inputs(positions, batch_ptr, cell, pbc, cutoff, max_pairs);
    TORCH_CHECK(positions.is_cuda() && batch_ptr.is_cuda() && cell.is_cuda() && pbc.is_cuda(),
                "neighbor_pairs: all inputs must be on the same CUDA device");
    TORCH_CHECK(batch_ptr.device() == positions.device() && cell.device() == positions.device() &&
                    pbc.device() == positions.device(),
                "neighbor_pairs: all inputs must be on the same CUDA device");
    TORCH_CHECK(positions.size(0) * kWarp < (int64_t{1} << 31) * kThreads, "neighbor_pairs: too many atoms for one grid");
    const c10::cuda::CUDAGuard guard(positions.device());

    const Tensor pos = positions.contiguous();
    const Tensor ptr = batch_ptr.contiguous();
    const Tensor cells = cell.contiguous();
    const Tensor periodic = pbc.contiguous();
    const auto index_options = positions.options().dtype(torch::kInt64);
    Tensor pair_index = torch::empty({max_pairs, 2}, index_options);
    Tensor cell_shifts = torch::empty({max_pairs, 3}, positions.options().dtype(torch::kInt32));
    Tensor num_pairs = torch::zeros({1}, index_options);
    Tensor status = torch::zeros({1}, index_options);

    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "neighbor_pairs_cuda", [&] {
        if (half) {
            search<true, scalar_t>(pos, ptr, cells, periodic, cutoff, max_pairs, pair_index, cell_shifts, num_pairs, status);
        } else {
            search<false, scalar_t>(pos, ptr, cells, periodic, cutoff, max_pairs, pair_index, cell_shifts, num_pairs, status);
        }
    });
    return {pair_index, cell_shifts, num_pairs, status};
}

}  // namespace

TORCH_LIBRARY_IMPL(neighborlist, CUDA, m) {
    m.impl("neighbor_pairs", &neighbor_pairs_cuda);
}
