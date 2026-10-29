// Shared core of `neighborlist::neighbor_pairs`: the periodic geometry of one
// structure, the wrap of one atom and the pair predicate of one (source, target).
// Compiled into both the CPU and the CUDA kernel, so the two decide every
// candidate with the same IEEE operations in the same order (the CUDA file is
// built with --fmad=false) and emit the same rows in the same order.
//
// Portions derived from tonari (https://github.com/songfeitong/tonari @ 803aec8,
// csrc/core/geometry.cpp and csrc/core/pair_policy.h), Copyright (c) 2026
// Feitong Song, MIT License — see LICENSES/tonari.txt.
//
// Conventions:
//   pair (source s, target t, integer cell shift S) is accepted iff
//   d = pos[t] - pos[s] + S @ cell[b] has d.d < cutoff^2, and it is not the
//   zero-shift self pair (s, s, 0). Every periodic image is kept, including
//   several images of one pair and periodic self-images (s, s, S != 0); no
//   minimum-image reduction. Half lists keep the canonical side: s < t, or for
//   s == t the shift whose first nonzero component is negative.
//
// Image pruning: with f = rep[t] - rep[s] the fractional separation of the
// wrapped representatives, the component of d along the normal of cell axis i
// is |f_i + I_i| w_i (w_i = 1 / |dual_i| the perpendicular width), and it is at
// most |d| < cutoff. So only I_i in [-f_i - reach_i, -f_i + reach_i] with
// reach_i = cutoff |dual_i| can hold a pair — usually one or two images per
// axis. The bound is widened by kImageSlack (fractional) so float rounding in
// the final predicate can never accept a pair the bound dropped.

#pragma once

#include <cstdint>

#if defined(__CUDACC__)
#define NEIGHBORLIST_HD __host__ __device__
#define NEIGHBORLIST_INLINE __forceinline__
#else
#define NEIGHBORLIST_HD
#define NEIGHBORLIST_INLINE inline
#endif

namespace neighborlist {

// Bits of the int64 status word, mirrored by neighborlist.SearchStatus.
constexpr int64_t kNonFinitePosition = 1;
constexpr int64_t kWrapOverflow = 2;     // a fractional coordinate outside int32
constexpr int64_t kShiftOverflow = 4;    // an accepted cell shift outside int32
constexpr int64_t kCapacity = 8;         // more pairs than max_pairs
constexpr int64_t kDegenerateCell = 16;  // non-finite cell, or dependent active rows
constexpr int64_t kImageLimit = 32;      // periodic image box over kMaxImages
constexpr int64_t kBadBatchPtr = 64;     // batch_ptr not 0 ... N, nondecreasing

// A tiny cell under a long cutoff is refused rather than enumerated.
constexpr int64_t kMaxImages = int64_t{1} << 24;
// Fractional widening of the per-pair image bound: covers the scalar_t rounding
// of the distance test (about 1e-6 relative in float32) for coordinates within
// ~1e3 cell widths of the origin, at the cost of rarely one extra candidate.
constexpr double kImageSlack = 1e-3;
constexpr double kDoubleEpsilon = 2.220446049250313e-16;
constexpr double kInt32Bound = 2147483648.0;

template <typename T>
NEIGHBORLIST_HD NEIGHBORLIST_INLINE bool is_finite(T x) {
    return x == x && x - x == T(0);
}

// Index of the structure holding `index`: the b with ptr[b] <= index < ptr[b+1].
NEIGHBORLIST_HD NEIGHBORLIST_INLINE int64_t segment_of(int64_t index, const int64_t* ptr, int64_t n_segments) {
    int64_t lower = 0;
    int64_t upper = n_segments;
    while (lower < upper) {
        const int64_t middle = lower + (upper - lower) / 2;
        if (ptr[middle + 1] <= index) {
            lower = middle + 1;
        } else {
            upper = middle;
        }
    }
    return lower < n_segments ? lower : n_segments - 1;
}

// Active-row pseudo-inverse of one cell by one-sided Jacobi on the rows (a Gram
// matrix would square the condition number). Writes `dual[3 * cartesian + axis]`
// for the active axes (zero elsewhere) and `reach[axis]` = cutoff * |dual_axis|,
// the fractional distance a cutoff sphere spans along that axis (zero on the
// inactive ones). A structure that cannot be searched gets reach = -1 on every
// axis and a status bit.
template <typename scalar_t>
NEIGHBORLIST_HD int64_t structure_geometry(
    const scalar_t* cell, const bool* pbc, double cutoff, double* dual, double* reach) {
    for (int k = 0; k < 9; ++k) dual[k] = 0.0;
    for (int axis = 0; axis < 3; ++axis) reach[axis] = 0.0;

    int active_axes[3] = {0, 0, 0};
    int active_count = 0;
    for (int axis = 0; axis < 3; ++axis) {
        if (pbc[axis]) active_axes[active_count++] = axis;
    }
    if (active_count == 0) return 0;

    for (int k = 0; k < 9; ++k) {
        if (!is_finite(cell[k])) {
            for (int axis = 0; axis < 3; ++axis) reach[axis] = -1.0;
            return kDegenerateCell;
        }
    }

    double columns[3][3] = {};
    double right[3][3] = {};
    for (int column = 0; column < active_count; ++column) {
        right[column][column] = 1.0;
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            columns[cartesian][column] =
                static_cast<double>(cell[3 * active_axes[column] + cartesian]);
        }
    }
    for (int sweep = 0; sweep < 64; ++sweep) {
        bool converged = true;
        for (int first = 0; first < active_count; ++first) {
            for (int second = first + 1; second < active_count; ++second) {
                double first_norm = 0.0;
                double second_norm = 0.0;
                double inner = 0.0;
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    first_norm += columns[cartesian][first] * columns[cartesian][first];
                    second_norm += columns[cartesian][second] * columns[cartesian][second];
                    inner += columns[cartesian][first] * columns[cartesian][second];
                }
                if (::fabs(inner) <= 8 * kDoubleEpsilon * ::sqrt(first_norm * second_norm)) {
                    continue;
                }
                converged = false;
                const double tau = (second_norm - first_norm) / (2 * inner);
                // sqrt(1 + x^2), not hypot: sqrt is correctly rounded on both
                // sides, hypot is not, and CPU and CUDA must agree bit for bit.
                const double tangent =
                    tau == 0.0 ? 1.0 : ::copysign(1.0, tau) / (::fabs(tau) + ::sqrt(1.0 + tau * tau));
                const double cosine = 1.0 / ::sqrt(1.0 + tangent * tangent);
                const double sine = cosine * tangent;
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    const double a = columns[cartesian][first];
                    const double b = columns[cartesian][second];
                    columns[cartesian][first] = cosine * a - sine * b;
                    columns[cartesian][second] = sine * a + cosine * b;
                }
                for (int row = 0; row < active_count; ++row) {
                    const double a = right[row][first];
                    const double b = right[row][second];
                    right[row][first] = cosine * a - sine * b;
                    right[row][second] = sine * a + cosine * b;
                }
            }
        }
        if (converged) break;
    }

    double singular[3] = {0.0, 0.0, 0.0};
    double largest = 0.0;
    double smallest = 0.0;
    for (int column = 0; column < active_count; ++column) {
        double norm = 0.0;
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            norm += columns[cartesian][column] * columns[cartesian][column];
        }
        singular[column] = ::sqrt(norm);
        largest = singular[column] > largest ? singular[column] : largest;
        smallest = column == 0 || singular[column] < smallest ? singular[column] : smallest;
    }
    const double tolerance = kDoubleEpsilon * 3 * (largest > 1.0 ? largest : 1.0);
    if (!(smallest > tolerance)) {
        for (int axis = 0; axis < 3; ++axis) reach[axis] = -1.0;
        return kDegenerateCell;
    }

    int64_t images = 1;
    for (int axis_index = 0; axis_index < active_count; ++axis_index) {
        double norm = 0.0;
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            double value = 0.0;
            for (int k = 0; k < active_count; ++k) {
                value += columns[cartesian][k] * right[axis_index][k] / (singular[k] * singular[k]);
            }
            dual[3 * cartesian + active_axes[axis_index]] = value;
            norm += value * value;
        }
        const double span = cutoff * ::sqrt(norm);
        const double images_on_axis = 2.0 * ::ceil(span) + 1.0;
        if (!is_finite(span) || span >= kInt32Bound || static_cast<double>(images) * images_on_axis > kMaxImages) {
            for (int axis = 0; axis < 3; ++axis) reach[axis] = -1.0;
            return kImageLimit;
        }
        reach[active_axes[axis_index]] = span;
        images *= static_cast<int64_t>(images_on_axis);
    }
    return 0;
}

// Integer cell coordinates of one atom — floor of its fractional coordinate on
// each active axis, zero on the inactive ones — and the fractional coordinate
// of its wrapped representative, `rep = fractional - wraps` in [0, 1). Positions
// need not be wrapped: the wrap only centres the image search and is
// compensated in every shift.
template <typename scalar_t>
NEIGHBORLIST_HD int64_t atom_wraps(const scalar_t* position, const double* dual, int32_t* wraps, double* rep) {
    for (int axis = 0; axis < 3; ++axis) {
        wraps[axis] = 0;
        rep[axis] = 0.0;
    }
    for (int cartesian = 0; cartesian < 3; ++cartesian) {
        if (!is_finite(position[cartesian])) return kNonFinitePosition;
    }
    int64_t status = 0;
    for (int axis = 0; axis < 3; ++axis) {
        double fractional = 0.0;
        for (int cartesian = 0; cartesian < 3; ++cartesian) {
            fractional += static_cast<double>(position[cartesian]) * dual[3 * cartesian + axis];
        }
        const double cell_index = ::floor(fractional);
        if (!is_finite(cell_index) || cell_index < -kInt32Bound || cell_index >= kInt32Bound) {
            status |= kWrapOverflow;
            continue;
        }
        wraps[axis] = static_cast<int32_t>(cell_index);
        rep[axis] = fractional - cell_index;
    }
    return status;
}

// Read-only view of everything the per-pair search needs.
template <typename scalar_t>
struct SearchView {
    const scalar_t* positions;  // (N, 3)
    const int64_t* batch_ptr;   // (B + 1,)
    const scalar_t* cell;       // (B, 3, 3), rows are cell vectors
    const double* reach;        // (B, 3), -1 marks a structure that is not searched
    const int32_t* wraps;       // (N, 3)
    const double* reps;         // (N, 3) wrapped fractional representatives
    int64_t n_atoms;
    int64_t n_structures;
    scalar_t cutoff_squared;
};

// The targets one source atom is paired with: [first, end). Empty for a
// structure that is not searched; clamped so a malformed batch_ptr (reported,
// not trusted) never reads out of bounds. A half list starts at the source.
template <bool Half, typename scalar_t>
NEIGHBORLIST_HD NEIGHBORLIST_INLINE void target_range(
    const SearchView<scalar_t>& view, int64_t source, int64_t& structure, int64_t& first, int64_t& end) {
    structure = segment_of(source, view.batch_ptr, view.n_structures);
    const int64_t lo_raw = view.batch_ptr[structure];
    const int64_t hi_raw = view.batch_ptr[structure + 1];
    const int64_t lo = lo_raw < 0 ? 0 : (lo_raw > view.n_atoms ? view.n_atoms : lo_raw);
    end = hi_raw < lo ? lo : (hi_raw > view.n_atoms ? view.n_atoms : hi_raw);
    first = Half ? (source > lo ? source : lo) : lo;
    if (view.reach[3 * structure] < 0.0) end = first;  // reported by the structure pass
}

// Visit every accepted image of one (source, target) pair, images (x, y, z)
// lexicographically ascending. `emit(shift)` receives each one; the return
// value is the status bits raised.
#if defined(__CUDACC__)
#pragma nv_exec_check_disable
#endif
template <bool Half, typename scalar_t, typename Emit>
NEIGHBORLIST_HD NEIGHBORLIST_INLINE int64_t visit_pair(
    const SearchView<scalar_t>& view, int64_t structure, int64_t source, int64_t target, Emit&& emit) {
    const double* reach = view.reach + 3 * structure;
    const double* source_rep = view.reps + 3 * source;
    const double* target_rep = view.reps + 3 * target;
    int64_t low[3];
    int64_t high[3];
    for (int axis = 0; axis < 3; ++axis) {
        const double separation = target_rep[axis] - source_rep[axis];
        const double span = reach[axis] + (reach[axis] > 0.0 ? kImageSlack : 0.0);
        low[axis] = static_cast<int64_t>(::ceil(-separation - span));
        high[axis] = static_cast<int64_t>(::floor(-separation + span));
    }
    const scalar_t* source_position = view.positions + 3 * source;
    const scalar_t* target_position = view.positions + 3 * target;
    const int32_t* source_wraps = view.wraps + 3 * source;
    const int32_t* target_wraps = view.wraps + 3 * target;
    const scalar_t* cell = view.cell + 9 * structure;
    int64_t status = 0;
    for (int64_t ix = low[0]; ix <= high[0]; ++ix) {
        for (int64_t iy = low[1]; iy <= high[1]; ++iy) {
            for (int64_t iz = low[2]; iz <= high[2]; ++iz) {
                const int64_t shift[3] = {
                    ix - target_wraps[0] + source_wraps[0],
                    iy - target_wraps[1] + source_wraps[1],
                    iz - target_wraps[2] + source_wraps[2],
                };
                if (target == source) {
                    if (shift[0] == 0 && shift[1] == 0 && shift[2] == 0) continue;
                    if (Half) {
                        const int64_t leading = shift[0] != 0 ? shift[0] : (shift[1] != 0 ? shift[1] : shift[2]);
                        if (leading > 0) continue;
                    }
                }
                scalar_t distance_squared = scalar_t(0);
                for (int cartesian = 0; cartesian < 3; ++cartesian) {
                    scalar_t component = target_position[cartesian] - source_position[cartesian];
                    for (int axis = 0; axis < 3; ++axis) {
                        component += static_cast<scalar_t>(shift[axis]) * cell[3 * axis + cartesian];
                    }
                    distance_squared += component * component;
                }
                if (!(distance_squared < view.cutoff_squared)) continue;
                if (shift[0] < -2147483648LL || shift[0] > 2147483647LL ||
                    shift[1] < -2147483648LL || shift[1] > 2147483647LL ||
                    shift[2] < -2147483648LL || shift[2] > 2147483647LL) {
                    status |= kShiftOverflow;
                    continue;
                }
                emit(shift);
            }
        }
    }
    return status;
}

// batch_ptr must run 0 = ptr[0] <= ptr[1] <= ... <= ptr[B] = N.
NEIGHBORLIST_HD NEIGHBORLIST_INLINE int64_t check_segment(const int64_t* ptr, int64_t b, int64_t n_structures, int64_t n_atoms) {
    const bool ordered = ptr[b] <= ptr[b + 1];
    const bool starts = b != 0 || ptr[0] == 0;
    const bool ends = b != n_structures - 1 || ptr[n_structures] == n_atoms;
    return ordered && starts && ends ? 0 : kBadBatchPtr;
}

}  // namespace neighborlist

#undef NEIGHBORLIST_INLINE
#undef NEIGHBORLIST_HD
