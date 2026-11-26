#ifndef COMPACT_SEARCH_H
#define COMPACT_SEARCH_H

#include <cstdint>

#if defined(_WIN32)
#define COMPACT_EXPORT __declspec(dllexport)
#else
#define COMPACT_EXPORT
#endif

// Read-only arrays owned by the caller. CSR offsets include the terminal offset.
// Move IDs are 0: move, 1: Kempe, 2: swap, 3: shift.
// Flags: 1 enables pillar costs, 2 enables exact free-coordinate refinement.
// Negative remaining time is unlimited.
struct SearchInput {
    std::int32_t n, cap, budget, flags;
    double quadratic_factor, reg, degree_factor, remaining_seconds;
    const std::int32_t *initial, *adj_offsets, *adj, *group_offsets, *incidence;
    std::int32_t group_count, pair_count;
    const std::int32_t *left, *right;
    const double* weights;
    std::int32_t move_count;
    const std::int32_t* moves;
    std::int32_t component_count;
    const std::int32_t *component_offsets, *free_edges;
    std::uint32_t (*randbelow)(std::uint32_t);
    double (*random)();
    // Iteration and elapsed seconds are relative to this stage.
    void (*record)(std::int32_t iteration, std::int32_t layers,
                   double value, double elapsed_seconds);
};

struct SearchOutput {
    std::int32_t* layers;
    std::int32_t iterations, updates, timed_out;
    double value;
    // Caller allocates 4 slots for each move counter.
    std::int32_t *attempted, *evaluated, *accepted;
};

// Returns zero on success, 1 for invalid parameters, 2 for an internal failure.
extern "C" COMPACT_EXPORT int compact_search(const SearchInput*, SearchOutput*) noexcept;
extern "C" COMPACT_EXPORT int compact_search_abi_version() noexcept;

#endif
