#include "compact_search.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Layers = std::vector<std::int32_t>;
constexpr double tolerance = 1e-9;

class Search {
public:
    Search(const SearchInput& input, SearchOutput& output)
        : in(input), out(output), started(Clock::now()), labels(in.cap), seen(in.n) {
        for (int edge = 0; edge < in.n; ++edge) {
            if (in.adj_offsets[edge] != in.adj_offsets[edge + 1])
                constrained.push_back(edge);
        }
        if (!(in.flags & 2)) return;
        pair_neighbors.resize(in.n);
        group_memberships.resize(in.n);
        if (in.quadratic_factor) {
            for (int k = 0; k < in.pair_count; ++k) {
                const int left = in.left[k], right = in.right[k];
                pair_neighbors[left].emplace_back(right, in.weights[k]);
                pair_neighbors[right].emplace_back(left, in.weights[k]);
            }
        }
        if (in.flags & 1) {
            for (int group = 0; group < in.group_count; ++group) {
                for (int k = in.group_offsets[group]; k < in.group_offsets[group + 1]; ++k) {
                    const int edge = in.incidence[k];
                    if (in.adj_offsets[edge] == in.adj_offsets[edge + 1])
                        group_memberships[edge].push_back(group);
                }
            }
        }
    }

    void run() {
        std::fill_n(out.attempted, 4, 0);
        std::fill_n(out.evaluated, 4, 0);
        std::fill_n(out.accepted, 4, 0);
        out.iterations = 0;
        out.updates = 0;
        Layers best(in.initial, in.initial + in.n);
        double best_value = cost(best);
        record(best, best_value);
        if (in.flags & 2) {
            Layers refined = best;
            refine(refined);
            double refined_value = cost(refined);
            if (better(refined, refined_value, best, best_value))
                record(refined, refined_value);
            best = std::move(refined);
            best_value = refined_value;
        }

        Layers current = best;
        Layers candidate(in.n);
        double current_value = best_value;
        const double start_temperature = std::max(1e-3, best_value / std::sqrt(in.n));
        for (int index = 0; index < in.budget && !expired(); ++index) {
            ++out.iterations;
            const int move = in.moves[draw(in.move_count)];
            ++out.attempted[move];
            if (!propose(current, candidate, move)) continue;
            const double value = cost(candidate);
            ++out.evaluated[move];
            const double delta = value - current_value;
            const double progress = static_cast<double>(index) / std::max(1, in.budget - 1);
            const double temperature = start_temperature * std::pow(0.001, progress);
            if (delta <= 0 || accept_uphill(delta, temperature)) {
                ++out.accepted[move];
                current.swap(candidate);
                current_value = value;
                if (better(current, current_value, best, best_value)) {
                    best = current;
                    best_value = current_value;
                    record(best, best_value);
                }
            }
        }

        if ((in.flags & 2) && out.iterations) {
            Layers refined = best;
            refine(refined);
            const double value = cost(refined);
            if (better(refined, value, best, best_value)) {
                best = std::move(refined);
                best_value = value;
                record(best, best_value);
            }
        }
        std::copy(best.begin(), best.end(), out.layers);
        out.value = best_value;
        out.timed_out = expired();
    }

private:
    const SearchInput& in;
    SearchOutput& out;
    Clock::time_point started;
    Layers constrained, labels, seen, chain;
    std::vector<std::vector<std::pair<int, double>>> pair_neighbors;
    std::vector<Layers> group_memberships;

    double elapsed() const {
        return std::chrono::duration<double>(Clock::now() - started).count();
    }

    bool expired() const {
        return in.remaining_seconds >= 0 && elapsed() >= in.remaining_seconds;
    }

    int draw(int limit) const {
        const auto value = in.randbelow(static_cast<std::uint32_t>(limit));
        if (value >= static_cast<std::uint32_t>(limit))
            throw std::runtime_error("Integer random callback failed.");
        return static_cast<int>(value);
    }

    bool accept_uphill(double delta, double temperature) const {
        const double value = in.random();
        if (!(0 <= value && value < 1))
            throw std::runtime_error("Unit random callback failed.");
        return value < std::exp(-delta / temperature);
    }

    int count(const Layers& layers) const {
        return *std::max_element(layers.begin(), layers.end()) + 1;
    }

    void record(const Layers& layers, double value) const {
        in.record(out.iterations, count(layers), value, elapsed());
    }

    bool better(const Layers& candidate, double value, const Layers& best, double best_value) const {
        return value < best_value - tolerance ||
               (value <= best_value && count(candidate) < count(best));
    }

    bool compact(Layers& layers) {
        std::fill(labels.begin(), labels.end(), -1);
        for (int layer : layers) labels[layer] = 0;
        int occupied = 0;
        bool changed = false;
        for (int& label : labels) {
            if (label == 0) label = occupied++;
        }
        for (int& layer : layers) {
            changed |= labels[layer] != layer;
            layer = labels[layer];
        }
        return changed;
    }

    std::pair<int, int> sample_layers() const {
        // Match random.sample(range(cap), 2): CPython uses a pool through
        // size 21 and duplicate rejection for larger populations.
        const int source = draw(in.cap);
        if (in.cap <= 21) {
            const int pick = draw(in.cap - 1);
            return {source, pick == source ? in.cap - 1 : pick};
        }
        int target = draw(in.cap);
        while (target == source) target = draw(in.cap);
        return {source, target};
    }

    bool propose(const Layers& current, Layers& candidate, int move) {
        if (in.cap < 2) return false;
        if (move >= 2) {
            auto [source, target] = sample_layers();
            if (move == 3) {
                const int used = count(current);
                // After compaction, shifting into an empty suffix appends
                // the layer directly after the other occupied levels.
                target = std::min(target, used - 1);
                if (source >= used || source == target) return false;
                for (int edge = 0; edge < in.n; ++edge) {
                    const int layer = current[edge];
                    int replacement = layer;
                    if (layer == source) replacement = target;
                    else if (source < layer && layer <= target) --replacement;
                    else if (target <= layer && layer < source) ++replacement;
                    candidate[edge] = replacement;
                }
                return true;
            }
            for (int edge = 0; edge < in.n; ++edge) {
                const int layer = current[edge];
                candidate[edge] = layer == source ? target : layer == target ? source : layer;
            }
        } else {
            if (constrained.empty()) return false;
            const int edge = constrained[draw(static_cast<int>(constrained.size()))];
            const int source = current[edge];
            int target = draw(in.cap - 1);
            if (target >= source) ++target;
            if (move == 0) {
                for (int k = in.adj_offsets[edge]; k < in.adj_offsets[edge + 1]; ++k) {
                    if (current[in.adj[k]] == target) return false;
                }
                candidate = current;
                candidate[edge] = target;
            } else {
                candidate = current;
                std::fill(seen.begin(), seen.end(), 0);
                chain.clear();
                chain.push_back(edge);
                seen[edge] = 1;
                for (std::size_t index = 0; index < chain.size(); ++index) {
                    const int member = chain[index];
                    candidate[member] = current[member] == source ? target : source;
                    for (int k = in.adj_offsets[member]; k < in.adj_offsets[member + 1]; ++k) {
                        const int neighbor = in.adj[k];
                        if (!seen[neighbor] && (current[neighbor] == source || current[neighbor] == target)) {
                            seen[neighbor] = 1;
                            chain.push_back(neighbor);
                        }
                    }
                }
            }
        }
        compact(candidate);
        return candidate != current;
    }

    double cost(const Layers& layers) const {
        double pillar = 0, degree_span = 0, energy = 0;
        if (in.flags & 1) {
            for (int group = 0; group < in.group_count; ++group) {
                int low = in.cap, high = 0;
                for (int k = in.group_offsets[group]; k < in.group_offsets[group + 1]; ++k) {
                    const int layer = layers[in.incidence[k]];
                    low = std::min(low, layer);
                    high = std::max(high, layer);
                }
                const double span = high - low;
                pillar += span;
                if (in.degree_factor)
                    degree_span += (in.group_offsets[group + 1] - in.group_offsets[group]) * span * span;
            }
        }
        if (in.quadratic_factor) {
            for (int k = 0; k < in.pair_count; ++k) {
                const double gap = layers[in.left[k]] - layers[in.right[k]];
                energy += in.weights[k] * (gap * gap);
            }
            if (in.reg) {
                // All unordered squared gaps reduce to two moments.
                double sum = 0, squares = 0;
                for (double layer : layers) {
                    sum += layer;
                    squares += layer * layer;
                }
                energy += in.reg * (in.n * squares - sum * sum);
            }
        }
        return pillar + in.degree_factor * degree_span + in.quadratic_factor * energy;
    }

    void refine(Layers& layers) {
        std::vector<double> deltas(in.cap);
        while (!expired()) {
            bool moved = false;
            double total = 0;
            for (int layer : layers) total += layer;
            for (int component = 0; component < in.component_count; ++component) {
                for (int k = in.component_offsets[component]; k < in.component_offsets[component + 1]; ++k) {
                    if (expired()) {
                        compact(layers);
                        return;
                    }
                    const int edge = in.free_edges[k], old = layers[edge];
                    std::fill(deltas.begin(), deltas.end(), 0.0);
                    for (int group : group_memberships[edge]) {
                        int low = in.cap, high = 0;
                        for (int member = in.group_offsets[group]; member < in.group_offsets[group + 1]; ++member) {
                            if (in.incidence[member] == edge) continue;
                            const int layer = layers[in.incidence[member]];
                            low = std::min(low, layer);
                            high = std::max(high, layer);
                        }
                        const double old_span = std::max(high, old) - std::min(low, old);
                        const double degree = in.group_offsets[group + 1] - in.group_offsets[group];
                        for (int target = 0; target < in.cap; ++target) {
                            const double span = std::max(high, target) - std::min(low, target);
                            deltas[target] += span - old_span;
                            if (in.degree_factor)
                                deltas[target] += in.degree_factor * degree * (span * span - old_span * old_span);
                        }
                    }
                    if (in.quadratic_factor) {
                        // Uniform regularization couples every other edge,
                        // including those in different shared-vertex components.
                        double weight = in.reg * (in.n - 1);
                        double weighted_layer = in.reg * (total - old);
                        for (const auto& [neighbor, shared_weight] : pair_neighbors[edge]) {
                            weight += shared_weight;
                            weighted_layer += shared_weight * layers[neighbor];
                        }
                        for (int target = 0; target < in.cap; ++target) {
                            deltas[target] += in.quadratic_factor * (target - old) *
                                ((static_cast<double>(target) + old) * weight - 2 * weighted_layer);
                        }
                    }
                    const auto best = std::min_element(deltas.begin(), deltas.end());
                    if (*best < -tolerance) {
                        const int target = static_cast<int>(best - deltas.begin());
                        layers[edge] = target;
                        total += target - old;
                        ++out.updates;
                        moved = true;
                    }
                }
            }
            const bool compressed = compact(layers);
            if (!moved && !compressed) break;
        }
    }
};
}

extern "C" COMPACT_EXPORT int compact_search_abi_version() noexcept {
    return 1;
}

extern "C" COMPACT_EXPORT int compact_search(const SearchInput* input, SearchOutput* output) noexcept {
    if (!input || !output || input->n < 1 || input->cap < 1 ||
        input->budget < 0 || input->move_count < 1 || !input->randbelow ||
        !input->random || !input->record) return 1;
    try {
        Search(*input, *output).run();
        return 0;
    } catch (...) {
        return 2;
    }
}
