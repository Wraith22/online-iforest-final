// Benchmark implementation of the binary, axis-parallel, fully sampled forest.
// Keep MT19937 draws in the same order as the NumPy reference implementation.
#include <algorithm>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {
class Random {
public:
    Random(const uint32_t* state, int position) {
        std::stringstream description;
        for (int i = 0; i < 624; ++i) {
            description << state[i] << ' ';
        }
        description << position;
        if (!(description >> engine)) {
            throw std::runtime_error("Invalid MT19937 state");
        }
    }

    void skip_uniforms(int count) { engine.discard(2ULL * count); }

    double uniform(double lower, double upper) {
        const auto high = engine() >> 5;
        const auto low = engine() >> 6;
        const double unit = (high * 67108864.0 + low) / 9007199254740992.0;
        return lower + (upper - lower) * unit;
    }

    int choice(int length) {
        const unsigned maximum = length - 1;
        if (maximum == 0) {
            return 0;
        }
        unsigned mask = maximum;
        for (int shift = 1; shift < 32; shift *= 2) {
            mask |= mask >> shift;
        }
        unsigned index;
        do {
            index = engine() & mask;
        } while (index > maximum);
        return index;
    }

private:
    std::mt19937 engine;
};

struct Node {
    int samples = 0;
    int axis = -1;
    double cut = 0;
};

class Tree {
public:
    Tree(int features, int max_depth, Random& rng)
        : features(features), rng(rng), nodes(1 << (max_depth + 1)),
          lower(nodes.size() * features), upper(nodes.size() * features) {}

    void learn(const double* data, int count, double depth_limit, bool initialize) {
        rng.skip_uniforms(count);
        limit = depth_limit;
        auto indices = row_indices(count);
        if (initialize) {
            build(1, 0, data, indices.data(), count);
        } else {
            update(1, 0, data, indices.data(), count);
        }
    }

    void forget(const double* data, int count) {
        rng.skip_uniforms(count);
        auto indices = row_indices(count);
        remove(1, 0, data, indices.data(), count);
    }

    void paths(const double* data, int count, const double* corrections, int correction_count, double* output) const {
        for (int row = 0; row < count; ++row, data += features) {
            int node = 1;
            int depth = 0;
            while (nodes[node].axis >= 0) {
                const auto& split = nodes[node];
                node = 2 * node + (data[split.axis] >= split.cut);
                ++depth;
            }
            const int leaf_samples = std::max(0, nodes[node].samples);
            if (leaf_samples >= correction_count) {
                throw std::out_of_range("Leaf size outside path length table");
            }
            output[row] = depth + corrections[leaf_samples];
        }
    }

private:
    static constexpr int eta = 32;
    const int features;
    Random& rng;
    std::vector<Node> nodes;
    std::vector<double> lower;
    std::vector<double> upper;
    double limit = 0;

    static std::vector<int> row_indices(int count) {
        std::vector<int> indices(count);
        std::iota(indices.begin(), indices.end(), 0);
        return indices;
    }

    int partition(const Node& node, const double* data, int* indices, int count) const {
        return std::partition(indices, indices + count, [&](int row) {
            return data[row * features + node.axis] < node.cut;
        }) - indices;
    }

    void update_bounds(int node, const double* data, const int* indices, int count) {
        double* low = &lower[node * features];
        double* high = &upper[node * features];
        for (int i = 0; i < count; ++i) {
            const double* row = data + indices[i] * features;
            for (int feature = 0; feature < features; ++feature) {
                low[feature] = std::min(low[feature], row[feature]);
                high[feature] = std::max(high[feature], row[feature]);
            }
        }
    }

    void merge_bounds(int node) {
        for (int f = 0; f < features; ++f) {
            lower[node * features + f] = std::min(lower[2 * node * features + f], lower[(2 * node + 1) * features + f]);
            upper[node * features + f] = std::max(upper[2 * node * features + f], upper[(2 * node + 1) * features + f]);
        }
    }

    void build(int node, int depth, const double* data, int* indices, int count) {
        auto& current = nodes[node];
        current = {count, -1, 0};
        std::fill_n(&lower[node * features], features, std::numeric_limits<double>::infinity());
        std::fill_n(&upper[node * features], features, -std::numeric_limits<double>::infinity());
        update_bounds(node, data, indices, count);
        if (count < (eta << depth) || depth >= limit) {
            return;
        }
        current.axis = rng.choice(features);
        const int axis_index = node * features + current.axis;
        current.cut = rng.uniform(lower[axis_index], upper[axis_index]);
        const int left = partition(current, data, indices, count);
        build(2 * node, depth + 1, data, indices, left);
        build(2 * node + 1, depth + 1, data, indices + left, count - left);
    }

    void update(int node, int depth, const double* data, int* indices, int count) {
        auto& current = nodes[node];
        current.samples += count;
        update_bounds(node, data, indices, count);
        if (current.axis >= 0) {
            const int left = partition(current, data, indices, count);
            update(2 * node, depth + 1, data, indices, left);
            update(2 * node + 1, depth + 1, data, indices + left, count - left);
        } else if (current.samples >= (eta << depth) && depth < limit) {
            const int samples = current.samples;
            std::vector<double> simulated(samples * features);
            for (int row = 0; row < samples; ++row) {
                for (int feature = 0; feature < features; ++feature) {
                    const int index = node * features + feature;
                    simulated[row * features + feature] = rng.uniform(lower[index], upper[index]);
                }
            }
            auto simulated_indices = row_indices(samples);
            build(node, depth, simulated.data(), simulated_indices.data(), samples);
        }
    }

    void collapse(int node) {
        if (nodes[node].axis >= 0) {
            collapse(2 * node);
            collapse(2 * node + 1);
            merge_bounds(node);
            nodes[node].axis = -1;
        }
    }

    void remove(int node, int depth, const double* data, int* indices, int count) {
        auto& current = nodes[node];
        current.samples -= count;
        if (current.axis < 0) {
            return;
        }
        if (current.samples < (eta << depth)) {
            collapse(node);
            return;
        }
        const int left = partition(current, data, indices, count);
        remove(2 * node, depth + 1, data, indices, left);
        remove(2 * node + 1, depth + 1, data, indices + left, count - left);
        merge_bounds(node);
    }
};
}  // namespace

extern "C" int benchmark_binary_forest(
    const double* data, int rows, int features, int window, int max_depth,
    const uint32_t* seed_state, int seed_position, const double* limits,
    const double* corrections, int correction_count, double* paths
) {
    constexpr int tree_count = 32;
    constexpr int batch_size = 100;
    try {
        Random rng(seed_state, seed_position);
        std::vector<Tree> trees;
        trees.reserve(tree_count);
        for (int i = 0; i < tree_count; ++i) {
            trees.emplace_back(features, max_depth, rng);
        }
        const int batches = rows / batch_size + 1;
        int start = 0;
        for (int batch = 0; batch < batches; ++batch) {
            const int count = rows / batches + (batch < rows % batches);
            const int stop = start + count;
            const int oldest = std::max(0, start - window);
            const int next_oldest = std::max(0, stop - window);
            const auto* batch_data = data + start * features;
            for (auto& tree : trees) {
                tree.learn(batch_data, count, limits[stop - oldest], batch == 0);
            }
            if (next_oldest > oldest) {
                for (auto& tree : trees) {
                    tree.forget(data + oldest * features, next_oldest - oldest);
                }
            }
            for (int i = 0; i < tree_count; ++i) {
                trees[i].paths(batch_data, count, corrections, correction_count, paths + i * rows + start);
            }
            start = stop;
        }
    } catch (...) {
        return 1;
    }
    return 0;
}
