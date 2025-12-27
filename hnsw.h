#ifndef HNSW_HNSW_H
#define HNSW_HNSW_H

#include "distance.h"
#include <algorithm>
#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>

struct Node {
    std::vector<float> vec;
    std::vector<std::vector<int>> neighbors;
    int level;
    mutable std::shared_mutex node_mutex;// Protects neighbors list

    Node(const std::vector<float> &v, int lvl)
        : vec(v), neighbors(lvl + 1), level(lvl) {}
};

class HNSW {
public:
    HNSW(int dim, int M = 16, int ef_construction = 200)
        : dim_(dim), M_(M), ef_(ef_construction), entry_point_(-1), max_level_(-1) {
        nodes_.reserve(100000);
    }

    // Parallel batch insertion
    void insert_batch(const std::vector<std::vector<float>> &data, int num_threads = 8) {
        if (data.empty()) return;

        // Phase 1: Sequential Core (Stabilizes the top of the graph)
        size_t core_size = std::min(data.size(), (size_t) 500);
        for (size_t i = 0; i < core_size; ++i) {
            insert_internal(data[i]);
        }

        if (core_size >= data.size()) return;

        // Phase 2: Parallel Workers
        std::vector<std::thread> workers;
        std::atomic<size_t> next_idx(core_size);
        for (int i = 0; i < num_threads; ++i) {
            workers.emplace_back([&]() {
                while (true) {
                    size_t idx = next_idx.fetch_add(1);
                    if (idx >= data.size()) break;
                    insert_internal(data[idx]);
                }
            });
        }
        for (auto &t: workers) t.join();
    }

    void insert(const std::vector<float> &vec) {
        insert_internal(vec);
    }

    std::vector<int> search(const std::vector<float> &query, int k, int ef_search = -1) const;

private:
    int dim_, M_, ef_;
    std::vector<std::unique_ptr<Node>> nodes_;// Unique_ptr ensures stable memory addresses
    std::atomic<int> entry_point_;
    std::atomic<int> max_level_;
    mutable std::shared_mutex global_lock_;// For adding to nodes_ vector and max_level

    // Thread-local visited list for 0-contention search
    struct VisitedList {
        std::vector<unsigned int> list;
        unsigned int version = 0;
    };
    static thread_local VisitedList tl_visited;

    void prepare_visited_list() const {
        if (tl_visited.list.size() < nodes_.size() + 1024) {
            tl_visited.list.resize(nodes_.size() + 8192, 0);
        }
        if (++tl_visited.version == 0) {
            std::fill(tl_visited.list.begin(), tl_visited.list.end(), 0);
            tl_visited.version = 1;
        }
    }

    void insert_internal(const std::vector<float> &vec);
    std::vector<int> search_layer_internal(const std::vector<float> &q, int entry, int level, int ef) const;
    void prune_neighbors_heuristic(int base_id, std::vector<int> &neighbors);
};

// Thread-local storage definition
thread_local HNSW::VisitedList HNSW::tl_visited;

inline void HNSW::insert_internal(const std::vector<float> &vec) {
    // Generate level
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    int lvl = 0;
    while (dist(gen) < 0.5f && lvl < 16) ++lvl;

    int new_id;
    int curr_ep;
    int max_l;

    // 1. Register new node
    {
        std::unique_lock lock(global_lock_);
        new_id = nodes_.size();
        nodes_.push_back(std::make_unique<Node>(vec, lvl));
        curr_ep = entry_point_.load();
        max_l = max_level_.load();

        if (curr_ep == -1) {
            entry_point_ = new_id;
            max_level_ = lvl;
            return;
        }
    }

    // 2. Greedy search down to lvl
    int ep = curr_ep;
    for (int l = max_l; l > lvl; --l) {
        auto res = search_layer_internal(vec, ep, l, 1);
        if (!res.empty()) ep = res[0];
    }

    // 3. Connect layers
    for (int l = std::min(lvl, max_l); l >= 0; --l) {
        auto candidates = search_layer_internal(vec, ep, l, ef_);

        // Node's outgoing neighbors (No lock needed for self initialization)
        nodes_[new_id]->neighbors[l] = candidates;
        prune_neighbors_heuristic(new_id, nodes_[new_id]->neighbors[l]);

        // Link neighbors TO new node (Locking neighbors)
        std::vector<int> final_n_ids = nodes_[new_id]->neighbors[l];
        for (int nb: final_n_ids) {
            std::unique_lock nb_lock(nodes_[nb]->node_mutex);
            nodes_[nb]->neighbors[l].push_back(new_id);
            if (nodes_[nb]->neighbors[l].size() > (size_t) ((l == 0) ? M_ * 2 : M_)) {
                prune_neighbors_heuristic(nb, nodes_[nb]->neighbors[l]);
            }
        }
        if (!candidates.empty()) ep = candidates[0];
    }

    // 4. Update global peak
    if (lvl > max_l) {
        std::unique_lock lock(global_lock_);
        if (lvl > max_level_) {
            max_level_ = lvl;
            entry_point_ = new_id;
        }
    }
}

inline std::vector<int> HNSW::search_layer_internal(const std::vector<float> &q, int entry, int level, int ef) const {
    using PQElem = std::pair<float, int>;
    std::priority_queue<PQElem> top;
    std::priority_queue<PQElem, std::vector<PQElem>, std::greater<PQElem>> cand;

    prepare_visited_list();
    float d0 = l2_distance(q, nodes_[entry]->vec);
    top.emplace(d0, entry);
    cand.emplace(d0, entry);
    tl_visited.list[entry] = tl_visited.version;

    while (!cand.empty()) {
        auto [d_curr, curr] = cand.top();
        cand.pop();

        if (top.size() >= (size_t) ef && d_curr > top.top().first) break;

        // Copy neighbors under shared lock to minimize blocking
        std::vector<int> nbs;
        {
            std::shared_lock nb_read(nodes_[curr]->node_mutex);
            if (level < nodes_[curr]->neighbors.size())
                nbs = nodes_[curr]->neighbors[level];
        }

        for (int nb: nbs) {
            if (tl_visited.list[nb] == tl_visited.version) continue;
            tl_visited.list[nb] = tl_visited.version;

            float d = l2_distance(q, nodes_[nb]->vec);
            if (top.size() < (size_t) ef || d < top.top().first) {
                cand.emplace(d, nb);
                top.emplace(d, nb);
                if (top.size() > (size_t) ef) top.pop();
            }
        }
    }

    std::vector<int> res;
    while (!top.empty()) {
        res.push_back(top.top().second);
        top.pop();
    }
    std::reverse(res.begin(), res.end());
    return res;
}

inline void HNSW::prune_neighbors_heuristic(int base_id, std::vector<int> &neighbors) {
    if (neighbors.size() < (size_t) M_) return;

    std::vector<std::pair<float, int>> scored;
    for (int nb: neighbors) scored.push_back({l2_distance(nodes_[base_id]->vec, nodes_[nb]->vec), nb});
    std::sort(scored.begin(), scored.end());

    std::vector<int> selected;
    for (auto &pair: scored) {
        bool good = true;
        for (int s: selected) {
            if (l2_distance(nodes_[pair.second]->vec, nodes_[s]->vec) < pair.first) {
                good = false;
                break;
            }
        }
        if (good) selected.push_back(pair.second);
        if (selected.size() >= (size_t) M_) break;
    }
    neighbors.swap(selected);
}

inline std::vector<int> HNSW::search(const std::vector<float> &query, int k, int ef_search) const {
    std::shared_lock lock(global_lock_);
    int ep = entry_point_.load();
    if (ep == -1) return {};

    int ef = (ef_search > 0) ? ef_search : std::max(ef_, k);
    int max_l = max_level_.load();

    for (int l = max_l; l > 0; --l) {
        auto res = search_layer_internal(query, ep, l, 1);
        if (!res.empty()) ep = res[0];
    }

    auto candidates = search_layer_internal(query, ep, 0, ef);
    if (candidates.size() > (size_t) k) candidates.resize(k);
    return candidates;
}

#endif