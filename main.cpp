#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <vector>

#include "cmd_args.h"
#include "hnsw.h"

// ------------------------- Search -------------------------
//------------------------- Exact KNN -------------------------
std::vector<int> exact_knn_L2(const std::vector<std::vector<float>> &data, const std::vector<float> &query, int k) {
    std::vector<std::pair<float, int>> dist;
    for (int i = 0; i < (int) data.size(); i++)
        dist.emplace_back(l2_distance(query, data[i]), i);

    std::nth_element(dist.begin(), dist.begin() + k, dist.end());
    dist.resize(k);
    std::sort(dist.begin(), dist.end());

    std::vector<int> res;
    for (auto &[d, id]: dist)
        res.push_back(id);
    return res;
}

// ------------------------- Orthonormal Centers -------------------------
std::vector<std::vector<float>> generate_well_separated_centers(int dim, int nclusters, float min_dist) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    std::vector<std::vector<float>> centers;

    while (centers.size() < (size_t) nclusters) {
        std::vector<float> candidate(dim);
        for (float &x: candidate) x = dist(rng);

        bool too_close = false;
        for (const auto &c: centers) {
            if (std::sqrt(l2_distance(candidate, c)) < min_dist) {
                too_close = true;
                break;
            }
        }
        if (!too_close) centers.push_back(candidate);
    }
    return centers;
}
// ------------------------- Sample Near -------------------------
std::vector<float> sample_near(const std::vector<float> &center, float sigma, std::mt19937 &rng) {
    std::normal_distribution<float> noise(0.0f, sigma);
    std::vector<float> v = center;
    for (float &x: v)
        x += noise(rng);
    return v;
}

// ------------------------- Test UT -------------------------

void test_hnsw_vs_exact_knn(const CmdArgs &p) {
    std::cout << "[UT] HNSW vs Exact KNN (L2)\n";

    HNSW index(p.dim, p.M, p.efc);
    std::mt19937 rng(p.seed);

    // --- 1. DATA GENERATION ---
    auto centers = generate_well_separated_centers(
            p.dim, p.clusters, p.center_dist);

    std::vector<std::vector<float>> dataset;
    dataset.reserve(p.clusters * p.pts);

    for (int c = 0; c < p.clusters; c++) {
        for (int i = 0; i < p.pts; i++) {
            dataset.push_back(sample_near(centers[c], p.sigma, rng));
        }
    }

    // --- 2. INDEX BUILD (single vs multi-thread) ---
    auto t0_build = std::chrono::high_resolution_clock::now();

    if (p.threads <= 1) {
        std::cout << "Starting single-threaded index build...\n";
        for (const auto &v: dataset) {
            index.insert(v);
        }
    } else {
        std::cout << "Starting parallel index build with "
                  << p.threads << " threads...\n";
        index.insert_batch(dataset, p.threads);
    }

    auto t1_build = std::chrono::high_resolution_clock::now();
    double build_time =
            std::chrono::duration<double>(t1_build - t0_build).count();

    std::cout << "[TIME] Total index insert: "
              << build_time << " sec\n";

    // --- 3. QUERY / SEARCH ---
    int total_queries = 0;
    int top1_correct = 0;
    float avg_recall = 0.0f;
    double search_time_total = 0.0;

    // Pre-generate queries
    std::vector<std::vector<float>> queries;
    queries.reserve(p.clusters * p.queries);

    for (int c = 0; c < p.clusters; c++) {
        for (int q = 0; q < p.queries; q++) {
            queries.push_back(sample_near(centers[c], p.sigma, rng));
        }
    }

    for (const auto &query_vec: queries) {
        // Exact KNN (not timed)
        auto exact = exact_knn_L2(dataset, query_vec, p.k);

        // Approximate search (timed)
        auto t0 = std::chrono::high_resolution_clock::now();
        auto approx = index.search(query_vec, p.k, p.efs);
        auto t1 = std::chrono::high_resolution_clock::now();

        search_time_total +=
                std::chrono::duration<double>(t1 - t0).count();

        int hit = 0;
        for (int id: approx) {
            if (std::find(exact.begin(), exact.end(), id) != exact.end())
                hit++;
        }
        avg_recall += float(hit) / p.k;

        if (!approx.empty() && !exact.empty() && approx[0] == exact[0])
            top1_correct++;

        total_queries++;
    }

    avg_recall /= total_queries;
    double avg_search_time = search_time_total / total_queries;

    // --- RESULTS ---
    std::cout << "Top-1 accuracy: "
              << float(top1_correct) / total_queries << "\n";
    std::cout << "Recall@" << p.k << ": " << avg_recall << "\n";
    std::cout << "[TIME] Avg search per query: "
              << avg_search_time << " sec\n";

    if (avg_recall < 0.95f) {
        std::cout << "[FAIL] Recall is too low: "
                  << avg_recall << "\n";
    } else {
        std::cout << "[PASS] Exact KNN validation\n";
    }

    assert(avg_recall > 0.95f);
}

//------------------------ Pretty print confusion matrix -------------------------
void print_normalized_confusion_matrix(
        const std::vector<std::vector<int>> &cm) {
    int C = cm.size();

    // column sums (true labels)
    std::vector<float> col_sum(C, 0.0f);
    for (int pred = 0; pred < C; pred++)
        for (int true_c = 0; true_c < C; true_c++)
            col_sum[true_c] += cm[pred][true_c];

    std::cout << "\nNormalized confusion matrix "
              << "(rows = predicted, cols = true)\n\n    ";

    for (int j = 0; j < C; j++)
        std::cout << " T" << j << "    ";
    std::cout << "\n";

    for (int i = 0; i < C; i++) {
        std::cout << "P" << i << " ";
        for (int j = 0; j < C; j++) {
            float v = col_sum[j] > 0
                              ? cm[i][j] / col_sum[j]
                              : 0.0f;
            std::cout << std::setw(6)
                      << std::fixed << std::setprecision(2)
                      << v << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// ------------------------- Majority vote -------------------------
int majority_vote(const std::vector<int> &labels, int nclusters) {
    std::vector<int> cnt(nclusters, 0);
    for (int c: labels) cnt[c]++;
    return std::max_element(cnt.begin(), cnt.end()) - cnt.begin();
}

float derive_recall_from_confusion(const std::vector<std::vector<int>> &cm) {
    int C = cm.size();
    int total_correct = 0;
    int total_queries = 0;
    for (int t = 0; t < C; t++) {
        total_correct += cm[t][t];// diagonal
        for (int p = 0; p < C; p++)
            total_queries += cm[p][t];// sum of column
    }
    return float(total_correct) / total_queries;// micro-average recall
}

void test_hnsw_per_cluster_precision(const CmdArgs &p) {
    std::cout << "\n[UT] HNSW per-cluster precision + confusion matrix\n";

    HNSW index(p.dim, p.M, p.efc);
    std::mt19937 rng(p.seed);

    // --- generate well-separated centers ---
    auto centers = generate_well_separated_centers(p.dim, p.clusters, p.center_dist);

    std::vector<std::vector<float>> dataset;
    std::vector<int> labels;

    // --- BUILD INDEX: measure time for insert only ---
    double build_time = 0.0;
    for (int c = 0; c < p.clusters; c++) {
        for (int i = 0; i < p.pts; i++) {
            auto v = sample_near(centers[c], p.sigma, rng);
            dataset.push_back(v);
            labels.push_back(c);

            auto t0 = std::chrono::high_resolution_clock::now();
            index.insert(v);// timed
            auto t1 = std::chrono::high_resolution_clock::now();
            build_time += std::chrono::duration<double>(t1 - t0).count();
        }
    }
    std::cout << "[TIME] Total index insert: " << build_time << " sec\n";

    // --- PREPARE confusion matrix ---
    std::vector<std::vector<int>> confusion(
            p.clusters, std::vector<int>(p.clusters, 0));

    int total_queries = 0;

    // --- QUERY / SEARCH: measure time for search only ---
    double search_time_total = 0.0;
    for (int true_c = 0; true_c < p.clusters; true_c++) {
        for (int q = 0; q < p.queries; q++) {
            auto query = sample_near(centers[true_c], p.sigma, rng);

            auto t0 = std::chrono::high_resolution_clock::now();
            auto knn = index.search(query, p.k, p.efs);// timed
            auto t1 = std::chrono::high_resolution_clock::now();
            search_time_total += std::chrono::duration<double>(t1 - t0).count();

            std::vector<int> knn_labels;
            for (int id: knn) knn_labels.push_back(labels[id]);

            int pred_c = majority_vote(knn_labels, p.clusters);
            confusion[pred_c][true_c]++;

            total_queries++;
        }
    }

    double avg_search_time = search_time_total / total_queries;
    std::cout << "[TIME] Avg search per query: " << avg_search_time << " sec\n";

    // --- PRINT confusion matrix ---
    print_normalized_confusion_matrix(confusion);

    float recall = derive_recall_from_confusion(confusion);
    std::cout << "[UT2] Recall: " << recall << "\n";
}

// ------------------------- Main -------------------------
int main(int argc, char **argv) {
    auto args = parse_args(argc, argv);

    if (!args.ut1 && !args.ut2) {
        print_usage(argv[0]);
        return 0;
    }

    if (args.ut1) {
        test_hnsw_vs_exact_knn(args);
    }

    if (args.ut2) {
        test_hnsw_per_cluster_precision(args);
    }

    return 0;
}
