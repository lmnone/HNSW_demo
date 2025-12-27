//
// Created by Konstantin Sobolev on 26/12/2025.
//

#ifndef HNSW_DISTANCE_H
#define HNSW_DISTANCE_H

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include <vector>

// ------------------------- L2 Distance -------------------------
inline float l2_distance_(const std::vector<float> &a, const std::vector<float> &b) {
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}


inline float l2_distance(const std::vector<float> &a, const std::vector<float> &b) {

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

    size_t n = a.size();
    const float *pa = a.data();
    const float *pb = b.data();

    // Accumulator register initialized to [0, 0, 0, 0]
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;
    // Process 4 elements per iteration
    for (; i <= n - 4; i += 4) {
        float32x4_t va = vld1q_f32(pa + i);
        float32x4_t vb = vld1q_f32(pb + i);

        // Compute difference: d = a - b
        float32x4_t diff = vsubq_f32(va, vb);

        // Compute sum += d * d (Fused Multiply-Add)
        sum_vec = vmlaq_f32(sum_vec, diff, diff);
    }

    // Horizontal sum of the 4 lanes in the vector
    float total_sum = vaddvq_f32(sum_vec);

    // Handle tail elements if vector size is not a multiple of 4
    for (; i < n; ++i) {
        float d = pa[i] - pb[i];
        total_sum += d * d;
    }

    return total_sum;

#else
    // Non-ARM fallback — use scalar implementation
    return l2_distance_(a, b);
#endif
}

#endif// HNSW_DISTANCE_H
