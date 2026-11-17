#ifndef MX_TILED_ATTENTION_H
#define MX_TILED_ATTENTION_H
/* Exact selected QK/PV loops. Including this file changes no defaults.
 * Profile evidence and per-mix adoption are in gen/micromix_profiles.json.
 * Compile with -fno-fast-math -ffp-contract=off. */
#include "mx_gqa_attention.h"
#include "qwen_gqa_attention.h"
#include "mx_mha_attention.h"
static inline int mx_tiled_gqa8_attention_bf16(const mxmm_gqa_attention_args *a,
                                               mxmm_attention_mode requested, uint16_t *output,
                                               float *workspace, size_t workspace_floats,
                                               mxmm_attention_mode *actual_mode) {
  size_t need = mxmm_gqa_attention_workspace_floats(a, requested);
  if (!need || !output || !workspace || !a->query || !a->keys || !a->values ||
      !isfinite(a->qk_scale) || a->qk_scale <= 0)
    return -1;
  if (workspace_floats < need)
    return -2;
  mxmm_attention_mode mode =
      requested == MXMM_ATTN_GROUPED8_EXPLICIT_FMA && mxmm_gqa_attention_grouped8_supported(a)
          ? requested
          : MXMM_ATTN_SCALAR_SELECTED;
  if (actual_mode)
    *actual_mode = mode;
  const size_t dim = a->head_dim, capacity = a->capacity, valid = a->valid_keys;
  if (mode == MXMM_ATTN_SCALAR_SELECTED) {
    float *scores = workspace, *z = workspace + capacity;
    for (size_t h = 0; h < a->heads; h++) {
      size_t kv = h / (a->heads / a->kv_heads);
      float maximum = -INFINITY;
      for (size_t t = 0; t < valid; t++) {
        float sum = 0;
        const float *key = a->keys + (kv * capacity + t) * dim;
        for (size_t j = 0; j < dim; j++)
          sum += a->query[h * dim + j] * key[j];
        sum *= a->qk_scale;
        scores[t] = sum;
        if (sum > maximum)
          maximum = sum;
      }
      float denominator = 0;
      for (size_t t = 0; t < valid; t++) {
        scores[t] = expf(scores[t] - maximum);
        denominator += scores[t];
      }
      for (size_t t = 0; t < valid; t++)
        scores[t] = scores[t] / denominator;
      for (size_t j = 0; j < dim; j++)
        z[j] = 0;
      for (size_t t = 0; t < valid; t++) {
        float probability = scores[t];
        const float *row = a->values + (kv * capacity + t) * dim;
        for (size_t j = 0; j < dim; j++)
          z[j] += probability * row[j];
      }
      for (size_t j = 0; j < dim; j++)
        output[h * dim + j] = mxmm_gqa_attention_bf16_bits(z[j]);
    }
    return 0;
  }
  for (size_t kv = 0; kv < a->kv_heads; kv++) {
    size_t hbase = kv * 8;
    float qt[64][8], maxima[8];
    for (size_t j = 0; j < 64; j++)
      for (size_t h = 0; h < 8; h++)
        qt[j][h] = a->query[(hbase + h) * 64 + j];
    for (size_t h = 0; h < 8; h++)
      maxima[h] = -INFINITY;

    for (size_t tb = 0; tb < valid; tb += 2) {
      if (tb + 2 <= valid) {
        {
          float s0_0 = 0, s0_1 = 0, s0_2 = 0, s0_3 = 0, s0_4 = 0, s0_5 = 0, s0_6 = 0, s0_7 = 0,
                s1_0 = 0, s1_1 = 0, s1_2 = 0, s1_3 = 0, s1_4 = 0, s1_5 = 0, s1_6 = 0, s1_7 = 0;
          for (size_t j = 0; j < 64; j++) {
            float q0 = qt[j][0];
            float q1 = qt[j][1];
            float q2 = qt[j][2];
            float q3 = qt[j][3];
            float q4 = qt[j][4];
            float q5 = qt[j][5];
            float q6 = qt[j][6];
            float q7 = qt[j][7];
            float k0 = a->keys[(kv * capacity + tb + 0) * 64 + j];
            float k1 = a->keys[(kv * capacity + tb + 1) * 64 + j];
            s0_0 = fmaf(q0, k0, s0_0);
            s0_1 = fmaf(q1, k0, s0_1);
            s0_2 = fmaf(q2, k0, s0_2);
            s0_3 = fmaf(q3, k0, s0_3);
            s0_4 = fmaf(q4, k0, s0_4);
            s0_5 = fmaf(q5, k0, s0_5);
            s0_6 = fmaf(q6, k0, s0_6);
            s0_7 = fmaf(q7, k0, s0_7);
            s1_0 = fmaf(q0, k1, s1_0);
            s1_1 = fmaf(q1, k1, s1_1);
            s1_2 = fmaf(q2, k1, s1_2);
            s1_3 = fmaf(q3, k1, s1_3);
            s1_4 = fmaf(q4, k1, s1_4);
            s1_5 = fmaf(q5, k1, s1_5);
            s1_6 = fmaf(q6, k1, s1_6);
            s1_7 = fmaf(q7, k1, s1_7);
          }
          s0_0 *= a->qk_scale;
          workspace[0 * capacity + tb + 0] = s0_0;
          if (s0_0 > maxima[0])
            maxima[0] = s0_0;
          s0_1 *= a->qk_scale;
          workspace[1 * capacity + tb + 0] = s0_1;
          if (s0_1 > maxima[1])
            maxima[1] = s0_1;
          s0_2 *= a->qk_scale;
          workspace[2 * capacity + tb + 0] = s0_2;
          if (s0_2 > maxima[2])
            maxima[2] = s0_2;
          s0_3 *= a->qk_scale;
          workspace[3 * capacity + tb + 0] = s0_3;
          if (s0_3 > maxima[3])
            maxima[3] = s0_3;
          s0_4 *= a->qk_scale;
          workspace[4 * capacity + tb + 0] = s0_4;
          if (s0_4 > maxima[4])
            maxima[4] = s0_4;
          s0_5 *= a->qk_scale;
          workspace[5 * capacity + tb + 0] = s0_5;
          if (s0_5 > maxima[5])
            maxima[5] = s0_5;
          s0_6 *= a->qk_scale;
          workspace[6 * capacity + tb + 0] = s0_6;
          if (s0_6 > maxima[6])
            maxima[6] = s0_6;
          s0_7 *= a->qk_scale;
          workspace[7 * capacity + tb + 0] = s0_7;
          if (s0_7 > maxima[7])
            maxima[7] = s0_7;
          s1_0 *= a->qk_scale;
          workspace[0 * capacity + tb + 1] = s1_0;
          if (s1_0 > maxima[0])
            maxima[0] = s1_0;
          s1_1 *= a->qk_scale;
          workspace[1 * capacity + tb + 1] = s1_1;
          if (s1_1 > maxima[1])
            maxima[1] = s1_1;
          s1_2 *= a->qk_scale;
          workspace[2 * capacity + tb + 1] = s1_2;
          if (s1_2 > maxima[2])
            maxima[2] = s1_2;
          s1_3 *= a->qk_scale;
          workspace[3 * capacity + tb + 1] = s1_3;
          if (s1_3 > maxima[3])
            maxima[3] = s1_3;
          s1_4 *= a->qk_scale;
          workspace[4 * capacity + tb + 1] = s1_4;
          if (s1_4 > maxima[4])
            maxima[4] = s1_4;
          s1_5 *= a->qk_scale;
          workspace[5 * capacity + tb + 1] = s1_5;
          if (s1_5 > maxima[5])
            maxima[5] = s1_5;
          s1_6 *= a->qk_scale;
          workspace[6 * capacity + tb + 1] = s1_6;
          if (s1_6 > maxima[6])
            maxima[6] = s1_6;
          s1_7 *= a->qk_scale;
          workspace[7 * capacity + tb + 1] = s1_7;
          if (s1_7 > maxima[7])
            maxima[7] = s1_7;
        }
      } else {
        for (size_t t = tb; t < valid; t++) {
          for (size_t h = 0; h < 8; h++) {
            float sum = 0;
            for (size_t j = 0; j < 64; j++)
              sum = fmaf(qt[j][h], a->keys[(kv * capacity + t) * 64 + j], sum);
            sum *= a->qk_scale;
            workspace[h * capacity + t] = sum;
            if (sum > maxima[h])
              maxima[h] = sum;
          }
        }
      }
    }

    for (size_t h = 0; h < 8; h++) {
      float denominator = 0, *scores = workspace + h * capacity;

      for (size_t t = 0; t < valid; t++) {
        scores[t] = expf(scores[t] - maxima[h]);
        denominator += scores[t];
      }

      for (size_t t = 0; t < valid; t++)
        scores[t] = scores[t] / denominator;
    }

    float z[64][8] = {{0}};
    for (size_t tb = 0; tb < valid; tb += 32) {
      size_t end = tb + 32 < valid ? tb + 32 : valid;
      for (int j0 = 0; j0 < 64; j0 += 4) {
        float a0_0 = z[j0 + 0][0], a0_1 = z[j0 + 1][0], a0_2 = z[j0 + 2][0], a0_3 = z[j0 + 3][0],
              a1_0 = z[j0 + 0][1], a1_1 = z[j0 + 1][1], a1_2 = z[j0 + 2][1], a1_3 = z[j0 + 3][1],
              a2_0 = z[j0 + 0][2], a2_1 = z[j0 + 1][2], a2_2 = z[j0 + 2][2], a2_3 = z[j0 + 3][2],
              a3_0 = z[j0 + 0][3], a3_1 = z[j0 + 1][3], a3_2 = z[j0 + 2][3], a3_3 = z[j0 + 3][3];
        for (size_t t = tb; t < end; t++) {
          float p0 = workspace[0 * capacity + t];
          float p1 = workspace[1 * capacity + t];
          float p2 = workspace[2 * capacity + t];
          float p3 = workspace[3 * capacity + t];
          const float *row = a->values + ((size_t)kv * capacity + t) * 64 + j0;
          {
            float value = row[0];
            a0_0 = fmaf(p0, value, a0_0);
            a1_0 = fmaf(p1, value, a1_0);
            a2_0 = fmaf(p2, value, a2_0);
            a3_0 = fmaf(p3, value, a3_0);
          }
          {
            float value = row[1];
            a0_1 = fmaf(p0, value, a0_1);
            a1_1 = fmaf(p1, value, a1_1);
            a2_1 = fmaf(p2, value, a2_1);
            a3_1 = fmaf(p3, value, a3_1);
          }
          {
            float value = row[2];
            a0_2 = fmaf(p0, value, a0_2);
            a1_2 = fmaf(p1, value, a1_2);
            a2_2 = fmaf(p2, value, a2_2);
            a3_2 = fmaf(p3, value, a3_2);
          }
          {
            float value = row[3];
            a0_3 = fmaf(p0, value, a0_3);
            a1_3 = fmaf(p1, value, a1_3);
            a2_3 = fmaf(p2, value, a2_3);
            a3_3 = fmaf(p3, value, a3_3);
          }
        }
        z[j0 + 0][0] = a0_0;
        z[j0 + 1][0] = a0_1;
        z[j0 + 2][0] = a0_2;
        z[j0 + 3][0] = a0_3;
        z[j0 + 0][1] = a1_0;
        z[j0 + 1][1] = a1_1;
        z[j0 + 2][1] = a1_2;
        z[j0 + 3][1] = a1_3;
        z[j0 + 0][2] = a2_0;
        z[j0 + 1][2] = a2_1;
        z[j0 + 2][2] = a2_2;
        z[j0 + 3][2] = a2_3;
        z[j0 + 0][3] = a3_0;
        z[j0 + 1][3] = a3_1;
        z[j0 + 2][3] = a3_2;
        z[j0 + 3][3] = a3_3;
      }
      for (int j0 = 0; j0 < 64; j0 += 4) {
        float a0_0 = z[j0 + 0][4], a0_1 = z[j0 + 1][4], a0_2 = z[j0 + 2][4], a0_3 = z[j0 + 3][4],
              a1_0 = z[j0 + 0][5], a1_1 = z[j0 + 1][5], a1_2 = z[j0 + 2][5], a1_3 = z[j0 + 3][5],
              a2_0 = z[j0 + 0][6], a2_1 = z[j0 + 1][6], a2_2 = z[j0 + 2][6], a2_3 = z[j0 + 3][6],
              a3_0 = z[j0 + 0][7], a3_1 = z[j0 + 1][7], a3_2 = z[j0 + 2][7], a3_3 = z[j0 + 3][7];
        for (size_t t = tb; t < end; t++) {
          float p0 = workspace[4 * capacity + t];
          float p1 = workspace[5 * capacity + t];
          float p2 = workspace[6 * capacity + t];
          float p3 = workspace[7 * capacity + t];
          const float *row = a->values + ((size_t)kv * capacity + t) * 64 + j0;
          {
            float value = row[0];
            a0_0 = fmaf(p0, value, a0_0);
            a1_0 = fmaf(p1, value, a1_0);
            a2_0 = fmaf(p2, value, a2_0);
            a3_0 = fmaf(p3, value, a3_0);
          }
          {
            float value = row[1];
            a0_1 = fmaf(p0, value, a0_1);
            a1_1 = fmaf(p1, value, a1_1);
            a2_1 = fmaf(p2, value, a2_1);
            a3_1 = fmaf(p3, value, a3_1);
          }
          {
            float value = row[2];
            a0_2 = fmaf(p0, value, a0_2);
            a1_2 = fmaf(p1, value, a1_2);
            a2_2 = fmaf(p2, value, a2_2);
            a3_2 = fmaf(p3, value, a3_2);
          }
          {
            float value = row[3];
            a0_3 = fmaf(p0, value, a0_3);
            a1_3 = fmaf(p1, value, a1_3);
            a2_3 = fmaf(p2, value, a2_3);
            a3_3 = fmaf(p3, value, a3_3);
          }
        }
        z[j0 + 0][4] = a0_0;
        z[j0 + 1][4] = a0_1;
        z[j0 + 2][4] = a0_2;
        z[j0 + 3][4] = a0_3;
        z[j0 + 0][5] = a1_0;
        z[j0 + 1][5] = a1_1;
        z[j0 + 2][5] = a1_2;
        z[j0 + 3][5] = a1_3;
        z[j0 + 0][6] = a2_0;
        z[j0 + 1][6] = a2_1;
        z[j0 + 2][6] = a2_2;
        z[j0 + 3][6] = a2_3;
        z[j0 + 0][7] = a3_0;
        z[j0 + 1][7] = a3_1;
        z[j0 + 2][7] = a3_2;
        z[j0 + 3][7] = a3_3;
      }
    }
    for (int h = 0; h < 8; h++)
      for (int j = 0; j < 64; j++)
        output[(hbase + h) * 64 + j] = mxmm_gqa_attention_bf16_bits(z[j][h]);
  }
  return 0;
}

static inline int mx_tiled_gqa7_attention_bf16(const qwen_gqa_attention_args *a,
                                               qwen_attention_mode requested, uint16_t *output,
                                               float *workspace, size_t workspace_floats,
                                               qwen_attention_mode *actual_mode) {
  size_t need = qwen_gqa_attention_workspace_floats(a, requested);
  if (!need || !output || !workspace || !a->query || !a->keys || !a->values ||
      !isfinite(a->qk_scale) || a->qk_scale <= 0)
    return -1;
  if (workspace_floats < need)
    return -2;
  qwen_attention_mode mode =
      requested == QWEN_ATTN_GROUPED7_EXPLICIT_FMA && qwen_gqa_attention_grouped7_supported(a)
          ? requested
          : QWEN_ATTN_SCALAR;
  if (actual_mode)
    *actual_mode = mode;
  const size_t dim = a->head_dim, capacity = a->capacity, valid = a->valid_keys;
  if (mode == QWEN_ATTN_SCALAR) {
    float *scores = workspace, *z = workspace + capacity;
    for (size_t h = 0; h < a->heads; h++) {
      size_t kv = h / (a->heads / a->kv_heads);
      float maximum = -INFINITY;
      for (size_t t = 0; t < valid; t++) {
        float sum = 0;
        const float *key = a->keys + (kv * capacity + t) * dim;
        for (size_t j = 0; j < dim; j++)
          sum += a->query[h * dim + j] * key[j];
        sum *= a->qk_scale;
        scores[t] = sum;
        if (sum > maximum)
          maximum = sum;
      }
      float denominator = 0;
      for (size_t t = 0; t < valid; t++) {
        scores[t] = expf(scores[t] - maximum);
        denominator += scores[t];
      }
      for (size_t t = 0; t < valid; t++)
        scores[t] = scores[t] / denominator;
      for (size_t j = 0; j < dim; j++)
        z[j] = 0;
      for (size_t t = 0; t < valid; t++) {
        float probability = scores[t];
        const float *row = a->values + (kv * capacity + t) * dim;
        for (size_t j = 0; j < dim; j++)
          z[j] += probability * row[j];
      }
      for (size_t j = 0; j < dim; j++)
        output[h * dim + j] = qwen_gqa_attention_bf16_bits(z[j]);
    }
    return 0;
  }
  for (size_t kv = 0; kv < a->kv_heads; kv++) {
    size_t hbase = kv * 7;
    float qt[128][7], maxima[7];
    for (size_t j = 0; j < 128; j++)
      for (size_t h = 0; h < 7; h++)
        qt[j][h] = a->query[(hbase + h) * 128 + j];
    for (size_t h = 0; h < 7; h++)
      maxima[h] = -INFINITY;

    for (size_t tb = 0; tb < valid; tb += 2) {
      if (tb + 2 <= valid) {
        {
          float s0_0 = 0, s0_1 = 0, s0_2 = 0, s0_3 = 0, s0_4 = 0, s0_5 = 0, s0_6 = 0, s1_0 = 0,
                s1_1 = 0, s1_2 = 0, s1_3 = 0, s1_4 = 0, s1_5 = 0, s1_6 = 0;
          for (size_t j = 0; j < 128; j++) {
            float q0 = qt[j][0];
            float q1 = qt[j][1];
            float q2 = qt[j][2];
            float q3 = qt[j][3];
            float q4 = qt[j][4];
            float q5 = qt[j][5];
            float q6 = qt[j][6];
            float k0 = a->keys[(kv * capacity + tb + 0) * 128 + j];
            float k1 = a->keys[(kv * capacity + tb + 1) * 128 + j];
            s0_0 = fmaf(q0, k0, s0_0);
            s0_1 = fmaf(q1, k0, s0_1);
            s0_2 = fmaf(q2, k0, s0_2);
            s0_3 = fmaf(q3, k0, s0_3);
            s0_4 = fmaf(q4, k0, s0_4);
            s0_5 = fmaf(q5, k0, s0_5);
            s0_6 = fmaf(q6, k0, s0_6);
            s1_0 = fmaf(q0, k1, s1_0);
            s1_1 = fmaf(q1, k1, s1_1);
            s1_2 = fmaf(q2, k1, s1_2);
            s1_3 = fmaf(q3, k1, s1_3);
            s1_4 = fmaf(q4, k1, s1_4);
            s1_5 = fmaf(q5, k1, s1_5);
            s1_6 = fmaf(q6, k1, s1_6);
          }
          s0_0 *= a->qk_scale;
          workspace[0 * capacity + tb + 0] = s0_0;
          if (s0_0 > maxima[0])
            maxima[0] = s0_0;
          s0_1 *= a->qk_scale;
          workspace[1 * capacity + tb + 0] = s0_1;
          if (s0_1 > maxima[1])
            maxima[1] = s0_1;
          s0_2 *= a->qk_scale;
          workspace[2 * capacity + tb + 0] = s0_2;
          if (s0_2 > maxima[2])
            maxima[2] = s0_2;
          s0_3 *= a->qk_scale;
          workspace[3 * capacity + tb + 0] = s0_3;
          if (s0_3 > maxima[3])
            maxima[3] = s0_3;
          s0_4 *= a->qk_scale;
          workspace[4 * capacity + tb + 0] = s0_4;
          if (s0_4 > maxima[4])
            maxima[4] = s0_4;
          s0_5 *= a->qk_scale;
          workspace[5 * capacity + tb + 0] = s0_5;
          if (s0_5 > maxima[5])
            maxima[5] = s0_5;
          s0_6 *= a->qk_scale;
          workspace[6 * capacity + tb + 0] = s0_6;
          if (s0_6 > maxima[6])
            maxima[6] = s0_6;
          s1_0 *= a->qk_scale;
          workspace[0 * capacity + tb + 1] = s1_0;
          if (s1_0 > maxima[0])
            maxima[0] = s1_0;
          s1_1 *= a->qk_scale;
          workspace[1 * capacity + tb + 1] = s1_1;
          if (s1_1 > maxima[1])
            maxima[1] = s1_1;
          s1_2 *= a->qk_scale;
          workspace[2 * capacity + tb + 1] = s1_2;
          if (s1_2 > maxima[2])
            maxima[2] = s1_2;
          s1_3 *= a->qk_scale;
          workspace[3 * capacity + tb + 1] = s1_3;
          if (s1_3 > maxima[3])
            maxima[3] = s1_3;
          s1_4 *= a->qk_scale;
          workspace[4 * capacity + tb + 1] = s1_4;
          if (s1_4 > maxima[4])
            maxima[4] = s1_4;
          s1_5 *= a->qk_scale;
          workspace[5 * capacity + tb + 1] = s1_5;
          if (s1_5 > maxima[5])
            maxima[5] = s1_5;
          s1_6 *= a->qk_scale;
          workspace[6 * capacity + tb + 1] = s1_6;
          if (s1_6 > maxima[6])
            maxima[6] = s1_6;
        }
      } else {
        for (size_t t = tb; t < valid; t++) {
          for (size_t h = 0; h < 7; h++) {
            float sum = 0;
            for (size_t j = 0; j < 128; j++)
              sum = fmaf(qt[j][h], a->keys[(kv * capacity + t) * 128 + j], sum);
            sum *= a->qk_scale;
            workspace[h * capacity + t] = sum;
            if (sum > maxima[h])
              maxima[h] = sum;
          }
        }
      }
    }

    for (size_t h = 0; h < 7; h++) {
      float denominator = 0, *scores = workspace + h * capacity;

      for (size_t t = 0; t < valid; t++) {
        scores[t] = expf(scores[t] - maxima[h]);
        denominator += scores[t];
      }

      for (size_t t = 0; t < valid; t++)
        scores[t] = scores[t] / denominator;
    }

    float z[128][7] = {{0}};
    for (size_t tb = 0; tb < valid; tb += 16) {
      size_t end = tb + 16 < valid ? tb + 16 : valid;
      for (int j0 = 0; j0 < 128; j0 += 2) {
        float a0_0 = z[j0 + 0][0], a0_1 = z[j0 + 1][0], a1_0 = z[j0 + 0][1], a1_1 = z[j0 + 1][1],
              a2_0 = z[j0 + 0][2], a2_1 = z[j0 + 1][2], a3_0 = z[j0 + 0][3], a3_1 = z[j0 + 1][3],
              a4_0 = z[j0 + 0][4], a4_1 = z[j0 + 1][4], a5_0 = z[j0 + 0][5], a5_1 = z[j0 + 1][5],
              a6_0 = z[j0 + 0][6], a6_1 = z[j0 + 1][6];
        for (size_t t = tb; t < end; t++) {
          float p0 = workspace[0 * capacity + t];
          float p1 = workspace[1 * capacity + t];
          float p2 = workspace[2 * capacity + t];
          float p3 = workspace[3 * capacity + t];
          float p4 = workspace[4 * capacity + t];
          float p5 = workspace[5 * capacity + t];
          float p6 = workspace[6 * capacity + t];
          const float *row = a->values + ((size_t)kv * capacity + t) * 128 + j0;
          {
            float value = row[0];
            a0_0 = fmaf(p0, value, a0_0);
            a1_0 = fmaf(p1, value, a1_0);
            a2_0 = fmaf(p2, value, a2_0);
            a3_0 = fmaf(p3, value, a3_0);
            a4_0 = fmaf(p4, value, a4_0);
            a5_0 = fmaf(p5, value, a5_0);
            a6_0 = fmaf(p6, value, a6_0);
          }
          {
            float value = row[1];
            a0_1 = fmaf(p0, value, a0_1);
            a1_1 = fmaf(p1, value, a1_1);
            a2_1 = fmaf(p2, value, a2_1);
            a3_1 = fmaf(p3, value, a3_1);
            a4_1 = fmaf(p4, value, a4_1);
            a5_1 = fmaf(p5, value, a5_1);
            a6_1 = fmaf(p6, value, a6_1);
          }
        }
        z[j0 + 0][0] = a0_0;
        z[j0 + 1][0] = a0_1;
        z[j0 + 0][1] = a1_0;
        z[j0 + 1][1] = a1_1;
        z[j0 + 0][2] = a2_0;
        z[j0 + 1][2] = a2_1;
        z[j0 + 0][3] = a3_0;
        z[j0 + 1][3] = a3_1;
        z[j0 + 0][4] = a4_0;
        z[j0 + 1][4] = a4_1;
        z[j0 + 0][5] = a5_0;
        z[j0 + 1][5] = a5_1;
        z[j0 + 0][6] = a6_0;
        z[j0 + 1][6] = a6_1;
      }
    }
    for (int h = 0; h < 7; h++)
      for (int j = 0; j < 128; j++)
        output[(hbase + h) * 128 + j] = qwen_gqa_attention_bf16_bits(z[j][h]);
  }
  return 0;
}
#endif
