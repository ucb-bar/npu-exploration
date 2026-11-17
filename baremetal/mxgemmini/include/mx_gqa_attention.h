#ifndef MX_GQA_ATTENTION_H
#define MX_GQA_ATTENTION_H
/* CPU FP32 GQA attention, BF16 output. Compile with -fno-fast-math and
 * -ffp-contract=off. Only the explicit grouped-mode fmaf calls fuse operations.
 * This header does not quantize, append, or otherwise modify the KV cache.
 * Q: [heads][head_dim]; K/V: [kv_heads][capacity][head_dim]. It reads only
 * [0,valid_keys), so poisoned future cache slots remain untouched.
 */
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#ifdef __FAST_MATH__
#error "mx_gqa_attention.h requires -fno-fast-math and -ffp-contract=off"
#endif

typedef enum mxmm_attention_mode {
  MXMM_ATTN_SCALAR_SELECTED = 0,
  MXMM_ATTN_GROUPED8_EXPLICIT_FMA = 1
} mxmm_attention_mode;

typedef struct mxmm_gqa_attention_args {
  size_t heads, kv_heads, head_dim, capacity, valid_keys;
  float qk_scale;
  const float *query, *keys, *values;
} mxmm_gqa_attention_args;

static inline int mxmm_gqa_attention_geometry_valid(const mxmm_gqa_attention_args *a) {
  if (!a || !a->heads || !a->kv_heads || !a->head_dim || !a->capacity || !a->valid_keys ||
      a->valid_keys > a->capacity || a->heads % a->kv_heads)
    return 0;
  if (a->head_dim > SIZE_MAX / sizeof(float) || a->heads > SIZE_MAX / sizeof(float) / a->head_dim ||
      a->capacity > SIZE_MAX / sizeof(float) / a->head_dim ||
      a->kv_heads > SIZE_MAX / sizeof(float) / a->head_dim / a->capacity)
    return 0;
  return 1;
}
static inline int mxmm_gqa_attention_grouped8_supported(const mxmm_gqa_attention_args *a) {
  return mxmm_gqa_attention_geometry_valid(a) && a->heads / a->kv_heads == 8 && a->head_dim == 64;
}
static inline size_t mxmm_gqa_attention_workspace_floats(const mxmm_gqa_attention_args *a,
                                                         mxmm_attention_mode requested) {
  if (!mxmm_gqa_attention_geometry_valid(a) ||
      (requested != MXMM_ATTN_SCALAR_SELECTED && requested != MXMM_ATTN_GROUPED8_EXPLICIT_FMA))
    return 0;
  if (requested == MXMM_ATTN_GROUPED8_EXPLICIT_FMA && mxmm_gqa_attention_grouped8_supported(a)) {
    if (a->capacity > SIZE_MAX / sizeof(float) / 8)
      return 0;
    return 8 * a->capacity;
  }
  if (a->capacity > SIZE_MAX / sizeof(float) - a->head_dim)
    return 0;
  return a->capacity + a->head_dim;
}
static inline uint16_t mxmm_gqa_attention_bf16_bits(float x) {
  union {
    float f;
    uint32_t u;
  } bits;
  bits.f = x;
  if (((bits.u >> 23) & 255) == 255)
    return (uint16_t)(bits.u >> 16);
  return (uint16_t)((bits.u + 0x7fff + ((bits.u >> 16) & 1)) >> 16);
}

/* Return 0 on success, -1 for invalid arguments, -2 for insufficient workspace.
 * Unsupported grouped geometry falls back to scalar_selected. actual_mode may
 * be NULL; otherwise it reports the mode that produced the output. Input,
 * output and workspace buffers must not overlap. Workspace size is in floats.
 */
static inline int mxmm_gqa_attention_bf16(const mxmm_gqa_attention_args *a,
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
    for (size_t t = 0; t < valid; t++) {
      float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
      const float *row = a->keys + (kv * capacity + t) * 64;
      for (size_t j = 0; j < 64; j++) {
        float key = row[j];
        const float *qj = qt[j];
        s0 = fmaf(qj[0], key, s0);
        s1 = fmaf(qj[1], key, s1);
        s2 = fmaf(qj[2], key, s2);
        s3 = fmaf(qj[3], key, s3);
        s4 = fmaf(qj[4], key, s4);
        s5 = fmaf(qj[5], key, s5);
        s6 = fmaf(qj[6], key, s6);
        s7 = fmaf(qj[7], key, s7);
      }
#define MXMM_GQA_SCORE(H)                                                                          \
  do {                                                                                             \
    s##H *= a->qk_scale;                                                                           \
    workspace[(H) * capacity + t] = s##H;                                                          \
    if (s##H > maxima[H])                                                                          \
      maxima[H] = s##H;                                                                            \
  } while (0)
      MXMM_GQA_SCORE(0);
      MXMM_GQA_SCORE(1);
      MXMM_GQA_SCORE(2);
      MXMM_GQA_SCORE(3);
      MXMM_GQA_SCORE(4);
      MXMM_GQA_SCORE(5);
      MXMM_GQA_SCORE(6);
      MXMM_GQA_SCORE(7);
#undef MXMM_GQA_SCORE
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
    for (size_t t = 0; t < valid; t++) {
      const float p0 = workspace[t], p1 = workspace[capacity + t], p2 = workspace[2 * capacity + t],
                  p3 = workspace[3 * capacity + t], p4 = workspace[4 * capacity + t],
                  p5 = workspace[5 * capacity + t], p6 = workspace[6 * capacity + t],
                  p7 = workspace[7 * capacity + t];
      const float *row = a->values + (kv * capacity + t) * 64;
      for (size_t j = 0; j < 64; j++) {
        float value = row[j], *zj = z[j];
        zj[0] = fmaf(p0, value, zj[0]);
        zj[1] = fmaf(p1, value, zj[1]);
        zj[2] = fmaf(p2, value, zj[2]);
        zj[3] = fmaf(p3, value, zj[3]);
        zj[4] = fmaf(p4, value, zj[4]);
        zj[5] = fmaf(p5, value, zj[5]);
        zj[6] = fmaf(p6, value, zj[6]);
        zj[7] = fmaf(p7, value, zj[7]);
      }
    }
    for (size_t h = 0; h < 8; h++)
      for (size_t j = 0; j < 64; j++)
        output[(hbase + h) * 64 + j] = mxmm_gqa_attention_bf16_bits(z[j][h]);
  }
  return 0;
}
#endif
