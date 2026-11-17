#ifndef MX_LLM_ATTENTION_H
#define MX_LLM_ATTENTION_H
/* Strict FP32 CPU attention, with qualified geometry specializations.
 * Compile with -fno-fast-math -ffp-contract=off. No quantization or KV append.
 * Query [heads][head_dim], K/V [kv_heads][physical_capacity][head_dim].
 * Only valid_keys are read. All input/output/workspace spans must be disjoint.
 * Specialized loops are unchanged; geometry dispatch happens outside them. */
#include <limits.h>
#include "mx_gqa_attention.h"
#include "qwen_gqa_attention.h"
#include "mx_mha_attention.h"
#include "mx_tiled_attention.h"
#include "mx_tiled_mha_attention.h"
typedef enum mx_llm_attention_mode {
  MX_LLM_ATTN_SCALAR = 0,
  MX_LLM_ATTN_GQA8_FMA = 1,
  MX_LLM_ATTN_GQA7_FMA = 2,
  MX_LLM_ATTN_MHA4_FMA = 3,
  /* QK time2, PV time32 with four heads and four output lanes. */
  MX_LLM_ATTN_GQA8_TILED_FMA = 4,
  /* QK time2, PV time16 with seven heads and two output lanes. */
  MX_LLM_ATTN_GQA7_TILED_FMA = 5,
  MX_LLM_ATTN_MHA_TILED_FMA = 6
} mx_llm_attention_mode;
typedef struct mx_llm_attention_args {
  size_t heads, kv_heads, head_dim, logical_capacity, physical_capacity, score_stride, valid_keys;
  float qk_scale;
  const float *query, *keys, *values;
} mx_llm_attention_args;
static inline int mx_llm_attention_geometry_valid(const mx_llm_attention_args *a) {
  if (!a || !a->heads || !a->kv_heads || !a->head_dim || !a->logical_capacity || !a->valid_keys ||
      a->heads % a->kv_heads || a->valid_keys > a->logical_capacity ||
      a->physical_capacity < a->logical_capacity || a->score_stride < a->logical_capacity)
    return 0;
  return a->head_dim <= SIZE_MAX / sizeof(float) &&
         a->heads <= SIZE_MAX / sizeof(float) / a->head_dim &&
         a->physical_capacity <= SIZE_MAX / sizeof(float) / a->head_dim &&
         a->kv_heads <= SIZE_MAX / sizeof(float) / a->head_dim / a->physical_capacity;
}
static inline int mx_llm_attention_mode_valid(mx_llm_attention_mode mode) {
  return mode >= MX_LLM_ATTN_SCALAR && mode <= MX_LLM_ATTN_MHA_TILED_FMA;
}
static inline mx_llm_attention_mode mx_llm_attention_resolve(const mx_llm_attention_args *a,
                                                             mx_llm_attention_mode requested) {
  if (!mx_llm_attention_geometry_valid(a) || !mx_llm_attention_mode_valid(requested))
    return MX_LLM_ATTN_SCALAR;
  if ((requested == MX_LLM_ATTN_GQA8_FMA || requested == MX_LLM_ATTN_GQA8_TILED_FMA) &&
      a->heads / a->kv_heads == 8 && a->head_dim == 64 && a->score_stride == a->physical_capacity)
    return requested;
  if ((requested == MX_LLM_ATTN_GQA7_FMA || requested == MX_LLM_ATTN_GQA7_TILED_FMA) &&
      a->heads / a->kv_heads == 7 && a->head_dim == 128 && a->score_stride == a->physical_capacity)
    return requested;
  if (requested == MX_LLM_ATTN_MHA4_FMA && a->heads >= 4 && a->heads % 4 == 0 &&
      a->heads == a->kv_heads && a->head_dim == 128 && a->heads <= INT_MAX &&
      a->logical_capacity <= INT_MAX && a->physical_capacity <= INT_MAX &&
      a->score_stride <= INT_MAX)
    return requested;
  if (requested == MX_LLM_ATTN_MHA_TILED_FMA && a->heads >= 4 && a->heads % 4 == 0 &&
      a->heads == a->kv_heads && a->head_dim == 128 && a->heads <= INT_MAX &&
      a->logical_capacity <= INT_MAX - 8 && a->physical_capacity <= INT_MAX &&
      a->score_stride <= INT_MAX - 8)
    return requested;
  return MX_LLM_ATTN_SCALAR;
}
static inline size_t mx_llm_attention_workspace_floats(const mx_llm_attention_args *a,
                                                       mx_llm_attention_mode requested) {
  if (!mx_llm_attention_geometry_valid(a) || !mx_llm_attention_mode_valid(requested))
    return 0;
  mx_llm_attention_mode mode = mx_llm_attention_resolve(a, requested);
  size_t factor = (mode == MX_LLM_ATTN_GQA8_FMA || mode == MX_LLM_ATTN_GQA8_TILED_FMA)   ? 8
                  : (mode == MX_LLM_ATTN_GQA7_FMA || mode == MX_LLM_ATTN_GQA7_TILED_FMA) ? 7
                  : (mode == MX_LLM_ATTN_MHA4_FMA || mode == MX_LLM_ATTN_MHA_TILED_FMA)  ? 4
                                                                                         : 1;
  if (mode == MX_LLM_ATTN_SCALAR) {
    if (a->score_stride > SIZE_MAX / sizeof(float) - a->head_dim)
      return 0;
    return a->score_stride + a->head_dim;
  }
  if (a->score_stride > SIZE_MAX / sizeof(float) / factor)
    return 0;
  return a->score_stride * factor;
}
/* Return0 success,-1 invalid args/mode,-2 insufficient workspace. Unsupported
 * requested geometry uses scalar and reports that mode through actual_mode. */
static inline int mx_llm_attention_bf16(const mx_llm_attention_args *a,
                                        mx_llm_attention_mode requested, uint16_t *out,
                                        float *workspace, size_t workspace_floats,
                                        mx_llm_attention_mode *actual_mode) {
  size_t need = mx_llm_attention_workspace_floats(a, requested);
  if (!need || !out || !workspace || !a->query || !a->keys || !a->values ||
      !isfinite(a->qk_scale) || a->qk_scale <= 0)
    return -1;
  if (workspace_floats < need)
    return -2;
  mx_llm_attention_mode mode = mx_llm_attention_resolve(a, requested);
  if (mode == MX_LLM_ATTN_GQA8_TILED_FMA) {
    mxmm_gqa_attention_args v = {a->heads,      a->kv_heads, a->head_dim, a->physical_capacity,
                                 a->valid_keys, a->qk_scale, a->query,    a->keys,
                                 a->values};
    int rc = mx_tiled_gqa8_attention_bf16(&v, MXMM_ATTN_GROUPED8_EXPLICIT_FMA, out, workspace,
                                          workspace_floats, NULL);
    if (!rc && actual_mode)
      *actual_mode = mode;
    return rc;
  }
  if (mode == MX_LLM_ATTN_GQA7_TILED_FMA) {
    qwen_gqa_attention_args v = {a->heads,      a->kv_heads, a->head_dim, a->physical_capacity,
                                 a->valid_keys, a->qk_scale, a->query,    a->keys,
                                 a->values};
    int rc = mx_tiled_gqa7_attention_bf16(&v, QWEN_ATTN_GROUPED7_EXPLICIT_FMA, out, workspace,
                                          workspace_floats, NULL);
    if (!rc && actual_mode)
      *actual_mode = mode;
    return rc;
  }
  if (mode == MX_LLM_ATTN_GQA8_FMA) {
    mxmm_gqa_attention_args v = {a->heads,      a->kv_heads, a->head_dim, a->physical_capacity,
                                 a->valid_keys, a->qk_scale, a->query,    a->keys,
                                 a->values};
    int rc = mxmm_gqa_attention_bf16(&v, MXMM_ATTN_GROUPED8_EXPLICIT_FMA, out, workspace,
                                     workspace_floats, NULL);
    if (!rc && actual_mode)
      *actual_mode = mode;
    return rc;
  }
  if (mode == MX_LLM_ATTN_GQA7_FMA) {
    qwen_gqa_attention_args v = {a->heads,      a->kv_heads, a->head_dim, a->physical_capacity,
                                 a->valid_keys, a->qk_scale, a->query,    a->keys,
                                 a->values};
    int rc = qwen_gqa_attention_bf16(&v, QWEN_ATTN_GROUPED7_EXPLICIT_FMA, out, workspace,
                                     workspace_floats, NULL);
    if (!rc && actual_mode)
      *actual_mode = mode;
    return rc;
  }
  if (mode == MX_LLM_ATTN_MHA_TILED_FMA) {
    int ok =
        mx_mha_qkt8_pvh1j16(a->query, a->keys, a->values, (int)a->heads, (int)a->kv_heads, 128,
                            (int)a->logical_capacity, (int)a->physical_capacity,
                            (int)a->score_stride, (int)a->valid_keys, a->qk_scale, out, workspace);
    if (ok && actual_mode)
      *actual_mode = mode;
    return ok ? 0 : -1;
  }
  if (mode == MX_LLM_ATTN_MHA4_FMA) {
    int ok = llama7b_mha4_strided_attention(a->query, a->keys, a->values, (int)a->heads,
                                            (int)a->kv_heads, 128, (int)a->logical_capacity,
                                            (int)a->physical_capacity, (int)a->score_stride,
                                            (int)a->valid_keys, a->qk_scale, out, workspace);
    if (ok && actual_mode)
      *actual_mode = mode;
    return ok ? 0 : -1;
  }
  float *scores = workspace, *z = workspace + a->score_stride;
  for (size_t h = 0; h < a->heads; h++) {
    size_t kv = h / (a->heads / a->kv_heads);
    float maximum = -INFINITY;
    for (size_t t = 0; t < a->valid_keys; t++) {
      float sum = 0;
      const float *row = a->keys + (kv * a->physical_capacity + t) * a->head_dim;
      for (size_t j = 0; j < a->head_dim; j++)
        sum += a->query[h * a->head_dim + j] * row[j];
      sum *= a->qk_scale;
      scores[t] = sum;
      if (sum > maximum)
        maximum = sum;
    }
    float denominator = 0;
    for (size_t t = 0; t < a->valid_keys; t++) {
      scores[t] = expf(scores[t] - maximum);
      denominator += scores[t];
    }
    for (size_t t = 0; t < a->valid_keys; t++)
      scores[t] /= denominator;
    for (size_t j = 0; j < a->head_dim; j++)
      z[j] = 0;
    for (size_t t = 0; t < a->valid_keys; t++) {
      float p = scores[t];
      const float *row = a->values + (kv * a->physical_capacity + t) * a->head_dim;
      for (size_t j = 0; j < a->head_dim; j++)
        z[j] += p * row[j];
    }
    for (size_t j = 0; j < a->head_dim; j++)
      out[h * a->head_dim + j] = mxmm_gqa_attention_bf16_bits(z[j]);
  }
  if (actual_mode)
    *actual_mode = mode;
  return 0;
}
#endif
