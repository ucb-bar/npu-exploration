#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "mx_llm_attention.h"
#ifdef BAREMETAL
int *__errno(void) {
  static int e;
  return &e;
}
#endif
#define C 37
#define MAXH 16
static float q[MAXH * 128], k[MAXH * C * 128], v[MAXH * C * 128], w[8 * C + 128 + 32],
    rw[8 * C + 128 + 32];
static uint16_t out[MAXH * 128 + 32], native[MAXH * 128 + 32];
static uint64_t hash = 1469598103934665603ULL;
static int checks, failed;
static void check(int ok) {
  checks++;
  failed += !ok;
}
static float sample(size_t i, int salt) {
  return ((int)((i * 17 + salt * 13) % 127) - 63) * 0x1p-8f;
}
static float unbf(uint16_t x) {
  uint32_t u = (uint32_t)x << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static void one(int heads, int kv, int hd, int physical, int stride, int valid,
                mx_llm_attention_mode requested) {
  int words = heads * hd;
  uint32_t poison = 0x7fc00000;
  for (int i = 0; i < words; i++)
    q[i] = sample(i, 1);
  for (int h = 0; h < kv; h++)
    for (int t = 0; t < physical; t++)
      for (int j = 0; j < hd; j++) {
        size_t i = ((size_t)h * physical + t) * hd + j;
        if (t < valid) {
          k[i] = sample(i, 2);
          v[i] = sample(i, 3);
        } else {
          memcpy(k + i, &poison, 4);
          memcpy(v + i, &poison, 4);
        }
      }
  for (int i = 0; i < MAXH * 128 + 32; i++)
    out[i] = native[i] = 0x5aa5;
  for (int i = 0; i < 8 * C + 128 + 32; i++)
    w[i] = rw[i] = 0x1.55p10f;
  mx_llm_attention_args a = {(size_t)heads,
                             (size_t)kv,
                             (size_t)hd,
                             34,
                             (size_t)physical,
                             (size_t)stride,
                             (size_t)valid,
                             hd == 64 ? .125f : 0x1.6a09e6p-4f,
                             q,
                             k,
                             v};
  size_t need = mx_llm_attention_workspace_floats(&a, requested);
  mx_llm_attention_mode actual = (mx_llm_attention_mode)99;
  check(need > 0 && need <= 8 * C + 128);
  check(mx_llm_attention_bf16(&a, requested, out, w, need, &actual) == 0);
  check(actual == mx_llm_attention_resolve(&a, requested));
  if (actual == MX_LLM_ATTN_GQA8_FMA || actual == MX_LLM_ATTN_GQA8_TILED_FMA) {
    mxmm_gqa_attention_args x = {
        a.heads, a.kv_heads, a.head_dim, a.physical_capacity, a.valid_keys, a.qk_scale, q, k, v};
    check(mxmm_gqa_attention_bf16(&x, MXMM_ATTN_GROUPED8_EXPLICIT_FMA, native, rw, 8 * C + 128,
                                  NULL) == 0);
  } else if (actual == MX_LLM_ATTN_GQA7_FMA || actual == MX_LLM_ATTN_GQA7_TILED_FMA) {
    qwen_gqa_attention_args x = {
        a.heads, a.kv_heads, a.head_dim, a.physical_capacity, a.valid_keys, a.qk_scale, q, k, v};
    check(qwen_gqa_attention_bf16(&x, QWEN_ATTN_GROUPED7_EXPLICIT_FMA, native, rw, 8 * C + 128,
                                  NULL) == 0);
  } else if (actual == MX_LLM_ATTN_MHA4_FMA || actual == MX_LLM_ATTN_MHA_TILED_FMA)
    check(llama7b_mha4_strided_attention(q, k, v, heads, kv, hd, 34, physical, stride, valid,
                                         a.qk_scale, native, rw) == 1);
  double error = 0, norm = 0;
  for (int h = 0; h < heads; h++) {
    double scores[C], maximum = -INFINITY, den = 0, z[128] = {0};
    int kh = h / (heads / kv);
    for (int t = 0; t < valid; t++) {
      double s = 0;
      for (int j = 0; j < hd; j++)
        s += (double)q[h * hd + j] * k[((size_t)kh * physical + t) * hd + j];
      s *= a.qk_scale;
      scores[t] = s;
      if (s > maximum)
        maximum = s;
    }
    for (int t = 0; t < valid; t++) {
      scores[t] = exp(scores[t] - maximum);
      den += scores[t];
    }
    for (int t = 0; t < valid; t++)
      for (int j = 0; j < hd; j++)
        z[j] += (scores[t] / den) * v[((size_t)kh * physical + t) * hd + j];
    for (int j = 0; j < hd; j++) {
      double x = unbf(out[h * hd + j]);
      check(isfinite(x));
      error += (x - z[j]) * (x - z[j]);
      norm += z[j] * z[j];
      if (actual)
        check(out[h * hd + j] == native[h * hd + j]);
      hash = (hash ^ out[h * hd + j]) * 1099511628211ULL;
    }
  }
  check(norm ? error <= norm * 1e-4 : error == 0);
  for (int i = words; i < MAXH * 128 + 32; i++)
    check(out[i] == 0x5aa5);
  for (size_t i = need; i < 8 * C + 128 + 32; i++)
    check(w[i] == 0x1.55p10f);
  for (int h = 0; h < kv; h++)
    for (int t = valid; t < physical; t++)
      for (int j = 0; j < hd; j++) {
        size_t i = ((size_t)h * physical + t) * hd + j;
        uint32_t x, y;
        memcpy(&x, k + i, 4);
        memcpy(&y, v + i, 4);
        check(x == poison && y == poison);
      }
  /* Invalid calls leave outputs and the reported mode unchanged. */
  uint16_t before = out[0];
  actual = (mx_llm_attention_mode)99;
  check(mx_llm_attention_bf16(&a, (mx_llm_attention_mode)99, out, w, need, &actual) == -1);
  check(out[0] == before && actual == 99);
  check(mx_llm_attention_bf16(&a, requested, out, w, need - 1, &actual) == -2);
  check(actual == 99);
  check(mx_llm_attention_bf16(NULL, requested, out, w, need, &actual) == -1);
}

#define LP_H 4
#define LP_HD 128
#define LP_P 2057
#define LP_S 2056
static float lp_k[LP_H * LP_P * LP_HD + 16], lp_v[LP_H * LP_P * LP_HD + 16], lp_q[LP_H * LP_HD],
    lp_w[4 * LP_S + 16], lp_rw[4 * LP_S + 16];
static uint16_t lp_o[LP_H * LP_HD + 32], lp_ref[LP_H * LP_HD + 32];
static void real_pitch_case(void) {
  for (int h = 0; h < LP_H; h++)
    for (int j = 0; j < LP_HD; j++)
      lp_q[h * LP_HD + j] = (float)((h + j) % 11 - 5) * 0.0625f;
  for (int h = 0; h < LP_H; h++)
    for (int t = 0; t < LP_P; t++)
      for (int j = 0; j < LP_HD; j++) {
        size_t i = ((size_t)h * LP_P + t) * LP_HD + j;
        lp_k[i] = t < 17 ? (float)((h * 3 + t + j) % 13 - 6) * 0.03125f : NAN;
        lp_v[i] = t < 17 ? (float)((h * 7 + t * 3 + j) % 17 - 8) * 0.03125f : NAN;
      }
  for (int i = 0; i < LP_H * LP_HD + 32; i++)
    lp_o[i] = lp_ref[i] = 0x5a5a;
  for (int i = 0; i < 4 * LP_S + 16; i++)
    lp_w[i] = lp_rw[i] = 123.5f;
  mx_llm_attention_args a = {LP_H, LP_H,           LP_HD, LP_S, LP_P, LP_S,
                             17,   0x1.6a09e6p-4f, lp_q,  lp_k, lp_v};
  mx_llm_attention_mode actual = MX_LLM_ATTN_SCALAR;
  check(mx_llm_attention_bf16(&a, MX_LLM_ATTN_MHA_TILED_FMA, lp_o, lp_w, 4 * LP_S, &actual) == 0);
  check(actual == MX_LLM_ATTN_MHA_TILED_FMA);
  check(llama7b_mha4_strided_attention(lp_q, lp_k, lp_v, LP_H, LP_H, LP_HD, LP_S, LP_P, LP_S, 17,
                                       a.qk_scale, lp_ref, lp_rw) == 1);
  for (int i = 0; i < LP_H * LP_HD; i++) {
    check(lp_o[i] == lp_ref[i]);
    hash = (hash ^ lp_o[i]) * 1099511628211ULL;
  }
  for (int i = LP_H * LP_HD; i < LP_H * LP_HD + 32; i++)
    check(lp_o[i] == 0x5a5a && lp_ref[i] == 0x5a5a);
  for (int i = 4 * LP_S; i < 4 * LP_S + 16; i++)
    check(lp_w[i] == 123.5f && lp_rw[i] == 123.5f);
  for (int h = 0; h < LP_H; h++)
    for (int t = 17; t < LP_P; t++)
      for (int j = 0; j < LP_HD; j++) {
        size_t i = ((size_t)h * LP_P + t) * LP_HD + j;
        check(isnan(lp_k[i]) && isnan(lp_v[i]));
      }
}
int main(void) {
  real_pitch_case();
  int valids[] = {1, 2, 3, 17, 34};
  for (int t = 0; t < 5; t++) {
    one(16, 2, 64, 34, 34, valids[t], MX_LLM_ATTN_GQA8_FMA);
    one(14, 2, 128, 34, 34, valids[t], MX_LLM_ATTN_GQA7_FMA);
    one(8, 8, 128, 35, 34, valids[t], MX_LLM_ATTN_MHA4_FMA);
    one(8, 8, 128, 35, 34, valids[t], MX_LLM_ATTN_MHA_TILED_FMA);
    one(6, 2, 32, 35, 34, valids[t], MX_LLM_ATTN_MHA_TILED_FMA);
    one(6, 2, 32, 35, 34, valids[t], MX_LLM_ATTN_GQA8_FMA);
    one(16, 2, 64, 34, 34, valids[t], MX_LLM_ATTN_GQA8_TILED_FMA);
    one(6, 2, 32, 35, 34, valids[t], MX_LLM_ATTN_GQA8_TILED_FMA);
    one(14, 2, 128, 34, 34, valids[t], MX_LLM_ATTN_GQA7_TILED_FMA);
    one(6, 2, 32, 35, 34, valids[t], MX_LLM_ATTN_GQA7_TILED_FMA);
  }
  printf("MX_LLM_TEST %s checks=%d checksum=%llu\n", failed ? "FAIL" : "PASS", checks,
         (unsigned long long)hash);
  return failed != 0;
}
