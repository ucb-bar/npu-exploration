#ifndef MX_MHA_ATTENTION_H
#define MX_MHA_ATTENTION_H
/* Four independent MHA heads; every lane reads its own K/V cache.
 * QK dimension order and PV time order remain ascending for each output.
 * No GQA reuse; workspace >=4*score_stride float words; physical head pitch separate from logical capacity; future keys untouched. */
#ifdef __FAST_MATH__
#error "Llama MHA attention requires strict FP and ffp-contract=off"
#endif
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#if defined(LLAMA_MHA4_EXPLICIT_FMA) && LLAMA_MHA4_EXPLICIT_FMA != 1
#error "The qualified MHA4 mode requires explicit FMA"
#endif
#ifndef LLAMA_MHA4_EXPLICIT_FMA
#define LLAMA_MHA4_EXPLICIT_FMA 1
#endif
static inline float llama_strided_madd(float a, float b, float c) {
#if LLAMA_MHA4_EXPLICIT_FMA
  return fmaf(a, b, c);
#else
  return c + a * b;
#endif
}
static inline uint16_t llama_strided_bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  return (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}
static int llama7b_mha4_strided_attention(const float *q, const float *keys, const float *values,
                                          int heads, int kv_heads, int hd, int logical_capacity,
                                          int physical_capacity, int score_stride, int valid,
                                          float scale, uint16_t *out, float *workspace) {
  if (!q || !keys || !values || !out || !workspace || heads < 4 || heads % 4 || kv_heads != heads ||
      hd != 128 || valid < 1 || valid > logical_capacity || physical_capacity < logical_capacity ||
      score_stride < logical_capacity)
    return 0;
  for (int h = 0; h < heads; h += 4) {
    float *s0 = workspace, *s1 = s0 + score_stride, *s2 = s1 + score_stride,
          *s3 = s2 + score_stride;
    const float *q0 = q + h * hd, *q1 = q0 + hd, *q2 = q1 + hd, *q3 = q2 + hd;
    float max0 = -INFINITY, max1 = -INFINITY, max2 = -INFINITY, max3 = -INFINITY;
    for (int t = 0; t < valid; t++) {
      const float *k0 = keys + ((size_t)h * physical_capacity + t) * hd,
                  *k1 = k0 + (size_t)physical_capacity * hd,
                  *k2 = k1 + (size_t)physical_capacity * hd,
                  *k3 = k2 + (size_t)physical_capacity * hd;
      float z0 = 0, z1 = 0, z2 = 0, z3 = 0;
      for (int j = 0; j < 128; j++) {
        z0 = llama_strided_madd(q0[j], k0[j], z0);
        z1 = llama_strided_madd(q1[j], k1[j], z1);
        z2 = llama_strided_madd(q2[j], k2[j], z2);
        z3 = llama_strided_madd(q3[j], k3[j], z3);
      }
      z0 *= scale;
      z1 *= scale;
      z2 *= scale;
      z3 *= scale;
      s0[t] = z0;
      s1[t] = z1;
      s2[t] = z2;
      s3[t] = z3;
      if (z0 > max0)
        max0 = z0;
      if (z1 > max1)
        max1 = z1;
      if (z2 > max2)
        max2 = z2;
      if (z3 > max3)
        max3 = z3;
    }
    float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
    for (int t = 0; t < valid; t++) {
      s0[t] = expf(s0[t] - max0);
      d0 += s0[t];
      s1[t] = expf(s1[t] - max1);
      d1 += s1[t];
      s2[t] = expf(s2[t] - max2);
      d2 += s2[t];
      s3[t] = expf(s3[t] - max3);
      d3 += s3[t];
    }
    for (int t = 0; t < valid; t++) {
      s0[t] /= d0;
      s1[t] /= d1;
      s2[t] /= d2;
      s3[t] /= d3;
    }
    float z0[128] = {0}, z1[128] = {0}, z2[128] = {0}, z3[128] = {0};
    for (int t = 0; t < valid; t++) {
      const float *v0 = values + ((size_t)h * physical_capacity + t) * hd,
                  *v1 = v0 + (size_t)physical_capacity * hd,
                  *v2 = v1 + (size_t)physical_capacity * hd,
                  *v3 = v2 + (size_t)physical_capacity * hd;
      float p0 = s0[t], p1 = s1[t], p2 = s2[t], p3 = s3[t];
      for (int j = 0; j < 128; j++) {
        z0[j] = llama_strided_madd(p0, v0[j], z0[j]);
        z1[j] = llama_strided_madd(p1, v1[j], z1[j]);
        z2[j] = llama_strided_madd(p2, v2[j], z2[j]);
        z3[j] = llama_strided_madd(p3, v3[j], z3[j]);
      }
    }
    for (int j = 0; j < 128; j++) {
      out[h * hd + j] = llama_strided_bf16(z0[j]);
      out[(h + 1) * hd + j] = llama_strided_bf16(z1[j]);
      out[(h + 2) * hd + j] = llama_strided_bf16(z2[j]);
      out[(h + 3) * hd + j] = llama_strided_bf16(z3[j]);
    }
  }
  return 1;
}
#endif
