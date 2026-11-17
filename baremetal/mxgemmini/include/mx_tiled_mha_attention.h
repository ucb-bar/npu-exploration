#ifndef MX_TILED_MHA_ATTENTION_H
#define MX_TILED_MHA_ATTENTION_H
/* Independent-head MHA with an eight-key QK tile and 16 output lanes per PV
 * tile. Selected for Llama2-7B mix50; other profile choices remain separate. */
#include "mx_mha_attention.h"
static void mx_mha_pv_h1j16(const float *v, const float *p, int phys, int stride, int valid,
                            int hbase, int kv, uint16_t *out) {
  for (int j0 = 0; j0 < 128; j0 += 16) {
    float a0_0 = 0, a0_1 = 0, a0_2 = 0, a0_3 = 0, a0_4 = 0, a0_5 = 0, a0_6 = 0, a0_7 = 0, a0_8 = 0,
          a0_9 = 0, a0_10 = 0, a0_11 = 0, a0_12 = 0, a0_13 = 0, a0_14 = 0, a0_15 = 0;
    for (int t = 0; t < valid; t++) {
      float p0 = p[0 * stride + t];
      const float *r0 = v + ((size_t)(hbase + 0) * phys + t) * 128 + j0;
      a0_0 = fmaf(p0, r0[0], a0_0);
      a0_1 = fmaf(p0, r0[1], a0_1);
      a0_2 = fmaf(p0, r0[2], a0_2);
      a0_3 = fmaf(p0, r0[3], a0_3);
      a0_4 = fmaf(p0, r0[4], a0_4);
      a0_5 = fmaf(p0, r0[5], a0_5);
      a0_6 = fmaf(p0, r0[6], a0_6);
      a0_7 = fmaf(p0, r0[7], a0_7);
      a0_8 = fmaf(p0, r0[8], a0_8);
      a0_9 = fmaf(p0, r0[9], a0_9);
      a0_10 = fmaf(p0, r0[10], a0_10);
      a0_11 = fmaf(p0, r0[11], a0_11);
      a0_12 = fmaf(p0, r0[12], a0_12);
      a0_13 = fmaf(p0, r0[13], a0_13);
      a0_14 = fmaf(p0, r0[14], a0_14);
      a0_15 = fmaf(p0, r0[15], a0_15);
    }
    out[(hbase + 0) * 128 + j0 + 0] = llama_strided_bf16(a0_0);
    out[(hbase + 0) * 128 + j0 + 1] = llama_strided_bf16(a0_1);
    out[(hbase + 0) * 128 + j0 + 2] = llama_strided_bf16(a0_2);
    out[(hbase + 0) * 128 + j0 + 3] = llama_strided_bf16(a0_3);
    out[(hbase + 0) * 128 + j0 + 4] = llama_strided_bf16(a0_4);
    out[(hbase + 0) * 128 + j0 + 5] = llama_strided_bf16(a0_5);
    out[(hbase + 0) * 128 + j0 + 6] = llama_strided_bf16(a0_6);
    out[(hbase + 0) * 128 + j0 + 7] = llama_strided_bf16(a0_7);
    out[(hbase + 0) * 128 + j0 + 8] = llama_strided_bf16(a0_8);
    out[(hbase + 0) * 128 + j0 + 9] = llama_strided_bf16(a0_9);
    out[(hbase + 0) * 128 + j0 + 10] = llama_strided_bf16(a0_10);
    out[(hbase + 0) * 128 + j0 + 11] = llama_strided_bf16(a0_11);
    out[(hbase + 0) * 128 + j0 + 12] = llama_strided_bf16(a0_12);
    out[(hbase + 0) * 128 + j0 + 13] = llama_strided_bf16(a0_13);
    out[(hbase + 0) * 128 + j0 + 14] = llama_strided_bf16(a0_14);
    out[(hbase + 0) * 128 + j0 + 15] = llama_strided_bf16(a0_15);
  }
  for (int j0 = 0; j0 < 128; j0 += 16) {
    float a0_0 = 0, a0_1 = 0, a0_2 = 0, a0_3 = 0, a0_4 = 0, a0_5 = 0, a0_6 = 0, a0_7 = 0, a0_8 = 0,
          a0_9 = 0, a0_10 = 0, a0_11 = 0, a0_12 = 0, a0_13 = 0, a0_14 = 0, a0_15 = 0;
    for (int t = 0; t < valid; t++) {
      float p0 = p[1 * stride + t];
      const float *r0 = v + ((size_t)(hbase + 1) * phys + t) * 128 + j0;
      a0_0 = fmaf(p0, r0[0], a0_0);
      a0_1 = fmaf(p0, r0[1], a0_1);
      a0_2 = fmaf(p0, r0[2], a0_2);
      a0_3 = fmaf(p0, r0[3], a0_3);
      a0_4 = fmaf(p0, r0[4], a0_4);
      a0_5 = fmaf(p0, r0[5], a0_5);
      a0_6 = fmaf(p0, r0[6], a0_6);
      a0_7 = fmaf(p0, r0[7], a0_7);
      a0_8 = fmaf(p0, r0[8], a0_8);
      a0_9 = fmaf(p0, r0[9], a0_9);
      a0_10 = fmaf(p0, r0[10], a0_10);
      a0_11 = fmaf(p0, r0[11], a0_11);
      a0_12 = fmaf(p0, r0[12], a0_12);
      a0_13 = fmaf(p0, r0[13], a0_13);
      a0_14 = fmaf(p0, r0[14], a0_14);
      a0_15 = fmaf(p0, r0[15], a0_15);
    }
    out[(hbase + 1) * 128 + j0 + 0] = llama_strided_bf16(a0_0);
    out[(hbase + 1) * 128 + j0 + 1] = llama_strided_bf16(a0_1);
    out[(hbase + 1) * 128 + j0 + 2] = llama_strided_bf16(a0_2);
    out[(hbase + 1) * 128 + j0 + 3] = llama_strided_bf16(a0_3);
    out[(hbase + 1) * 128 + j0 + 4] = llama_strided_bf16(a0_4);
    out[(hbase + 1) * 128 + j0 + 5] = llama_strided_bf16(a0_5);
    out[(hbase + 1) * 128 + j0 + 6] = llama_strided_bf16(a0_6);
    out[(hbase + 1) * 128 + j0 + 7] = llama_strided_bf16(a0_7);
    out[(hbase + 1) * 128 + j0 + 8] = llama_strided_bf16(a0_8);
    out[(hbase + 1) * 128 + j0 + 9] = llama_strided_bf16(a0_9);
    out[(hbase + 1) * 128 + j0 + 10] = llama_strided_bf16(a0_10);
    out[(hbase + 1) * 128 + j0 + 11] = llama_strided_bf16(a0_11);
    out[(hbase + 1) * 128 + j0 + 12] = llama_strided_bf16(a0_12);
    out[(hbase + 1) * 128 + j0 + 13] = llama_strided_bf16(a0_13);
    out[(hbase + 1) * 128 + j0 + 14] = llama_strided_bf16(a0_14);
    out[(hbase + 1) * 128 + j0 + 15] = llama_strided_bf16(a0_15);
  }
  for (int j0 = 0; j0 < 128; j0 += 16) {
    float a0_0 = 0, a0_1 = 0, a0_2 = 0, a0_3 = 0, a0_4 = 0, a0_5 = 0, a0_6 = 0, a0_7 = 0, a0_8 = 0,
          a0_9 = 0, a0_10 = 0, a0_11 = 0, a0_12 = 0, a0_13 = 0, a0_14 = 0, a0_15 = 0;
    for (int t = 0; t < valid; t++) {
      float p0 = p[2 * stride + t];
      const float *r0 = v + ((size_t)(hbase + 2) * phys + t) * 128 + j0;
      a0_0 = fmaf(p0, r0[0], a0_0);
      a0_1 = fmaf(p0, r0[1], a0_1);
      a0_2 = fmaf(p0, r0[2], a0_2);
      a0_3 = fmaf(p0, r0[3], a0_3);
      a0_4 = fmaf(p0, r0[4], a0_4);
      a0_5 = fmaf(p0, r0[5], a0_5);
      a0_6 = fmaf(p0, r0[6], a0_6);
      a0_7 = fmaf(p0, r0[7], a0_7);
      a0_8 = fmaf(p0, r0[8], a0_8);
      a0_9 = fmaf(p0, r0[9], a0_9);
      a0_10 = fmaf(p0, r0[10], a0_10);
      a0_11 = fmaf(p0, r0[11], a0_11);
      a0_12 = fmaf(p0, r0[12], a0_12);
      a0_13 = fmaf(p0, r0[13], a0_13);
      a0_14 = fmaf(p0, r0[14], a0_14);
      a0_15 = fmaf(p0, r0[15], a0_15);
    }
    out[(hbase + 2) * 128 + j0 + 0] = llama_strided_bf16(a0_0);
    out[(hbase + 2) * 128 + j0 + 1] = llama_strided_bf16(a0_1);
    out[(hbase + 2) * 128 + j0 + 2] = llama_strided_bf16(a0_2);
    out[(hbase + 2) * 128 + j0 + 3] = llama_strided_bf16(a0_3);
    out[(hbase + 2) * 128 + j0 + 4] = llama_strided_bf16(a0_4);
    out[(hbase + 2) * 128 + j0 + 5] = llama_strided_bf16(a0_5);
    out[(hbase + 2) * 128 + j0 + 6] = llama_strided_bf16(a0_6);
    out[(hbase + 2) * 128 + j0 + 7] = llama_strided_bf16(a0_7);
    out[(hbase + 2) * 128 + j0 + 8] = llama_strided_bf16(a0_8);
    out[(hbase + 2) * 128 + j0 + 9] = llama_strided_bf16(a0_9);
    out[(hbase + 2) * 128 + j0 + 10] = llama_strided_bf16(a0_10);
    out[(hbase + 2) * 128 + j0 + 11] = llama_strided_bf16(a0_11);
    out[(hbase + 2) * 128 + j0 + 12] = llama_strided_bf16(a0_12);
    out[(hbase + 2) * 128 + j0 + 13] = llama_strided_bf16(a0_13);
    out[(hbase + 2) * 128 + j0 + 14] = llama_strided_bf16(a0_14);
    out[(hbase + 2) * 128 + j0 + 15] = llama_strided_bf16(a0_15);
  }
  for (int j0 = 0; j0 < 128; j0 += 16) {
    float a0_0 = 0, a0_1 = 0, a0_2 = 0, a0_3 = 0, a0_4 = 0, a0_5 = 0, a0_6 = 0, a0_7 = 0, a0_8 = 0,
          a0_9 = 0, a0_10 = 0, a0_11 = 0, a0_12 = 0, a0_13 = 0, a0_14 = 0, a0_15 = 0;
    for (int t = 0; t < valid; t++) {
      float p0 = p[3 * stride + t];
      const float *r0 = v + ((size_t)(hbase + 3) * phys + t) * 128 + j0;
      a0_0 = fmaf(p0, r0[0], a0_0);
      a0_1 = fmaf(p0, r0[1], a0_1);
      a0_2 = fmaf(p0, r0[2], a0_2);
      a0_3 = fmaf(p0, r0[3], a0_3);
      a0_4 = fmaf(p0, r0[4], a0_4);
      a0_5 = fmaf(p0, r0[5], a0_5);
      a0_6 = fmaf(p0, r0[6], a0_6);
      a0_7 = fmaf(p0, r0[7], a0_7);
      a0_8 = fmaf(p0, r0[8], a0_8);
      a0_9 = fmaf(p0, r0[9], a0_9);
      a0_10 = fmaf(p0, r0[10], a0_10);
      a0_11 = fmaf(p0, r0[11], a0_11);
      a0_12 = fmaf(p0, r0[12], a0_12);
      a0_13 = fmaf(p0, r0[13], a0_13);
      a0_14 = fmaf(p0, r0[14], a0_14);
      a0_15 = fmaf(p0, r0[15], a0_15);
    }
    out[(hbase + 3) * 128 + j0 + 0] = llama_strided_bf16(a0_0);
    out[(hbase + 3) * 128 + j0 + 1] = llama_strided_bf16(a0_1);
    out[(hbase + 3) * 128 + j0 + 2] = llama_strided_bf16(a0_2);
    out[(hbase + 3) * 128 + j0 + 3] = llama_strided_bf16(a0_3);
    out[(hbase + 3) * 128 + j0 + 4] = llama_strided_bf16(a0_4);
    out[(hbase + 3) * 128 + j0 + 5] = llama_strided_bf16(a0_5);
    out[(hbase + 3) * 128 + j0 + 6] = llama_strided_bf16(a0_6);
    out[(hbase + 3) * 128 + j0 + 7] = llama_strided_bf16(a0_7);
    out[(hbase + 3) * 128 + j0 + 8] = llama_strided_bf16(a0_8);
    out[(hbase + 3) * 128 + j0 + 9] = llama_strided_bf16(a0_9);
    out[(hbase + 3) * 128 + j0 + 10] = llama_strided_bf16(a0_10);
    out[(hbase + 3) * 128 + j0 + 11] = llama_strided_bf16(a0_11);
    out[(hbase + 3) * 128 + j0 + 12] = llama_strided_bf16(a0_12);
    out[(hbase + 3) * 128 + j0 + 13] = llama_strided_bf16(a0_13);
    out[(hbase + 3) * 128 + j0 + 14] = llama_strided_bf16(a0_14);
    out[(hbase + 3) * 128 + j0 + 15] = llama_strided_bf16(a0_15);
  }
}

static int mx_mha_qkt8_pvh1j16(const float *q, const float *keys, const float *values, int heads,
                               int kv_heads, int hd, int logical_capacity, int physical_capacity,
                               int score_stride, int valid, float scale, uint16_t *out,
                               float *workspace) {
  if (!q || !keys || !values || !out || !workspace || heads < 4 || heads % 4 || kv_heads != heads ||
      hd != 128 || valid < 1 || valid > logical_capacity || physical_capacity < logical_capacity ||
      score_stride < logical_capacity)
    return 0;
  for (int h = 0; h < heads; h += 4) {
    float *s0 = workspace, *s1 = s0 + score_stride, *s2 = s1 + score_stride,
          *s3 = s2 + score_stride;
    const float *q0 = q + h * hd, *q1 = q0 + hd, *q2 = q1 + hd, *q3 = q2 + hd;
    float max0 = -INFINITY, max1 = -INFINITY, max2 = -INFINITY, max3 = -INFINITY;

    {
      const float *qv = q + (h + 0) * hd;
      for (int tb = 0; tb < valid; tb += 8) {
        if (tb + 8 <= valid) {
          float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
          for (int j = 0; j < 128; j++) {
            float qj = qv[j];
            float k0 = keys[((size_t)(h + 0) * physical_capacity + tb + 0) * 128 + j];
            float k1 = keys[((size_t)(h + 0) * physical_capacity + tb + 1) * 128 + j];
            float k2 = keys[((size_t)(h + 0) * physical_capacity + tb + 2) * 128 + j];
            float k3 = keys[((size_t)(h + 0) * physical_capacity + tb + 3) * 128 + j];
            float k4 = keys[((size_t)(h + 0) * physical_capacity + tb + 4) * 128 + j];
            float k5 = keys[((size_t)(h + 0) * physical_capacity + tb + 5) * 128 + j];
            float k6 = keys[((size_t)(h + 0) * physical_capacity + tb + 6) * 128 + j];
            float k7 = keys[((size_t)(h + 0) * physical_capacity + tb + 7) * 128 + j];
            a0 = fmaf(qj, k0, a0);
            a1 = fmaf(qj, k1, a1);
            a2 = fmaf(qj, k2, a2);
            a3 = fmaf(qj, k3, a3);
            a4 = fmaf(qj, k4, a4);
            a5 = fmaf(qj, k5, a5);
            a6 = fmaf(qj, k6, a6);
            a7 = fmaf(qj, k7, a7);
          }
          a0 *= scale;
          s0[tb + 0] = a0;
          if (a0 > max0)
            max0 = a0;
          a1 *= scale;
          s0[tb + 1] = a1;
          if (a1 > max0)
            max0 = a1;
          a2 *= scale;
          s0[tb + 2] = a2;
          if (a2 > max0)
            max0 = a2;
          a3 *= scale;
          s0[tb + 3] = a3;
          if (a3 > max0)
            max0 = a3;
          a4 *= scale;
          s0[tb + 4] = a4;
          if (a4 > max0)
            max0 = a4;
          a5 *= scale;
          s0[tb + 5] = a5;
          if (a5 > max0)
            max0 = a5;
          a6 *= scale;
          s0[tb + 6] = a6;
          if (a6 > max0)
            max0 = a6;
          a7 *= scale;
          s0[tb + 7] = a7;
          if (a7 > max0)
            max0 = a7;
        } else {
          for (int t = tb; t < valid; t++) {
            float a = 0;
            for (int j = 0; j < 128; j++)
              a = fmaf(qv[j], keys[((size_t)(h + 0) * physical_capacity + t) * 128 + j], a);
            a *= scale;
            s0[t] = a;
            if (a > max0)
              max0 = a;
          }
        }
      }
    }
    {
      const float *qv = q + (h + 1) * hd;
      for (int tb = 0; tb < valid; tb += 8) {
        if (tb + 8 <= valid) {
          float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
          for (int j = 0; j < 128; j++) {
            float qj = qv[j];
            float k0 = keys[((size_t)(h + 1) * physical_capacity + tb + 0) * 128 + j];
            float k1 = keys[((size_t)(h + 1) * physical_capacity + tb + 1) * 128 + j];
            float k2 = keys[((size_t)(h + 1) * physical_capacity + tb + 2) * 128 + j];
            float k3 = keys[((size_t)(h + 1) * physical_capacity + tb + 3) * 128 + j];
            float k4 = keys[((size_t)(h + 1) * physical_capacity + tb + 4) * 128 + j];
            float k5 = keys[((size_t)(h + 1) * physical_capacity + tb + 5) * 128 + j];
            float k6 = keys[((size_t)(h + 1) * physical_capacity + tb + 6) * 128 + j];
            float k7 = keys[((size_t)(h + 1) * physical_capacity + tb + 7) * 128 + j];
            a0 = fmaf(qj, k0, a0);
            a1 = fmaf(qj, k1, a1);
            a2 = fmaf(qj, k2, a2);
            a3 = fmaf(qj, k3, a3);
            a4 = fmaf(qj, k4, a4);
            a5 = fmaf(qj, k5, a5);
            a6 = fmaf(qj, k6, a6);
            a7 = fmaf(qj, k7, a7);
          }
          a0 *= scale;
          s1[tb + 0] = a0;
          if (a0 > max1)
            max1 = a0;
          a1 *= scale;
          s1[tb + 1] = a1;
          if (a1 > max1)
            max1 = a1;
          a2 *= scale;
          s1[tb + 2] = a2;
          if (a2 > max1)
            max1 = a2;
          a3 *= scale;
          s1[tb + 3] = a3;
          if (a3 > max1)
            max1 = a3;
          a4 *= scale;
          s1[tb + 4] = a4;
          if (a4 > max1)
            max1 = a4;
          a5 *= scale;
          s1[tb + 5] = a5;
          if (a5 > max1)
            max1 = a5;
          a6 *= scale;
          s1[tb + 6] = a6;
          if (a6 > max1)
            max1 = a6;
          a7 *= scale;
          s1[tb + 7] = a7;
          if (a7 > max1)
            max1 = a7;
        } else {
          for (int t = tb; t < valid; t++) {
            float a = 0;
            for (int j = 0; j < 128; j++)
              a = fmaf(qv[j], keys[((size_t)(h + 1) * physical_capacity + t) * 128 + j], a);
            a *= scale;
            s1[t] = a;
            if (a > max1)
              max1 = a;
          }
        }
      }
    }
    {
      const float *qv = q + (h + 2) * hd;
      for (int tb = 0; tb < valid; tb += 8) {
        if (tb + 8 <= valid) {
          float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
          for (int j = 0; j < 128; j++) {
            float qj = qv[j];
            float k0 = keys[((size_t)(h + 2) * physical_capacity + tb + 0) * 128 + j];
            float k1 = keys[((size_t)(h + 2) * physical_capacity + tb + 1) * 128 + j];
            float k2 = keys[((size_t)(h + 2) * physical_capacity + tb + 2) * 128 + j];
            float k3 = keys[((size_t)(h + 2) * physical_capacity + tb + 3) * 128 + j];
            float k4 = keys[((size_t)(h + 2) * physical_capacity + tb + 4) * 128 + j];
            float k5 = keys[((size_t)(h + 2) * physical_capacity + tb + 5) * 128 + j];
            float k6 = keys[((size_t)(h + 2) * physical_capacity + tb + 6) * 128 + j];
            float k7 = keys[((size_t)(h + 2) * physical_capacity + tb + 7) * 128 + j];
            a0 = fmaf(qj, k0, a0);
            a1 = fmaf(qj, k1, a1);
            a2 = fmaf(qj, k2, a2);
            a3 = fmaf(qj, k3, a3);
            a4 = fmaf(qj, k4, a4);
            a5 = fmaf(qj, k5, a5);
            a6 = fmaf(qj, k6, a6);
            a7 = fmaf(qj, k7, a7);
          }
          a0 *= scale;
          s2[tb + 0] = a0;
          if (a0 > max2)
            max2 = a0;
          a1 *= scale;
          s2[tb + 1] = a1;
          if (a1 > max2)
            max2 = a1;
          a2 *= scale;
          s2[tb + 2] = a2;
          if (a2 > max2)
            max2 = a2;
          a3 *= scale;
          s2[tb + 3] = a3;
          if (a3 > max2)
            max2 = a3;
          a4 *= scale;
          s2[tb + 4] = a4;
          if (a4 > max2)
            max2 = a4;
          a5 *= scale;
          s2[tb + 5] = a5;
          if (a5 > max2)
            max2 = a5;
          a6 *= scale;
          s2[tb + 6] = a6;
          if (a6 > max2)
            max2 = a6;
          a7 *= scale;
          s2[tb + 7] = a7;
          if (a7 > max2)
            max2 = a7;
        } else {
          for (int t = tb; t < valid; t++) {
            float a = 0;
            for (int j = 0; j < 128; j++)
              a = fmaf(qv[j], keys[((size_t)(h + 2) * physical_capacity + t) * 128 + j], a);
            a *= scale;
            s2[t] = a;
            if (a > max2)
              max2 = a;
          }
        }
      }
    }
    {
      const float *qv = q + (h + 3) * hd;
      for (int tb = 0; tb < valid; tb += 8) {
        if (tb + 8 <= valid) {
          float a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
          for (int j = 0; j < 128; j++) {
            float qj = qv[j];
            float k0 = keys[((size_t)(h + 3) * physical_capacity + tb + 0) * 128 + j];
            float k1 = keys[((size_t)(h + 3) * physical_capacity + tb + 1) * 128 + j];
            float k2 = keys[((size_t)(h + 3) * physical_capacity + tb + 2) * 128 + j];
            float k3 = keys[((size_t)(h + 3) * physical_capacity + tb + 3) * 128 + j];
            float k4 = keys[((size_t)(h + 3) * physical_capacity + tb + 4) * 128 + j];
            float k5 = keys[((size_t)(h + 3) * physical_capacity + tb + 5) * 128 + j];
            float k6 = keys[((size_t)(h + 3) * physical_capacity + tb + 6) * 128 + j];
            float k7 = keys[((size_t)(h + 3) * physical_capacity + tb + 7) * 128 + j];
            a0 = fmaf(qj, k0, a0);
            a1 = fmaf(qj, k1, a1);
            a2 = fmaf(qj, k2, a2);
            a3 = fmaf(qj, k3, a3);
            a4 = fmaf(qj, k4, a4);
            a5 = fmaf(qj, k5, a5);
            a6 = fmaf(qj, k6, a6);
            a7 = fmaf(qj, k7, a7);
          }
          a0 *= scale;
          s3[tb + 0] = a0;
          if (a0 > max3)
            max3 = a0;
          a1 *= scale;
          s3[tb + 1] = a1;
          if (a1 > max3)
            max3 = a1;
          a2 *= scale;
          s3[tb + 2] = a2;
          if (a2 > max3)
            max3 = a2;
          a3 *= scale;
          s3[tb + 3] = a3;
          if (a3 > max3)
            max3 = a3;
          a4 *= scale;
          s3[tb + 4] = a4;
          if (a4 > max3)
            max3 = a4;
          a5 *= scale;
          s3[tb + 5] = a5;
          if (a5 > max3)
            max3 = a5;
          a6 *= scale;
          s3[tb + 6] = a6;
          if (a6 > max3)
            max3 = a6;
          a7 *= scale;
          s3[tb + 7] = a7;
          if (a7 > max3)
            max3 = a7;
        } else {
          for (int t = tb; t < valid; t++) {
            float a = 0;
            for (int j = 0; j < 128; j++)
              a = fmaf(qv[j], keys[((size_t)(h + 3) * physical_capacity + t) * 128 + j], a);
            a *= scale;
            s3[t] = a;
            if (a > max3)
              max3 = a;
          }
        }
      }
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

    mx_mha_pv_h1j16(values, workspace, physical_capacity, score_stride, valid, h, 0, out);
  }
  return 1;
}
#endif
