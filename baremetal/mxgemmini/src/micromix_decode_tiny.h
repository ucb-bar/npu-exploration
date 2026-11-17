#ifndef MICROMIX_DECODE_ATTENTION_MODE
#define MICROMIX_DECODE_ATTENTION_MODE 1
#endif
static int native_attention_error;
/* TinyLlama layer 5, eight sequential batch-one decode calls.  Only the
 * projection engine pads the single valid row to 32.  Attention sees exactly
 * 2048+step+1 keys; only one K/V is appended on each call.  Scalar FP32 glue
 * is deliberate and included in end-to-end timing (not a VPU benchmark). */
#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#ifdef DECODE_CAPTURE_HEADER
#include DECODE_CAPTURE_HEADER
#else
#include "llama_layer_e2e.h"
#endif
#include "mx_host.h"
#include "mx_llm_attention.h"
#ifdef DECODE_HOST
#define MX_MICROMIX_H
typedef struct {
  int k, n, nb4, nb6, nb8;
  const uint8_t *b4, *b6, *b8, *scales;
} mxmm_weight;
typedef struct {
  int nc, run, fence_groups;
} mxmm_schedule;
#else
#include "mx_micromix.h"
#ifndef MXMM_HAS_ROUND2_PACK
#error "MicroMX decode requires the integer packing API"
#endif
#endif

#define D 2048
#define F 5632
#define HD 64
#define NH 32
#define NKV 4
#define PREFIX 2048
#define STEPS 8
#define CAP (PREFIX + STEPS)
#define PAD 32
#define A64 __attribute__((aligned(64)))
#ifndef DECODE_MIX
#define DECODE_MIX "fp8"
#endif
#ifndef DECODE_STEPS
#define DECODE_STEPS STEPS
#endif
#ifndef DECODE_REPEATS
#define DECODE_REPEATS 1
#endif
#ifndef DECODE_NC
#define DECODE_NC 256
#endif
#ifndef DECODE_RUN
#define DECODE_RUN 8
#endif
#ifndef DECODE_FENCE
#define DECODE_FENCE 1
#endif
#ifndef DECODE_BASELINE
#define DECODE_BASELINE 1
#endif
#if DECODE_STEPS < 1 || DECODE_STEPS > 8
#error "Decode capture contains exactly eight sequential tokens"
#endif
#if DECODE_REPEATS < 1
#error "Decode requires at least one independent sequence per schedule"
#endif

#ifdef DECODE_WEIGHT_HEADER
#include DECODE_WEIGHT_HEADER
#else
#define EW(K, N, T)                                                                                \
  {K,                                                                                              \
   N,                                                                                              \
   0,                                                                                              \
   0,                                                                                              \
   K / 32,                                                                                         \
   NULL,                                                                                           \
   NULL,                                                                                           \
   E2E_AT(E2E_OFF_W##T##_CODES, uint8_t),                                                          \
   E2E_AT(E2E_OFF_W##T##_SCALES, uint8_t)}
static const mxmm_weight mxmm_weights[7] = {EW(D, D, Q), EW(D, 256, K), EW(D, 256, V), EW(D, D, O),
                                            EW(D, F, G), EW(D, F, U),   EW(F, D, D)};
#endif

/* Independent reference bits are compared after execution, never supplied to
 * any projection, nonlinearity, attention or residual operation. */
#ifndef DECODE_NO_GOLDEN
extern const uint8_t _binary_llama_decode_expect_bin_start[];
static const uint16_t *golden = (const uint16_t *)_binary_llama_decode_expect_bin_start;
#endif
static const int widths[8] = {D, 256, 256, D, F, F, D, D};
static const int offsets[8] = {
    0, D, D + 256, D + 512, 2 * D + 512, 2 * D + 512 + F, 2 * D + 512 + 2 * F, 3 * D + 512 + 2 * F};
#define EXPECT_STRIDE (4 * D + 512 + 2 * F)
static uint16_t input[F] A64, q[PAD * D] A64, k[PAD * 256] A64, v[PAD * 256] A64;
static uint16_t o[D] A64, ya[PAD * D] A64, hm[D] A64, g[PAD * F] A64, u[PAD * F] A64,
    ym[PAD * D] A64, hout[D] A64;
static uint8_t workspace[PAD * (F + F / 32)] A64;
/* Dequantized cache, from the supplied valid FP8 prefix and locally computed
 * FP8 new vectors.  Empty slots contain NaNs to catch accidental fake keys. */
static float kc[NKV][CAP][HD] A64, vc[NKV][CAP][HD] A64;
static float query[NH][HD];
static uint16_t baseline[STEPS][EXPECT_STRIDE] A64;
static uint64_t phase_cycles[7], projection_cycles, glue_cycles;
static uint64_t decode_cycles(void) {
#ifdef DECODE_HOST
  static uint64_t c;
  return ++c; /* explicitly functional, never hardware time */
#else
  uint64_t c;
  asm volatile("csrr %0,mcycle" : "=r"(c)::"memory");
  return c;
#endif
}
static uint64_t decode_instret(void) {
#ifdef DECODE_HOST
  static uint64_t c;
  return ++c;
#else
  uint64_t c;
  asm volatile("csrr %0,minstret" : "=r"(c)::"memory");
  return c;
#endif
}
static float bf(float x) { return mx_bf16_to_f32(mx_f32_to_bf16_rne(x)); }
static uint64_t hash(const uint16_t *p, int n) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < n; i++) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}
static int gate(const uint16_t *a, const uint16_t *b, int n, double *relative) {
  double err = 0, ref = 0;
  for (int i = 0; i < n; i++) {
    double x = mx_bf16_to_f32(a[i]), y = mx_bf16_to_f32(b[i]);
    if (!isfinite(x) || !isfinite(y)) {
      *relative = INFINITY;
      return 1;
    }
    err += (x - y) * (x - y);
    ref += y * y;
  }
  *relative = ref ? sqrt(err / ref) : (err ? INFINITY : 0);
  return *relative > 0.01;
}
static int update_gate(const uint16_t *a, const uint16_t *b, const uint16_t *h, double *relative) {
  double err = 0, ref = 0;
  for (int i = 0; i < D; i++) {
    double x = mx_bf16_to_f32(a[i]), y = mx_bf16_to_f32(b[i]), z = mx_bf16_to_f32(h[i]);
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) {
      *relative = INFINITY;
      return 1;
    }
    err += (x - y) * (x - y);
    ref += (y - z) * (y - z);
  }
  *relative = ref ? sqrt(err / ref) : (err ? INFINITY : 0);
  return *relative > 0.01;
}
#ifdef DECODE_HOST
/* Functional C model of the fixed BF16 PE. NumPy in decode/reference.py is an
 * independently written reference.  Host times are never FPGA measurements. */
static float fmt_decode(unsigned c, int f) {
  if (f == 4) {
    static const float t[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    return (c & 8) ? -t[c & 7] : t[c & 7];
  }
  int m = f == 6 ? 2 : 3, bias = f == 6 ? 3 : 7, mask = f == 6 ? 31 : 127, e = (c & mask) >> m,
      frac = c & ((1 << m) - 1);
  float x =
      ldexpf(e ? 1.f + frac / (float)(1 << m) : frac / (float)(1 << m), e ? e - bias : 1 - bias);
  return c & (mask + 1) ? -x : x;
}
static unsigned fmt_encode(float x, int f) {
  unsigned lo = 0,
           hi = f == 4   ? 7
                : f == 6 ? 31
                         : 126,
           sign = signbit(x) ? (f == 4   ? 8
                                : f == 6 ? 32
                                         : 128)
                             : 0;
  float a = fabsf(x);
  while (lo < hi) {
    unsigned mid = (lo + hi) / 2;
    if (fmt_decode(mid, f) < a)
      lo = mid + 1;
    else
      hi = mid;
  }
  if (lo) {
    float dl = a - fmt_decode(lo - 1, f), dh = fmt_decode(lo, f) - a;
    if (dl < dh || (dl == dh && !((lo - 1) & 1)))
      --lo;
  }
  return lo | sign;
}
static int host_project(const uint16_t *a, const mxmm_weight *w, uint16_t *out) {
  for (int n = 0; n < PAD * w->n; n++)
    out[n] = 0;
  float total[F] = {0}, partial[F];
  int ns[] = {w->nb4, w->nb6, w->nb8}, fmts[] = {4, 6, 8}, group = 0;
  const uint8_t *bp[] = {w->b4, w->b6, w->b8};
  for (int f = 0; f < 3; f++)
    for (int block = 0; block < ns[f]; block++, group++) {
      float am = 0;
      for (int j = 0; j < 32; j++) {
        float x = fabsf(mx_bf16_to_f32(a[group * 32 + j]));
        if (x > am)
          am = x;
      }
      int ae = am ? mx_floor_log2(am) : 0;
      float av[32];
      for (int j = 0; j < 32; j++)
        av[j] = fmt_decode(fmt_encode(ldexpf(mx_bf16_to_f32(a[group * 32 + j]), -ae), fmts[f]),
                           fmts[f]);
      for (int half = 0; half < 2; half++) {
        for (int n = 0; n < w->n; n++)
          partial[n] = 0;
        for (int j = half * 16; j < (half + 1) * 16; j++)
          for (int n = 0; n < w->n; n++) {
            size_t ix =
                (size_t)(block * 32 + j) * (f == 0 ? w->n / 2 : w->n) + (f == 0 ? n / 2 : n);
            unsigned c = bp[f][ix];
            if (f == 0)
              c = (c >> (4 * (n & 1))) & 15;
            float p = av[j] * fmt_decode(c, fmts[f]);
            union {
              float f;
              uint32_t u;
            } bits = {p};
            bits.u &= 0xffff0000u;
            p = bits.f;
            partial[n] = bf(bf(partial[n]) + bf(p));
          }
        for (int n = 0; n < w->n; n++) {
          int se = ae + (int)w->scales[(size_t)group * w->n + n] - 127;
          float scaled = bf(ldexpf(partial[n], se));
          total[n] = bf(total[n] + scaled);
        }
      }
    }
  for (int n = 0; n < w->n; n++)
    out[n] = mx_f32_to_bf16_rne(total[n]);
  return 0;
}
#endif
static int project(int stage, const uint16_t *a, uint16_t *out, mxmm_schedule schedule) {
  uint64_t t = decode_cycles();
  int rc;
#ifdef DECODE_HOST
  (void)schedule;
  rc = host_project(a, &mxmm_weights[stage], out);
#else
  rc = mxmm_project_bf16(a, 1, PAD, &mxmm_weights[stage], out, workspace, schedule);
#endif
  uint64_t dt = decode_cycles() - t;
  phase_cycles[stage] += dt;
  projection_cycles += dt;
  return rc;
}
static void init_cache(void) {
  const uint8_t *kt = E2E_AT(E2E_OFF_KT_CACHE, uint8_t), *ks = E2E_AT(E2E_OFF_KT_SCALES, uint8_t),
                *vv = E2E_AT(E2E_OFF_V_CACHE, uint8_t), *vs = E2E_AT(E2E_OFF_V_SCALES, uint8_t);
  for (int h = 0; h < NKV; h++)
    for (int t = 0; t < CAP; t++)
      for (int d = 0; d < HD; d++) {
        kc[h][t][d] = t < PREFIX ? mx_e4m3_decode(kt[(h * HD + d) * 2112 + t]) *
                                       mx_e8m0_decode(ks[(h * 2 + d / 32) * 2112 + t])
                                 : NAN;
        vc[h][t][d] = t < PREFIX ? mx_e4m3_decode(vv[(h * 2112 + t) * HD + d]) *
                                       mx_e8m0_decode(vs[(h * 66 + t / 32) * HD + d])
                                 : NAN;
      }
}
static void rms(const uint16_t *h, const uint16_t *w, int n, uint16_t *out) {
  double ss = 0;
  for (int j = 0; j < n; j++) {
    double x = mx_bf16_to_f32(h[j]);
    ss += x * x;
  }
  float inv = 1.f / sqrtf((float)(ss / n) + E2E_EPS);
  for (int j = 0; j < n; j++)
    out[j] = mx_f32_to_bf16_rne(mx_bf16_to_f32(h[j]) * inv * mx_bf16_to_f32(w[j]));
}
static void quant_fp8(float *x) {
  for (int b = 0; b < HD; b += 32) {
    float am = 0;
    for (int j = 0; j < 32; j++) {
      float y = fabsf(x[b + j]);
      if (y > am)
        am = y;
    }
    int e = mx_floor_log2(fmaxf(am, 1.1920929e-7f));
    for (int j = 0; j < 32; j++)
      x[b + j] = mx_e4m3_decode(mx_e4m3_encode(ldexpf(x[b + j], -e))) * mx_pow2(e);
  }
}
static void rope_append(int step) {
  const uint16_t *cs = E2E_AT(E2E_OFF_ROPE_CQ, uint16_t) + step * (D / 2),
                 *sn = E2E_AT(E2E_OFF_ROPE_SQ, uint16_t) + step * (D / 2);
  for (int h = 0; h < NH; h++)
    for (int j = 0; j < 32; j++) {
      float a = mx_bf16_to_f32(q[h * 32 + j]), b = mx_bf16_to_f32(q[D / 2 + h * 32 + j]),
            c = mx_bf16_to_f32(cs[h * 32 + j]), s = mx_bf16_to_f32(sn[h * 32 + j]);
      query[h][j] = bf(a * c - b * s);
      query[h][j + 32] = bf(b * c + a * s);
    }
  for (int h = 0; h < NH; h++)
    quant_fp8(query[h]);
  for (int h = 0; h < NKV; h++) {
    float *key = kc[h][PREFIX + step], *val = vc[h][PREFIX + step];
    for (int j = 0; j < 32; j++) {
      float a = mx_bf16_to_f32(k[h * 32 + j]), b = mx_bf16_to_f32(k[128 + h * 32 + j]),
            c = mx_bf16_to_f32(cs[j]), s = mx_bf16_to_f32(sn[j]);
      key[j] = bf(a * c - b * s);
      key[j + 32] = bf(b * c + a * s);
    }
    for (int j = 0; j < HD; j++)
      val[j] = mx_bf16_to_f32(v[h * HD + j]);
    quant_fp8(key);
    quant_fp8(val);
  }
}

static float native_attention_workspace[8 * CAP + HD] A64;
static void attention(int valid) {
  const mx_llm_attention_args args = {
      NH, NKV, HD, CAP, CAP, CAP, (size_t)valid, 0.125f, &query[0][0], &kc[0][0][0], &vc[0][0][0]};
  int rc = mx_llm_attention_bf16(&args, (mx_llm_attention_mode)MICROMIX_DECODE_ATTENTION_MODE, o,
                                 native_attention_workspace,
                                 sizeof(native_attention_workspace) / sizeof(float), NULL);
  native_attention_error |= rc != 0;
}

static int cache_guard(int valid) {
  for (int h = 0; h < NKV; h++)
    for (int t = PREFIX; t < CAP; t++)
      for (int j = 0; j < HD; j++) {
        if (t < valid) {
          if (!isfinite(kc[h][t][j]) || !isfinite(vc[h][t][j]))
            return 1;
        } else if (!isnan(kc[h][t][j]) || !isnan(vc[h][t][j]))
          return 1;
      }
  return 0;
}
static int check_stage(int pass, int trial, int step, int stage, const uint16_t *value) {
  int n = widths[stage], off = offsets[stage];
  double re = 0;
  int bad = 0;
  if (pass == 0 && trial == 0)
    memcpy(baseline[step] + off, value, n * 2);
  else
    bad = gate(value, baseline[step] + off, n, &re);
  printf(
      "DECODE_GATE mix=%s variant=%d trial=%d step=%d stage=%d valid_rows=1 rel_l2_ppm=%lu pass=%d hash=%lx\n",
      DECODE_MIX, pass, trial, step, stage, (unsigned long)(isfinite(re) ? re * 1e6 : 1000000000),
      !bad, hash(value, n));
#ifndef DECODE_NO_GOLDEN
  const uint16_t *ref = golden + step * EXPECT_STRIDE + off;
  int mismatches = 0;
  for (int j = 0; j < n; j++)
    mismatches += value[j] != ref[j];
  double gr;
  int gb = gate(value, ref, n, &gr);
  bad |= gb;
  printf(
      "DECODE_REFERENCE variant=%d trial=%d step=%d stage=%d rel_l2_ppm=%lu bit_mismatches=%d pass=%d\n",
      pass, trial, step, stage, (unsigned long)(isfinite(gr) ? gr * 1e6 : 1000000000), mismatches,
      !gb);
#endif
  return bad;
}
int main(void) {
  mx_host_init();
  int failed = 0;
  printf(
      "MX_NATIVE_DECODE_BEGIN model=tiny mix=%s steps=%d repeats=%d passes=2 comparison=selected_profile_replay timing_qualified=0 attention_mode=%d\n",
      DECODE_MIX, DECODE_STEPS, DECODE_REPEATS, MICROMIX_DECODE_ATTENTION_MODE);
  printf(
      "MX_DECODE_BEGIN mix=%s valid_rows=1 padded_rows=32 steps=%d repeats=%d prefix_keys=2048 scalar_glue=fp32 cache=fp8_values_f32_mirror candidate_nc=%d candidate_run=%d candidate_fence=%d\n",
      DECODE_MIX, DECODE_STEPS, DECODE_REPEATS, DECODE_NC, DECODE_RUN, DECODE_FENCE);
#ifndef DECODE_HOST
  gemmini_flush(0);
#endif
  for (int pass = 0; pass < 2; pass++) {
    mxmm_schedule schedule = (mxmm_schedule){DECODE_NC, DECODE_RUN, DECODE_FENCE};
#ifdef MXMM_HAS_FAST_PACK
    mxmm_fast_pack = 1;
#endif
#ifdef MXMM_HAS_ROUND2_PACK
    mxmm_round2_pack = 1;
#endif
    for (int trial = 0; trial < DECODE_REPEATS; trial++) {
      projection_cycles = glue_cycles = 0;
      memset(phase_cycles, 0, sizeof(phase_cycles));
#ifndef DECODE_HOST
      mxmm_pack_cycles = mxmm_mesh_cycles = 0;
#endif
      uint64_t init = decode_cycles();
      init_cache();
      uint64_t cache_cycles = decode_cycles() - init, total = 0;
      for (int step = 0; step < DECODE_STEPS; step++) {
        const uint16_t *hin = E2E_AT(E2E_OFF_H_PRE, uint16_t) + step * D;
        uint64_t begin = decode_cycles(), begin_ins = decode_instret(), p0 = projection_cycles;
        rms(hin, E2E_AT(E2E_OFF_W_IN_LN, uint16_t), D, input);
        failed |= project(0, input, q, schedule);
        failed |= project(1, input, k, schedule);
        failed |= project(2, input, v, schedule);
        rope_append(step);
        failed |= cache_guard(PREFIX + step + 1);
        attention(PREFIX + step + 1);
        failed |= native_attention_error;
        failed |= project(3, o, ya, schedule);
        for (int j = 0; j < D; j++)
          hm[j] = mx_f32_to_bf16_rne(mx_bf16_to_f32(hin[j]) + mx_bf16_to_f32(ya[j]));
        rms(hm, E2E_AT(E2E_OFF_W_POST_LN, uint16_t), D, input);
        failed |= project(4, input, g, schedule);
        failed |= project(5, input, u, schedule);
        for (int j = 0; j < F; j++)
          input[j] = mx_f32_to_bf16_rne(mx_silu(mx_bf16_to_f32(g[j])) * mx_bf16_to_f32(u[j]));
        failed |= project(6, input, ym, schedule);
        for (int j = 0; j < D; j++)
          hout[j] = mx_f32_to_bf16_rne(mx_bf16_to_f32(hm[j]) + mx_bf16_to_f32(ym[j]));
        uint64_t end_ins = decode_instret(), end = decode_cycles(), elapsed = end - begin;
        total += elapsed;
        glue_cycles += elapsed - (projection_cycles - p0);
        const uint16_t *stages[] = {q, k, v, ya, g, u, ym, hout};
        for (int s = 0; s < 8; s++)
          failed |= check_stage(pass, trial, step, s, stages[s]);
        double re = 0;
        if (pass) {
          int bad = update_gate(hout, baseline[step] + offsets[7], hin, &re);
          failed |= bad;
          printf(
              "DECODE_UPDATE mix=%s variant=%d trial=%d step=%d comparison=baseline rel_l2_ppm=%lu pass=%d\n",
              DECODE_MIX, pass, trial, step, (unsigned long)(isfinite(re) ? re * 1e6 : 1000000000),
              !bad);
        }
#ifndef DECODE_NO_GOLDEN
        {
          int bad = update_gate(hout, golden + step * EXPECT_STRIDE + offsets[7], hin, &re);
          failed |= bad;
          printf(
              "DECODE_UPDATE mix=%s variant=%d trial=%d step=%d comparison=independent rel_l2_ppm=%lu pass=%d\n",
              DECODE_MIX, pass, trial, step, (unsigned long)(isfinite(re) ? re * 1e6 : 1000000000),
              !bad);
        }
#endif
        printf(
            "IRIS_REGION {\"case_id\":\"decode_%s_p%d_r%d_s%d\",\"region\":\"decode_token\",\"hart\":0,\"start_cycle\":%lu,\"end_cycle\":%lu,\"instret_delta\":%lu,\"trial\":%d,\"warmup\":%s,\"valid_rows\":1,\"padded_rows\":32,\"valid_keys\":%d,\"checksum\":%lu,\"pass\":%s}\n",
            DECODE_MIX, pass, trial, step, begin, end, end_ins - begin_ins, trial,
            trial == 0 ? "true" : "false", PREFIX + step + 1, hash(hout, D),
            failed ? "false" : "true");
      }
      printf(
          "DECODE_TIME mix=%s pass=%d trial=%d warmup=%d total_cycles=%lu total_with_cache_init=%lu projection_cycles=%lu glue_cycles=%lu cache_init_cycles=%lu valid_tokens=%d padded_rows=%d source=%s\n",
          DECODE_MIX, pass, trial, trial == 0, total, total + cache_cycles, projection_cycles,
          glue_cycles, cache_cycles, DECODE_STEPS, PAD,
#ifdef DECODE_HOST
          "host_functional"
#elif defined(SPIKE_SIM)
          "spike_functional"
#else
          "fpga_mcycle"
#endif
      );
#ifndef DECODE_HOST
      printf(
          "DECODE_PROFILE mix=%s pass=%d trial=%d pack_cycles=%lu mesh_cycles=%lu q_cycles=%lu k_cycles=%lu v_cycles=%lu o_cycles=%lu gate_cycles=%lu up_cycles=%lu down_cycles=%lu\n",
          DECODE_MIX, pass, trial, mxmm_pack_cycles, mxmm_mesh_cycles, phase_cycles[0],
          phase_cycles[1], phase_cycles[2], phase_cycles[3], phase_cycles[4], phase_cycles[5],
          phase_cycles[6]);
#endif
    }
  }
  printf("MX_DECODE_DONE mix=%s failed=%d\n", DECODE_MIX, failed);
  return failed;
}
