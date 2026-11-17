/* Independent integer-valued synthetic regression; no pretrained/timing claim. */
#include "mx_micromix_dim16.h"
#include "mx_host.h"
#include "mx_micromix_prepared.h"
#include "mx_micromix_numeric.h"
#include "mx_micromix_profiles.h"
int micromix_other_tu_defaults(void);
#define M 64
#define K 256
#define N 128
#define A64 __attribute__((aligned(64)))
static uint8_t a[M * K] A64, asc[M * K / 32] A64, b4[K * N / 2] A64, b6[K * N] A64, b8[K * N] A64,
    bsc[K * N / 32] A64;
static uint8_t workspace[M * (K + K / 32)] A64, owned[M * (K + K / 32)] A64;
static uint16_t out[M * N] A64, ref[M * N] A64, prepared[M * N] A64;
static int av(int r, int k) { return ((r * 17 + k * 31 + k / 7) % 5) ? 1 : -1; }
static int bv(int k, int n) { return ((k * 13 + n * 23 + n / 3) % 7) ? 1 : -1; }
static uint16_t bf16_integer(int v) {
  float x = v;
  uint32_t u;
  memcpy(&u, &x, 4);
  return u >> 16;
}
static uint64_t checksum(const uint16_t *x) {
  uint64_t h = 1469598103934665603ULL;
  for (int i = 0; i < M * N; i++) {
    h ^= x[i];
    h *= 1099511628211ULL;
  }
  return h;
}
int main(void) {
  gemmini_flush(0);
  int failed = 0;
  if (mxmm_integer_pack != 0 || mxmm_fast_pack != 1 || !micromix_other_tu_defaults())
    return 1;
  mxmm_schedule selected;
  if (mxmm_profile_apply(MX_LLM_TINYLLAMA, MXMM_PROFILE_PREFILL, MXMM_PROFILE_MIX50, &selected))
    return 1;
  if (!mxmm_integer_pack || !micromix_other_tu_defaults())
    return 1;
  for (int mix = 0; mix < 2; mix++) {
    int ns[3] = {mix ? 5 : 4, 2, mix ? 1 : 2}, kb = 0;
    memset(asc, 127, sizeof asc);
    memset(bsc, 127, sizeof bsc);
    for (int r = 0; r < M; r++)
      for (int k = 0; k < K; k++)
        a[r * K + k] = av(r, k) > 0 ? 0x38 : 0xb8;
    for (int g = 0; g < 3; g++) {
      for (int k = 0; k < ns[g] * 32; k++)
        for (int n = 0; n < N; n++) {
          int s = bv(kb * 32 + k, n) < 0;
          if (!g) {
            unsigned c = 2 | (s ? 8 : 0);
            size_t i = (size_t)k * N / 2 + n / 2;
            if (!(n & 1))
              b4[i] = c;
            else
              b4[i] |= c << 4;
          } else if (g == 1)
            b6[k * N + n] = 12 | (s ? 32 : 0);
          else
            b8[k * N + n] = 56 | (s ? 128 : 0);
        }
      kb += ns[g];
    }
    for (int r = 0; r < M; r++)
      for (int n = 0; n < N; n++) {
        int sum = 0;
        for (int k = 0; k < K; k++)
          sum += av(r, k) * bv(k, n);
        ref[r * N + n] = bf16_integer(sum);
      }
    mxmm_weight w = {K, N, ns[0], ns[1], ns[2], b4, b6, b8, bsc};
    mxmm_schedule schedule = {128, 4, 1};
    failed += mxmm_project_fp8(a, asc, M, 0, &w, out, workspace, schedule);
    mxmp_prepared p = {0};
    failed += mxmp_prepare_fp8(&p, a, asc, M, 0, &w, owned, sizeof owned, schedule, mix + 1, 1, 2);
    failed += mxmp_emit_fp8(&p, a, asc, M, 0, &w, prepared, schedule, mix + 1, 1);
    failed += mxmp_emit_fp8(&p, a, asc, M, 0, &w, prepared, schedule, mix + 1, 1);
    failed += mxmp_release(&p);
    mxmm_numeric_result x = mxmm_numeric_compare(out, ref, NULL, M * N),
                        y = mxmm_numeric_compare(prepared, ref, NULL, M * N);
    failed += !x.pass || !y.pass || x.bit_mismatches || y.bit_mismatches;
    printf(
        "MICROMIX_PROJECTION mix=%s words=%d canonical_bits=%llu prepared_bits=%llu checksum=%llu pass=%d\n",
        mix ? "mix625" : "mix50", M * N, (unsigned long long)x.bit_mismatches,
        (unsigned long long)y.bit_mismatches, (unsigned long long)checksum(out), !failed);
  }
  return failed != 0;
}
