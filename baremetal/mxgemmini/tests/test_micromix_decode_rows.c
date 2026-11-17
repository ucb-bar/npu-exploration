/* Exact row helper/packer; native calls explicitly recorded, never executed. */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
static void rows_trace_fence(void);
#define MXMM_TEST_FENCE() rows_trace_fence()
#include "host_shim.h"
static unsigned prologue[16], prologue_count;
static void rows_trace_fence(void) {
  if (prologue_count < 4)
    prologue[prologue_count++] = 99;
}
static void cfg_record(unsigned a, unsigned b) {
  if (prologue_count < 16)
    prologue[prologue_count++] = 10 * a + b;
  device_calls++;
}
#define gemmini_mxquant_config_mvout(P, A, B, C, H1, H2, Z) cfg_record(H1, H2)
typedef struct {
  int format, i, j, k, as, bs, cs, ap, bp, acc, h1, h2;
  const void *a, *b, *c, *sa, *sb;
} command;
static command commands[8192];
static int count, current_code;
static void record(int single, int i, int j, int k, const void *a, const void *b, const void *c,
                   int as, int bs, int cs, const void *sa, const void *sb, int ap, int bp, int acc,
                   int h1, int h2) {
  if (count >= 8192)
    exit(91);
  device_calls++;
  commands[count++] = (command){single ? (current_code == 1 ? 6 : 8) : 4,
                                i,
                                j,
                                k,
                                as,
                                bs,
                                cs,
                                ap,
                                bp,
                                acc,
                                h1,
                                h2,
                                a,
                                b,
                                c,
                                sa,
                                sb};
}
#undef gemmini_extended3_config_ex
#define gemmini_extended3_config_ex(a, b, c, d, e, f, g, h, z, ca, cb, s, t)                       \
  do {                                                                                             \
    current_code = (ca);                                                                           \
    device_calls++;                                                                                \
  } while (0)
#undef gemmini_loop_ws_mx
#undef gemmini_loop_ws_mx_shared
#define gemmini_loop_ws_mx(I, J, K, A, B, C, AS, BS, CS, SA, SB, AP, BP, ACC, H1, H2)              \
  record(0, I, J, K, A, B, C, AS, BS, CS, SA, SB, AP, BP, ACC, H1, H2);
#define gemmini_loop_ws_mx_shared(I, J, K, A, B, C, AS, BS, CS, SA, SB, AP, BP, ACC, H1, H2)       \
  record(1, I, J, K, A, B, C, AS, BS, CS, SA, SB, AP, BP, ACC, H1, H2);
static unsigned canonical_calls;
static int mxmm_project_bf16(const uint16_t *a, int rows, int m, const mxmm_weight *w,
                             uint16_t *out, uint8_t *scratch, mxmm_schedule schedule) {
  (void)a;
  (void)rows;
  (void)m;
  (void)w;
  (void)out;
  (void)scratch;
  (void)schedule;
  canonical_calls++;
  return 123;
}
#include "mx_micromix_decode_rows.h"
static unsigned checks, failures, bytes, shape_checks;
static void check(int value) {
  checks++;
  if (!value)
    failures++;
}
static void *alloc64(size_t n) {
  void *p;
  if (posix_memalign(&p, 64, n))
    exit(92);
  memset(p, 0xa5, n);
  return p;
}
static void reference_pack(const uint16_t *a, int rows, int k, int *ns, uint8_t *out, uint8_t *sc) {
  size_t offset = 0;
  int kb = 0;
  const int fmts[] = {4, 6, 8};
  for (int g = 0; g < 3; g++) {
    int kg = ns[g] * 32;
    for (int r = 0; r < 32; r++)
      for (int b = 0; b < ns[g]; b++) {
        int e = 0;
        float am = 0;
        if (r < rows) {
          for (int j = 0; j < 32; j++) {
            uint32_t bits = (uint32_t)a[r * k + (kb + b) * 32 + j] << 16;
            float x;
            memcpy(&x, &bits, 4);
            if (fabsf(x) > am)
              am = fabsf(x);
          }
          if (am > 0) {
            frexpf(am, &e);
            e--;
          }
        }
        sc[(kb + b) * 32 + r] = e + 127;
        for (int j = 0; j < 32; j++) {
          float x = 0;
          if (r < rows) {
            uint32_t bits = (uint32_t)a[r * k + (kb + b) * 32 + j] << 16;
            memcpy(&x, &bits, 4);
          }
          uint8_t v = mxmm_encode(ldexpf(x, -e), fmts[g]);
          size_t at = offset + (size_t)(g == 0 ? r / 2 : r) * kg + b * 32 + j;
          if (g == 0) {
            if (r & 1)
              out[at] |= v << 4;
            else
              out[at] = v;
          } else
            out[at] = v;
        }
      }
    offset += (size_t)32 * kg / (g == 0 ? 2 : 1);
    kb += ns[g];
  }
}
int main(void) {
  int shapes[][2] = {{3584, 3584}, {3584, 512}, {3584, 18944}, {18944, 3584}};
  for (int mix = 0; mix < 2; mix++)
    for (int ri = 0; ri < 2; ri++)
      for (int shape = 0; shape < 4; shape++) {
        int rows = ri ? 16 : 1, k = shapes[shape][0], n = shapes[shape][1], nb = k / 32,
            ns[] = {mix ? nb * 5 / 8 : nb / 2, nb / 4, mix ? nb / 8 : nb / 4};
        uint16_t *a = alloc64((size_t)rows * k * 2), *out = alloc64((size_t)32 * n * 2 + 128);
        uint8_t *scratch = alloc64(mxdr_band_workspace_bytes(k) + 128),
                *expected = alloc64(mxdr_band_workspace_bytes(k) + 128), *b[3];
        uint8_t *sc = alloc64((size_t)nb * n);
        for (int g = 0; g < 3; g++)
          b[g] = alloc64((size_t)ns[g] * 32 * n / (g == 0 ? 2 : 1));
        for (int r = 0; r < rows; r++)
          for (int j = 0; j < k; j++) {
            unsigned x = r * 7349u + j * 613u + 19;
            a[r * k + j] =
                x % 23 == 0
                    ? (x & 1 ? 0x8000 : 0)
                    : (uint16_t)((x & 1 ? 0x8000 : 0) | ((119 + x % 15) << 7) | ((x >> 4) & 127));
          }
        mxmm_weight w = {k, n, ns[0], ns[1], ns[2], b[0], b[1], b[2], sc};
        mxmm_schedule schedule = {256, 8, 1};
        reference_pack(a, rows, k, ns, expected, expected + (size_t)32 * k);
        count = 0;
        prologue_count = 0;
        int initial_loops = mxn_loops;
        check(mxmm_decode_project_bf16(a, rows, &w, out, scratch, schedule,
                                       MXMM_DECODE_ISSUE_MIXED16) == 0);
        check(prologue_count == 4 && prologue[0] == 11 && prologue[1] == 99 && prologue[2] == 0 &&
              prologue[3] == 99);
        check(memcmp(scratch, expected, mxdr_band_workspace_bytes(k)) == 0);
        bytes += mxdr_band_workspace_bytes(k);
        int at = 0, kb = 0;
        size_t off = 0;
        for (int col = 0; col < n; col += 256) {
          kb = 0;
          off = 0;
          for (int g = 0; g < 3; g++) {
            int kg = ns[g] * 32, half = g == 0 ? 2 : 1;
            for (int block = 0; block < ns[g]; block += 8) {
              int run = ns[g] - block < 8 ? ns[g] - block : 8;
              command *c = &commands[at];
              check(c->format == (g == 0   ? 4
                                  : g == 1 ? 6
                                           : 8) &&
                    c->i == 1 && c->j == 256 / (g == 0 ? 32 : 16) && c->k == run * 2);
              check(c->as == kg && c->bs == n / half && c->cs == n && c->ap == 32 && c->bp == n &&
                    c->acc == !(g == 0 && !block));
              check(c->a == scratch + off + block * 32 &&
                    c->b == b[g] + (size_t)block * 32 * (n / half) + col / half);
              check(c->sa == scratch + (size_t)32 * k + (kb + block) * 32 &&
                    c->sb == sc + (size_t)(kb + block) * n + col);
              check(c->c == (g == 2 && block + run == ns[g] ? (void *)(out + col) : NULL));
              check(c->h1 == 1 + ((initial_loops + at) & 1) && c->h2 == c->h1);
              at++;
              shape_checks++;
            }
            off += (size_t)32 * kg / half;
            kb += ns[g];
          }
        }
        check(at == count);
        for (int j = 0; j < 64; j++)
          check(scratch[mxdr_band_workspace_bytes(k) + j] == 0xa5);
        /* Unsupported rows/FP8/fences must reject before packing/native side effects. */
        for (int guard = 0; guard < 12; guard++) {
          mxmm_weight bad = w;
          mxmm_schedule s = schedule;
          int nr = rows;
          if (guard == 0)
            nr = 17;
          if (guard == 1) {
            bad.nb4 = bad.nb6 = 0;
            bad.nb8 = nb;
          }
          if (guard == 2)
            s.fence_groups = 0;
          if (guard == 3)
            nr = 0;
          const uint16_t *ap = a;
          uint16_t *op = out;
          uint8_t *sp = scratch;
          const mxmm_weight *wp = &bad;
          if (guard == 4)
            ap = NULL;
          if (guard == 5)
            op = NULL;
          if (guard == 6)
            sp = NULL;
          if (guard == 7)
            wp = NULL;
          if (guard == 8)
            bad.scales = NULL;
          if (guard == 9)
            bad.b6 = NULL;
          if (guard == 10)
            op = out + 1;
          if (guard == 11)
            bad.b8 = b[2] + 1;
          unsigned calls = device_calls, prologues = prologue_count;
          uint64_t ticks = host_ticks;
          uint64_t pc = mxmm_pack_cycles, mc = mxmm_mesh_cycles;
          int loops = mxn_loops;
          check(mxmm_decode_project_bf16(ap, nr, wp, op, sp, s, MXMM_DECODE_ISSUE_MIXED16) == 1);
          check(calls == device_calls && prologues == prologue_count && ticks == host_ticks &&
                pc == mxmm_pack_cycles && mc == mxmm_mesh_cycles && loops == mxn_loops);
        }
        /* The selector is independent of attention and does not auto-enable. */
        unsigned calls = device_calls, canon = canonical_calls;
        uint64_t ticks = host_ticks;
        check(mxmm_decode_project_bf16(a, rows, &w, out, scratch, schedule,
                                       (mxmm_decode_issue_mode)99) == 1);
        check(device_calls == calls && host_ticks == ticks && canonical_calls == canon);
        check(mxmm_decode_project_bf16(a, rows, &w, out, scratch, schedule,
                                       MXMM_DECODE_ISSUE_M32) == 123);
        check(canonical_calls == canon + 1 && device_calls == calls && host_ticks == ticks);
        free(a);
        free(out);
        free(scratch);
        free(expected);
        free(sc);
        for (int g = 0; g < 3; g++)
          free(b[g]);
      }
  printf(
      "MX_PUBLIC_ROWS_HOST pass=%d failures=%u checks=%u packed_bytes=%u command_shapes=%u native_calls_recorded_only=1 matmul_numerics_claim=0 timing_usable=0\n",
      failures == 0, failures, checks, bytes, shape_checks);
  return failures ? 1 : 0;
}
