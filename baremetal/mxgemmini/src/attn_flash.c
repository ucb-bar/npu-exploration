// Flash attention on MxGemmini: K/V in key blocks of BK, online softmax on the VPU, software-pipelined so the VPU
// works on block j while the mesh computes other blocks. Same data and goldens as attn_vpu (gen_attn_vpu.py).
//
// per block j (rows of S_j: SQ x BK):
//   mesh   S_j = Q @ K_j^T                                   (BF16, spad, double-buffered)
//   VPU    S_j *= 1/sqrt(d);  mt = rowmax(S_j);  m_j = max(m_{j-1}, mt);  a_j = exp(m_{j-1} - m_j)
//          S_j = exp(S_j - m_j);  lt = rowsum(S_j);  l = l * a_j + lt              (block 0: m = mt, l = lt)
//   SPAD_REQUANT  P_j -> E4M3 tiles + resident act scales (act half 0)
//   mesh   O_j = P_j @ V_j                                   (BF16, spad, double-buffered; block 0 writes O)
//   VPU    O = O * a_j + O_j                                 (j > 0)
// end: O *= 1/l.
//
// Scales: Q's in act half 1 once; every matmul's B-scale slice is a GATED load into weight half (n % 2) for the
// n-th matmul issued, applied by a MANAGED config -- the managed-half protocol orders them with no fence.
// Pass 1: serial (a fence after every stage; per-stage cycles). Pass 2: pipelined issue order
//   QK0 QK1 sm0 SR0 | PV0 sm1 SR1 ld3 QK2 | PV1 sm2 SR2 ld4 QK3 upd1 | ...   (mesh: PV(j) QK(j+2); VPU: sm(j+1) SR(j+1) upd(j))
// Both must produce the same O, equal to Spike's (hash), graded against fp64 attention.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "include/gemmini_testutils.h"
#include "include/vpu_ref.h"

#ifndef ATTN_HEADER
#define ATTN_HEADER "attn_vpu_fa.h"
#define ATTN_EXPECT "attn_flash_expect.h"
#endif
#include ATTN_HEADER
#ifndef BK
#define BK 64
#endif
#ifndef ATTN_CAUSAL
#define ATTN_CAUSAL 0   // 1 (from the data header): queries are the last SQ keys; causal mask on the last key block
#endif
// d a power of 4: 1/sqrt(d) = 2^-(log4 d) exactly -> fold it into Q's E8M0 scales (the mesh then produces S/sqrt(d)
// directly, bit-identical to the separate VPU multiply) and drop that pass; otherwise keep the MULS
#ifndef ATTN_FOLD_SCALE
#define ATTN_FOLD_SCALE (ATTN_D == 16 || ATTN_D == 64 || ATTN_D == 256)
#endif
#define FOLD_SHIFT (ATTN_D == 16 ? 2 : ATTN_D == 64 ? 3 : 4)

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: needs the VPU config (MX_ROCKET) or Spike\n"); return 0; }
#else

#ifndef ATTN_EXPSUB
#define ATTN_EXPSUB 1   // 0: softmax's subtract and exp as two VPU passes (before the fused op)
#endif
#define DIM 16
#undef BANK_ROWS
#define BANK_ROWS 4096
#define SQ ATTN_SQ
#define SK ATTN_SK
#define D  ATTN_D
#define NB (SK / BK)
#define OUT_BF16 3
#define SPAD_STORE 0x38
#define INC_ACC    0x100   // LOOP_WS rs2[8]: each loop takes the other accumulator half, so back-to-back loops never
                           // share accumulator rows (a loop's footprint: (M/16) * ceil4(N/16) * 4 rows <= 256)
#define ACC_ROWS(m, n) (((m) / DIM) * ((((n) / DIM) + 3) / 4 * 4) * 4)
typedef char flash_acc_half_fits[(ACC_ROWS(SQ, BK) <= 256 && ACC_ROWS(SQ, D) <= 256) ? 1 : -1];
typedef char flash_shape_ok[(SK % BK == 0 && BK % 32 == 0 && (SQ * BK / 32) % 32 == 0 && NB >= 2) ? 1 : -1];

#define ROWS8(m, n)  ((m) * (n) / DIM)
#define ROWS16(m, n) ((m) * (n) * 2 / DIM)
// K/V: all blocks resident when they fit, else STREAMED through two block buffers (loads run under the matmuls)
#define KV_BLK_ROWS (ROWS8(D, BK) + ROWS8(BK, D))
#define FIXED_ROWS  (ROWS8(SQ, D) + 2 * ROWS16(SQ, BK) + 2 * ROWS8(SQ, BK) + 2 * ROWS16(SQ, D) + ROWS16(SQ, D))
#ifndef ATTN_STREAM
#define ATTN_STREAM (FIXED_ROWS + NB * KV_BLK_ROWS > 0x3000)
#endif
// streamed: K double-buffered (its buffer's last reader, QK(j), precedes the softmax ahead of the load), V TRIPLE-
// buffered (V(j+2) reuses PV(j-1)'s buffer, already done) so a V load never waits on a running PV and never fills
// the RS load queue in front of the next block's VPU work
// O_LOW (split layout): O and the stats live in bank 0 instead of next to S, so while SPAD_REQUANT reads one S bank
// the VPU's O update and next softmax touch only the other S bank and bank 0 (they never share a bank and can run
// together every block). Bank 0 then has room for two V buffers, not three.
#ifndef ATTN_O_LOW_REQ
#define ATTN_O_LOW_REQ 0   // measured slower (119.1k vs 101.5k): VPU reads in bank 0 starve the mesh
#endif
#ifndef ATTN_V3
#define ATTN_V3 (!ATTN_O_LOW_REQ)   // 0: V double-buffered (V(j+2) is loaded into PV(j)'s buffer after it)
#endif
#define K_BUFS   (ATTN_STREAM ? 2 : NB)
#define V_BUFS   (ATTN_STREAM ? (ATTN_V3 ? 3 : 2) : NB)
// Bank split (when it fits): the mesh's operands in banks 0-1, each matmul's A and B in different banks (bank 0: Q, V;
// bank 1: K, P), and the VPU's (S, O_j, O, stats) in banks 2-3. The VPU read port always wins its bank, so sharing a
// bank starves the mesh's reads. S0 / O_0 in bank 2 and S1 / O_1 + O + stats in bank 3: softmax(j+1)'s writes never
// meet QK(j+2)'s S(j) or PV(j)'s O_j stores.
#ifndef ATTN_SPLIT
#define ATTN_SPLIT (ROWS8(SQ, D) + V_BUFS * ROWS8(BK, D) <= BANK_ROWS && \
                    K_BUFS * ROWS8(D, BK) + 2 * ROWS8(SQ, BK) <= BANK_ROWS && \
                    ROWS16(SQ, BK) + 2 * ROWS16(SQ, D) + 7 * SQ <= BANK_ROWS)
#endif
#define SP_Q     0
#if ATTN_SPLIT
#define SP_V     (SP_Q + ROWS8(SQ, D))
#define SP_KT    BANK_ROWS
#define SP_P     (SP_KT + K_BUFS * ROWS8(D, BK))       // 2 x ROWS8(SQ, BK)
#else
#define SP_KT    (SP_Q + ROWS8(SQ, D))
#define SP_V     (SP_KT + K_BUFS * ROWS8(D, BK))
#endif
#define ATTN_O_LOW (ATTN_SPLIT && ATTN_O_LOW_REQ && \
                    ROWS8(SQ, D) + V_BUFS * ROWS8(BK, D) + ROWS16(SQ, D) + 7 * SQ <= BANK_ROWS)
#if ATTN_SPLIT
#define S_BUF(j)  ((2 + ((j) & 1)) * BANK_ROWS)
#if ATTN_O_LOW
#define SP_O     (SP_V + V_BUFS * ROWS8(BK, D))       // bank 0, after V; the stats follow
#else
#define SP_O     (3 * BANK_ROWS + ROWS16(SQ, BK) + ROWS16(SQ, D))
#endif
#define OJ_BUF(j) ((j) == 0 ? SP_O : S_BUF(j) + ROWS16(SQ, BK))
#define SP_END   SP_O
#else
#define SP_S     (SP_V + V_BUFS * ROWS8(BK, D))       // 2 x ROWS16(SQ, BK)
#define SP_P     (SP_S + 2 * ROWS16(SQ, BK))           // 2 x ROWS8(SQ, BK)
#define SP_OJ    (SP_P + 2 * ROWS8(SQ, BK))            // 2 x ROWS16(SQ, D)
#define SP_END   (SP_OJ + 2 * ROWS16(SQ, D))
#define SP_O     (SP_END > 0x3000 ? SP_END : 0x3000)    // O in bank 3 when it fits: the O update reads O and O_j apart
#define S_BUF(j)  (SP_S + ((j) & 1) * ROWS16(SQ, BK))
#define OJ_BUF(j) ((j) == 0 ? SP_O : SP_OJ + ((j) & 1) * ROWS16(SQ, D))
#endif
#define ST       (SP_O + ROWS16(SQ, D))                // per-row stats after O (SQ rows each)
#define ST_M(b)  (ST + (b) * SQ)                        // running max, 2 buffers
#define ST_MT    (ST + 2 * SQ)
#define ST_A(b)  (ST + (3 + (b)) * SQ)                  // a_j = exp(m_{j-1} - m_j), 2 buffers
#define ST_L     (ST + 5 * SQ)
#define ST_LT    (ST + 6 * SQ)
typedef char flash_spad_fits[(SP_END <= SP_O && ST + 7 * SQ <= 4 * BANK_ROWS) ? 1 : -1];
#define P_BUF(j)  (SP_P + ((j) & 1) * ROWS8(SQ, BK))
// causal mask tile [SQ][SQ] BF16 (0 / -inf), added to the last SQ columns of the last block's S: in the bank the
// last S is not in (split layout), so the per-row ADD reads its two operands from different banks
#if ATTN_SPLIT && (ATTN_O_LOW || ((NB - 1) & 1))
#define SP_MASK  (S_BUF(NB) + ROWS16(SQ, BK) + ROWS16(SQ, D))   // after O_j in the bank the last S is not in
#else
#define SP_MASK  (ST + 7 * SQ)
#endif
typedef char flash_mask_fits[(!ATTN_CAUSAL || (SQ <= BK && SP_MASK + ROWS16(SQ, SQ) <= 4 * BANK_ROWS &&
  (!ATTN_SPLIT || !((NB - 1) & 1) || SP_MASK + ROWS16(SQ, SQ) <= 3 * BANK_ROWS))) ? 1 : -1];
#define KT_BLK(j) (SP_KT + (ATTN_STREAM ? ((j) & 1) : (j)) * ROWS8(D, BK))
#define V_BLK(j)  (SP_V + (ATTN_STREAM ? ((j) % V_BUFS) : (j)) * ROWS8(BK, D))

static uint16_t O_hw[SQ][D] __attribute__((aligned(64)));
static uint16_t O_ser[SQ][D];
static uint16_t st_hw[7 * SQ][8] __attribute__((aligned(64))), st_ser[7 * SQ][8];
static uint8_t  p_scales[2][SQ * BK / 32] __attribute__((aligned(64)));
static uint32_t scale_sink[512] __attribute__((aligned(64)));
static uint8_t  q_scales_folded[ATTN_D / 32][ATTN_SQ] __attribute__((aligned(64)));

int *__errno(void) { static int e; return &e; }
static inline float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint64_t fnv(const void *p, size_t n) {
  const uint64_t *w = (const uint64_t *)p; uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}

// tile (r,c) -> sp + (r*cols/16 + c)*16; up to 4 tiles per mvin (64-byte rows: 4x fewer RS load entries, faster DMA)
static void mvin_tiles(const uint8_t *src, int ld, int rows, int cols, uint32_t sp) {
  const int tc = cols / DIM, w = tc % 4 == 0 ? 4 : tc % 2 == 0 ? 2 : 1;
  gemmini_config_ld(ld);
  for (int r = 0; r < rows / DIM; r++)
    for (int c = 0; c < tc; c += w)
      gemmini_extended_mvin(src + r * DIM * ld + c * DIM, sp + (r * tc + c) * DIM, DIM * w, DIM);
}
static void mvout_rows(void *dst, uint32_t sp, int rows) {
  gemmini_config_st(DIM);
  for (int r = 0; r < rows; r += DIM) gemmini_extended_mvout((uint8_t *)dst + r * DIM, sp + r, DIM, DIM);
}

// ---- matmuls, with the managed weight-scale halves. A matmul is (kind, block); kind 0 = QK, 1 = PV ----
typedef struct { int kind, j; } mm_t;
static mm_t sched[2 * NB];
static int n_mm, mm_next;
static void gated_load(int n) {   // B-scale slice of the n-th matmul into weight half n % 2
  const mm_t m = sched[n];
  // braced: the gemmini_* macros expand to { ... } blocks
  if (m.kind == 0) { gemmini_mx_load_scales_2d_gated((uint64_t)&KT_SCALES[0][m.j * BK], BK, D / 32, SK, (n & 1) << 12, 1); }
  else             { gemmini_mx_load_scales_2d_gated((uint64_t)&V_SCALES[m.j * BK / 32][0], D, BK / 32, D, (n & 1) << 12, 1); }
}
static void issue_mm(void) {   // issue the next scheduled matmul
  const int n = mm_next++;
  const mm_t m = sched[n];
  const int M = SQ, K = m.kind ? BK : D, N = m.kind ? D : BK;
  // ex/store configs are the same for every matmul (spad stores carry their own row step): once per pass, so between
  // matmuls only the scale config and gated scale load sit outside a loop (both may pass a store-only loop)
  if (n == 0) {
    gated_load(0);
    gemmini_extended3_config_ex(WEIGHT_STATIONARY, 0, 0, ACC_SCALE_IDENTITY, 1, 1, 0, 0, false, 0, 0, OUT_BF16, 0);
    gemmini_config_st(N * sizeof(uint16_t));
  }
  gemmini_mxquant_config_mvout_managed((uint64_t)scale_sink, M / DIM, N / DIM, K / DIM, m.kind ? 0 : 1, n & 1, 1, 1);
  if (n + 1 < n_mm) gated_load(n + 1);   // waits for this config to free its half; lands under this matmul
  const uint32_t a = m.kind ? P_BUF(m.j) : SP_Q;
  const uint32_t b_arg = m.kind ? V_BLK(m.j) + ROWS8(BK, D) : KT_BLK(m.j) + ROWS8(D, BK);
  const uint32_t c = m.kind ? OJ_BUF(m.j) : S_BUF(m.j);
  gemmini_loop_ws_spad(M / DIM, N / DIM, K / DIM, 0, 0, 0, a, b_arg, 0, c,
                       false, false, false, false, false, NO_ACTIVATION, 0, 0, false, SPAD_STORE | INC_ACC);
}

// ---- VPU stages ----
static void softmax_block(int j) {
  const uint16_t sc = vpu_f_to_bf16(1.0f / sqrtf((float)D));
  const int rows = SQ * BK / 8, rlen = BK / 8;
  const uint32_t S = S_BUF(j), m = ST_M(j & 1), m_old = ST_M((j & 1) ^ 1);
  if (!ATTN_FOLD_SCALE) gemmini_vpu_scalar(VPU_MULS, S, S, sc, rows);
  if (ATTN_CAUSAL && j == NB - 1)   // query r's row of S: mask its last SQ keys (the chunk's own tokens)
    for (int r = 0; r < SQ; r++)
      gemmini_vpu_binary(VPU_ADD, S + r * (BK / 8) + (BK - SQ) / 8, S + r * (BK / 8) + (BK - SQ) / 8,
                         SP_MASK + r * (SQ / 8), SQ / 8);
  if (j == 0) {
    gemmini_vpu_reduce(VPU_RMAX, m, S, rows, rlen);
  } else {
    gemmini_vpu_reduce(VPU_RMAX, ST_MT, S, rows, rlen);
    gemmini_vpu_binary(VPU_MAX, m, m_old, ST_MT, SQ);
    gemmini_vpu_binary(VPU_SUB, ST_A(j & 1), m_old, m, SQ);
    gemmini_vpu_unary(VPU_EXP, ST_A(j & 1), ST_A(j & 1), SQ);
  }
#if ATTN_EXPSUB
  gemmini_vpu_bcast(VPU_EXPSUB, S, S, m, rows, rlen);   // exp(S - m) in one pass
#else
  gemmini_vpu_bcast(VPU_SUB, S, S, m, rows, rlen);
  gemmini_vpu_unary(VPU_EXP, S, S, rows);
#endif
  if (j == 0) {
    gemmini_vpu_reduce(VPU_RSUM, ST_L, S, rows, rlen);
  } else {
    gemmini_vpu_reduce(VPU_RSUM, ST_LT, S, rows, rlen);
    gemmini_vpu_binary(VPU_MUL, ST_L, ST_L, ST_A(j & 1), SQ);
    gemmini_vpu_binary(VPU_ADD, ST_L, ST_L, ST_LT, SQ);
  }
}
static void requant_block(int j) {
  gemmini_spad_requant(P_BUF(j), S_BUF(j), SQ, BK, 1, (uint64_t)p_scales[j & 1], 1);
}
static void update_block(int j) {   // O = O * a_j + O_j
  const int rows = SQ * D / 8;
  gemmini_vpu_bcast(VPU_MUL, SP_O, SP_O, ST_A(j & 1), rows, D / 8);
  gemmini_vpu_binary(VPU_ADD, SP_O, SP_O, OJ_BUF(j), rows);
}
static void finalize(void) {
  gemmini_vpu_unary(VPU_RCP, ST_L, ST_L, SQ);
  gemmini_vpu_bcast(VPU_MUL, SP_O, SP_O, ST_L, SQ * D / 8, D / 8);
  mvout_rows(O_hw, SP_O, ROWS16(SQ, D));
  mvout_rows(st_hw, ST, 7 * SQ);   // m x2, mt, a x2, 1/l, lt (debug)
}

// mismatches of O_hw vs the serial O per 16x16 tile, and which stats rows differ
static int diff_report(const char *name) {
  int tot = 0, tiles[SQ / 16][D / 16];
  memset(tiles, 0, sizeof(tiles));
  for (int m = 0; m < SQ; m++)
    for (int n = 0; n < D; n++)
      if (O_hw[m][n] != O_ser[m][n]) { tiles[m / 16][n / 16]++; tot++; }
  static const char *stn[7] = {"m0", "m1", "mt", "a0", "a1", "1/l", "lt"};
  printf("  %-10s O: %d/%d differ", name, tot, SQ * D);
  if (tot) {
    printf("; per 16x16 tile (row tiles x col tiles):");
    for (int i = 0; i < SQ / 16; i++) { printf(" |"); for (int k = 0; k < D / 16; k++) printf(" %d", tiles[i][k]); }
  }
  int sbad = 0;
  for (int g = 0; g < 7; g++) {
    int rows = 0;
    for (int r = 0; r < SQ; r++) rows += memcmp(st_hw[g * SQ + r], st_ser[g * SQ + r], 16) != 0;
    if (rows) { printf("%s %s: %d rows", sbad++ ? "," : "; stats", stn[g], rows); }
  }
  printf("\n");
  return tot != 0;
}

static void load_q(void) {
  mvin_tiles((const uint8_t *)Q_IN, D, SQ, D, SP_Q);
#if ATTN_CAUSAL
  mvin_tiles((const uint8_t *)MASK_BF16, DIM, ROWS16(SQ, SQ), DIM, SP_MASK);   // BF16 rows, 16 B each
#endif
}
static void load_k(int j) { if (j < NB) mvin_tiles(&KT_IN[0][j * BK], SK, D, BK, KT_BLK(j)); }
static void load_v(int j) { if (j < NB) mvin_tiles(&V_IN[j * BK][0], D, BK, D, V_BLK(j)); }
static void loads(void) {   // everything up front (resident K/V)
  load_q();
  for (int j = 0; j < NB; j++) { load_k(j); load_v(j); }
}

// stage cycle totals for the serial pass: loads, QK, softmax, requant, PV, update, finalize
static uint64_t stage[7];
static uint64_t tmark;
static void fence_into(int s) { gemmini_fence(); uint64_t t = read_cycles(); stage[s] += t - tmark; tmark = t; }

static uint64_t run_serial(void) {
  n_mm = 0; mm_next = 0;
  for (int j = 0; j < NB; j++) { sched[n_mm++] = (mm_t){0, j}; sched[n_mm++] = (mm_t){1, j}; }
  memset(stage, 0, sizeof(stage));
  gemmini_fence();
  const uint64_t t0 = read_cycles(); tmark = t0;
  if (ATTN_STREAM) load_q(); else loads();
  fence_into(0);
  for (int j = 0; j < NB; j++) {
    if (ATTN_STREAM) { load_k(j); load_v(j); fence_into(0); }
    issue_mm();                 fence_into(1);   // QK_j
    softmax_block(j);         fence_into(2);
    requant_block(j);         fence_into(3);
    issue_mm();                 fence_into(4);   // PV_j
    if (j > 0) { update_block(j); fence_into(5); }
  }
  finalize();                 fence_into(6);
  return read_cycles() - t0;
}

// fmask: fences inserted into the pipelined order, to bisect a broken overlap:
//   1 after each softmax, 2 after each requant, 4 after each QK, 8 after each PV, 16 after each O update
static int fmask;
#define FENCE_IF(bit) do { if (fmask & (bit)) gemmini_fence(); } while (0)
static uint64_t run_pipelined(void) {
  n_mm = 0; mm_next = 0;
  sched[n_mm++] = (mm_t){0, 0}; sched[n_mm++] = (mm_t){0, 1};
  for (int j = 0; j < NB; j++) { sched[n_mm++] = (mm_t){1, j}; if (j + 2 < NB) sched[n_mm++] = (mm_t){0, j + 2}; }
  gemmini_fence();
  const uint64_t t0 = read_cycles();
  if (ATTN_STREAM) { load_q(); load_k(0); load_k(1); for (int v = 0; v < V_BUFS; v++) load_v(v); } else loads();
  issue_mm(); FENCE_IF(4); issue_mm(); FENCE_IF(4);   // QK0, QK1
  softmax_block(0); FENCE_IF(1); requant_block(0); FENCE_IF(2);
  if (ATTN_STREAM) load_k(2);                    // K buffer 0 (QK0 is done)
  // softmax for block j+1 is issued right behind PV(j), so it bypasses the unrolling PV(j) / QK(j+2) loops
  for (int j = 0; j < NB; j++) {
    issue_mm(); FENCE_IF(8);                     // PV(j)
    // SR(j+1) ahead of QK(j+2): QK's S stores wait for it on the requantizer, but PV(j+1) no longer waits for them
    // (QK's managed scale config reads act half 1, so it does not wait for the SR's resident act-half-0 scales)
    if (j + 1 < NB) { softmax_block(j + 1); FENCE_IF(1); requant_block(j + 1); FENCE_IF(2); }
    // loads before QK(j+2): they wait only for PV(j) to unroll, not for QK(j+2)'s stores, so PV(j+1) is not behind them
    if (ATTN_STREAM) { load_k(j + 3); load_v(j + V_BUFS); }   // into the buffers QK(j+1) / PV(j) used
    if (j + 2 < NB) { issue_mm(); FENCE_IF(4); } // QK(j+2)
    if (j > 0) { update_block(j); FENCE_IF(16); }
  }
  finalize();
  gemmini_fence();
  return read_cycles() - t0;
}

int main() {
  printf("attn_flash: Sq=%d Sk=%d d=%d, Bk=%d (%d key blocks), online softmax on the VPU, K/V %s\n", SQ, SK, D, BK, NB,
         !ATTN_STREAM ? "resident" : ATTN_V3 ? "streamed (K x2, V x3 block buffers)" : "streamed (K x2, V x2 block buffers)");
  printf("mask: %s\n", ATTN_CAUSAL ? "causal (queries are the last Sq keys)" : "none (every key precedes every query)");
  printf("softmax: %s\n", ATTN_EXPSUB ? "fused EXPSUB (exp(S - m) in one pass)" : "SUB then EXP");
  printf("spad layout: %s\n", !ATTN_SPLIT ? "packed" : ATTN_O_LOW ?
         "bank split, O + stats in bank 0 (S / O_j in banks 2-3)" : "bank split (mesh operands banks 0-1, VPU buffers banks 2-3)");
  gemmini_flush(0);
  const uint8_t *q_sc = &Q_SCALES[0][0];
  if (ATTN_FOLD_SCALE) {   // Q * 2^-log4(d): every E8M0 byte - log4(d)
    for (int g = 0; g < D / 32; g++)
      for (int m = 0; m < SQ; m++) q_scales_folded[g][m] = Q_SCALES[g][m] - FOLD_SHIFT;
    q_sc = &q_scales_folded[0][0];
  }
  printf("1/sqrt(d): %s\n", ATTN_FOLD_SCALE ? "folded into Q's scales (no VPU pass)" : "VPU multiply");
  gemmini_mx_load_scales_2d((uint64_t)q_sc, SQ, D / 32, SQ, 4096, 0);   // Q -> act half 1
  gemmini_fence();

  memset(O_hw, 0xa5, sizeof(O_hw));
  const uint64_t c_ser = run_serial();
  const uint64_t h_ser = fnv(O_hw, sizeof(O_hw));
  memcpy(O_ser, O_hw, sizeof(O_ser)); memcpy(st_ser, st_hw, sizeof(st_ser));
  memset(O_hw, 0xa5, sizeof(O_hw));
  fmask = 0;
  const uint64_t c_pip = run_pipelined();
  const uint64_t h_pip = fnv(O_hw, sizeof(O_hw));
  int pip_bad = diff_report("pipelined");
  if (pip_bad) {   // bisect: which fence restores the serial result
    static const int masks[6] = {1, 2, 4, 8, 16, 31};
    static const char *mn[6] = {"+f softmax", "+f requant", "+f QK", "+f PV", "+f update", "+f all"};
    for (int i = 0; i < 6; i++) {
      memset(O_hw, 0xa5, sizeof(O_hw));
      fmask = masks[i];
      const uint64_t c = run_pipelined();
      printf("  bisect %-11s %6llu cycles:", mn[i], (unsigned long long)c);
      diff_report("");
    }
  }

  int fail = h_ser != h_pip;
  printf("check O hash: serial %016llx, pipelined %016llx (%s)\n", (unsigned long long)h_ser,
         (unsigned long long)h_pip, h_ser == h_pip ? "equal" : "DIFFER");
#ifdef ATTN_EXPECT
#include ATTN_EXPECT
  printf("check O hash vs Spike: %s\n", h_pip == EXP_FLASH_HASH_O ? "match" : "MISMATCH");
  fail |= h_pip != EXP_FLASH_HASH_O;
#endif
  float num = 0, den = 0;   // accuracy over the first 16 rows (O_REF_F: fp64 dense attention)
  for (int i = 0; i < 16 * D; i++) {
    float r; memcpy(&r, &((const uint32_t *)O_REF_F_F32)[i], 4);
    const float o = bf(((const uint16_t *)O_hw)[i]);
    num += (o - r) * (o - r); den += r * r;
  }
  printf("accuracy O (16 rows): rel_fro %d ppm vs fp64 attention\n", (int)(sqrtf(num / den) * 1e6f));

  const uint64_t mesh = 2ULL * SQ * SK * D / (DIM * DIM);
  printf("PERF serial    %6llu cycles: loads %llu | QK %llu | softmax %llu | requant %llu | PV %llu | O update %llu | final %llu\n",
         (unsigned long long)c_ser, (unsigned long long)stage[0], (unsigned long long)stage[1], (unsigned long long)stage[2],
         (unsigned long long)stage[3], (unsigned long long)stage[4], (unsigned long long)stage[5], (unsigned long long)stage[6]);
  printf("PERF pipelined %6llu cycles (%llu%% of serial); mesh ideal %llu -> util %llu%% (serial %llu%%)\n",
         (unsigned long long)c_pip, (unsigned long long)(c_pip * 100 / c_ser), (unsigned long long)mesh,
         (unsigned long long)(mesh * 100 / c_pip), (unsigned long long)(mesh * 100 / c_ser));
  printf("attn_flash %s\n", fail ? "FAILED" : "PASSED");
  return fail;
}
#endif
