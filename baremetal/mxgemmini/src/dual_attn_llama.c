// dual_attn_llama -- flash attention (E4M3 x E4M3, online softmax on the VPU) for one GQA KV group, on both Gemminis of
// the dual SoC: 16x16 (custom3) and 8x8 (custom2). One pass = one query head (Sq 64) over the whole KV cache.
//   head split (default): passes [0, DA_SPLIT) on the 16x16, [DA_SPLIT, NPASS) on the 8x8; one fence at the end.
//   key split (DA_KSPLIT): every head on both -- keys [0, DA_SKA) on the 16x16, [DA_SKA, SK) on the 8x8, unnormalized
//     partials to DRAM -- then one fence, the 16x16's VPU merges each head, and a final fence.
// The CPU only issues: it interleaves the two instances' key blocks (less estimated issued work goes next).
// Each head's O is checked against the golden hash (gen/golden_dual_attn.py / golden_dual_attn_split.py).
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#ifndef DA_DATA
#define DA_DATA "attn_vpu_dual_kv0.h"
#endif
#include DA_DATA
#include "dual_attn_api.h"
#ifndef DA_KSPLIT
#define DA_KSPLIT 0
#endif
#if DA_KSPLIT
#include DA_EXPECT   // EXP_SPLIT_{O,O16,M16,L16,O8,M8,L8}[NPASS]
#else
#include "attn_dual_kv0_d16_expect.h"
#include "attn_dual_kv0_d8_expect.h"
#endif

#ifndef DA_SPLIT
#define DA_SPLIT 7
#endif
#ifndef DA_ONLY
#define DA_ONLY 0   // key split: 16 / 8 = only that Gemmini issues its key range (isolated timing; partials checked, no merge)
#endif
#define SQ ATTN_SQ
#define SK ATTN_SK
#define D ATTN_D
#if ATTN_SCALE_IN_Q
#define FOLD 0   // Q already carries 1/sqrt(d)
#else
#define FOLD (D == 16 ? 2 : D == 64 ? 3 : 4)   // 1/sqrt(d) = 2^-FOLD, folded into Q's E8M0
#endif

static inline uint64_t rdcycle(void) { uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
static inline void fence_all(void) { asm volatile("fence"); }
int *__errno(void) { static int e; return &e; }

static uint8_t q_fold[ATTN_NPASS][D / 32][SQ] __attribute__((aligned(64)));
static uint16_t o_out[ATTN_NPASS][SQ][D] __attribute__((aligned(64)));
#if DA_KSPLIT
static uint16_t o_part[2][ATTN_NPASS][SQ][D] __attribute__((aligned(64)));
static uint16_t m_part[2][ATTN_NPASS][SQ][8] __attribute__((aligned(64)));
static uint16_t l_part[2][ATTN_NPASS][SQ][8] __attribute__((aligned(64)));
#endif

static uint64_t fnv(const void *p, size_t n) {
  const uint64_t *w = (const uint64_t *) p;
  uint64_t h = 1469598103934665603ULL;
  for (size_t i = 0; i < n / 8; i++) { h ^= w[i]; h *= 1099511628211ULL; }
  return h;
}

typedef int (*step_fn)(void);


int main() {
  for (int p = 0; p < ATTN_NPASS; p++)
    for (int g = 0; g < D / 32; g++)
      for (int m = 0; m < SQ; m++) q_fold[p][g][m] = LQ_SCALES[p][g][m] - FOLD;
  memset(o_out, 0xa5, sizeof(o_out));
  da_bufs_t b = {&LQ_IN[0][0][0], &q_fold[0][0][0], &LKT_IN[0][0][0], &LKT_SCALES[0][0][0], &LV_IN[0][0][0],
                 &LV_SCALES[0][0][0], &o_out[0][0][0]};
#if DA_KSPLIT
  for (int i = 0; i < 2; i++) { b.o_part[i] = &o_part[i][0][0][0]; b.m_part[i] = &m_part[i][0][0][0]; b.l_part[i] = &l_part[i][0][0][0]; }
  const int first[2] = {0, 0}, np[2] = {DA_ONLY == 8 ? 0 : ATTN_NPASS, DA_ONLY == 16 ? 0 : ATTN_NPASS};
#else
  const int first[2] = {0, DA_SPLIT}, np[2] = {DA_SPLIT, ATTN_NPASS - DA_SPLIT};
#endif
  da_info_t inf[2];
  da_info_g16(&inf[0]);
  da_info_g8(&inf[1]);
  for (int i = 0; i < 2; i++)
    printf("gemmini %dx%d (custom%d): %d pass(es), keys [%d:%d], Bk=%d (%d blocks), K/V %s, spad %s; ideal %llu cycles/pass\n",
           inf[i].dim, inf[i].dim, inf[i].opcode, np[i], inf[i].k0, inf[i].k0 + inf[i].skn, inf[i].bk, inf[i].nb,
           inf[i].stream ? "streamed" : "resident", inf[i].split ? "bank split" : "packed",
           (unsigned long long) inf[i].pass_ideal);
  da_init_g16(&b, first[0], np[0]);
  da_init_g8(&b, first[1], np[1]);
  step_fn step[2] = {da_step_g16, da_step_g8};
  fence_all();

  const uint64_t t0 = rdcycle();
  uint64_t issued[2] = {0, 0};
  int done[2] = {np[0] == 0, np[1] == 0}, steps[2] = {0, 0};
  // credit-based: the Gemmini with less estimated work issued goes next if its queue can take the step whole, else
  // the other one; if neither has room, re-query (QUEUE_FREE never blocks)
  int (*room[2])(void) = {da_room_g16, da_room_g8};
  uint64_t spins = 0;
  while (!done[0] || !done[1]) {
    const int pref = done[1] ? 0 : done[0] ? 1 : (issued[0] <= issued[1] ? 0 : 1);
    const int g = room[pref]() ? pref : (!done[pref ^ 1] && room[pref ^ 1]()) ? (pref ^ 1) : -1;
    if (g < 0) { spins++; continue; }
    const int r = step[g]();
    if (r < 0) { done[g] = 1; continue; }
    issued[g] += r;
    steps[g]++;
  }
  const uint64_t t_issue = rdcycle();
  fence_all();
  const uint64_t t1 = rdcycle();
#if DA_KSPLIT && !DA_ONLY
  da_merge_g16(ATTN_NPASS);
  fence_all();
#endif
  const uint64_t t2 = rdcycle();

  int bad = 0;
  for (int p = 0; p < ATTN_NPASS; p++) {
    const uint64_t h = fnv(o_out[p], sizeof(o_out[p]));
#if DA_KSPLIT && DA_ONLY
    const int x = DA_ONLY == 16 ? 0 : 1;
    const int ok = fnv(o_part[x][p], sizeof(o_part[x][p])) == (x ? EXP_SPLIT_O8 : EXP_SPLIT_O16)[p] &&
                   fnv(m_part[x][p], sizeof(m_part[x][p])) == (x ? EXP_SPLIT_M8 : EXP_SPLIT_M16)[p] &&
                   fnv(l_part[x][p], sizeof(l_part[x][p])) == (x ? EXP_SPLIT_L8 : EXP_SPLIT_L16)[p];
    printf("head %d partial on %s only: O / m / l %s\n", LPASS_HEAD0[p], x ? "8x8" : "16x16", ok ? "== golden" : "DIFFER");
    const uint64_t e = h;
    (void) e;
    bad += !ok;
    continue;
#elif DA_KSPLIT
    const uint64_t e = EXP_SPLIT_O[p];
    printf("head %d (both, merged on 16x16): O %016llx %s", LPASS_HEAD0[p], (unsigned long long) h, h == e ? "== golden" : "DIFFERS");
    if (h != e) {   // localize: which partial is off
      printf(" | O16 %s m16 %s l16 %s O8 %s m8 %s l8 %s",
             fnv(o_part[0][p], sizeof(o_part[0][p])) == EXP_SPLIT_O16[p] ? "ok" : "BAD",
             fnv(m_part[0][p], sizeof(m_part[0][p])) == EXP_SPLIT_M16[p] ? "ok" : "BAD",
             fnv(l_part[0][p], sizeof(l_part[0][p])) == EXP_SPLIT_L16[p] ? "ok" : "BAD",
             fnv(o_part[1][p], sizeof(o_part[1][p])) == EXP_SPLIT_O8[p] ? "ok" : "BAD",
             fnv(m_part[1][p], sizeof(m_part[1][p])) == EXP_SPLIT_M8[p] ? "ok" : "BAD",
             fnv(l_part[1][p], sizeof(l_part[1][p])) == EXP_SPLIT_L8[p] ? "ok" : "BAD");
    }
    printf("\n");
#else
    const int on8 = p >= DA_SPLIT;
    const uint64_t e = on8 ? EXP_DUAL_HASH_O_D8[p] : EXP_DUAL_HASH_O_D16[p];
    printf("pass %d (head %d) on %s: O hash %016llx %s\n", p, LPASS_HEAD0[p], on8 ? "8x8  " : "16x16",
           (unsigned long long) h, h == e ? "== golden" : "DIFFERS");
#endif
    bad += h != e;
  }
  const uint64_t tot = t2 - t0, macs = 2ull * ATTN_NPASS * SQ * SK * D;
  const uint64_t ideal = macs / (16 * 16 + 8 * 8);   // both meshes at E4M3 peak (320 MACs / cycle)
  printf("PERF total %llu cycles (passes %llu, merge %llu); ideal %llu (both meshes at peak) -> util %llu.%llu%%; "
         "CPU issue done at %llu; steps %d + %d\n", (unsigned long long) tot, (unsigned long long) (t1 - t0),
         (unsigned long long) (t2 - t1), (unsigned long long) ideal, (unsigned long long) (ideal * 100 / tot),
         (unsigned long long) (ideal * 1000 / tot % 10), (unsigned long long) (t_issue - t0), steps[0], steps[1]);
  printf("PERF issue: %llu QUEUE_FREE spins (both queues too full for their next step)\n", (unsigned long long) spins);
  printf("PERF per gemmini ideal: 16x16 %llu, 8x8 %llu cycles\n",
         (unsigned long long) (np[0] * inf[0].pass_ideal), (unsigned long long) (np[1] * inf[1].pass_ideal));
  printf("dual_attn_llama %s (%d/%d heads match)\n", bad ? "FAILED" : "PASSED", ATTN_NPASS - bad, ATTN_NPASS);
  return bad != 0;
}
