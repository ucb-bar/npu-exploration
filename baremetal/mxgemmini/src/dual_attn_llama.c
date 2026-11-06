// dual_attn_llama -- flash attention (E4M3 x E4M3, online softmax on the VPU) for one TinyLlama KV group -- layer 5,
// query heads 0-7 over kv head 0, 64 queries x 2048 keys, d = 64 -- split across both Gemminis of the dual SoC:
// passes [0, DA_SPLIT) on the 16x16 (custom3), [DA_SPLIT, 8) on the 8x8 (custom2). One pass = one query head.
// The CPU only issues: it interleaves the two instances' key blocks (less estimated issued work goes next) and fences
// once, at the end. Each pass's O is checked against golden_dual_attn.py's hash at the DIM that ran it.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "attn_vpu_dual_kv0.h"
#include "attn_dual_kv0_d16_expect.h"
#include "attn_dual_kv0_d8_expect.h"
#include "dual_attn_api.h"

#ifndef DA_SPLIT
#define DA_SPLIT 7
#endif
#define SQ ATTN_SQ
#define SK ATTN_SK
#define D ATTN_D
#define FOLD 3   // d = 64: 1/sqrt(d) = 2^-3, folded into Q's E8M0

static inline uint64_t rdcycle(void) { uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
static inline void fence_all(void) { asm volatile("fence"); }
int *__errno(void) { static int e; return &e; }

static uint8_t q_fold[ATTN_NPASS][D / 32][SQ] __attribute__((aligned(64)));
static uint16_t o_out[ATTN_NPASS][SQ][D] __attribute__((aligned(64)));

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
  da_info_t inf[2];
  da_info_g16(&inf[0]);
  da_info_g8(&inf[1]);
  const int np[2] = {DA_SPLIT, ATTN_NPASS - DA_SPLIT};
  for (int i = 0; i < 2; i++)
    printf("gemmini %dx%d (custom%d): %d pass(es), Bk=%d (%d blocks), K/V %s, spad %s; ideal %llu cycles/pass\n",
           inf[i].dim, inf[i].dim, inf[i].opcode, np[i], inf[i].bk, inf[i].nb, inf[i].stream ? "streamed" : "resident",
           inf[i].split ? "bank split" : "packed", (unsigned long long) inf[i].pass_ideal);
  da_init_g16(&b, 0, np[0]);
  da_init_g8(&b, DA_SPLIT, np[1]);
  step_fn step[2] = {da_step_g16, da_step_g8};
  fence_all();

  const uint64_t t0 = rdcycle();
  uint64_t issued[2] = {0, 0};
  int done[2] = {np[0] == 0, np[1] == 0}, steps[2] = {0, 0};
  while (!done[0] || !done[1]) {
    const int g = done[1] ? 0 : done[0] ? 1 : (issued[0] <= issued[1] ? 0 : 1);
    const int r = step[g]();
    if (r < 0) { done[g] = 1; continue; }
    issued[g] += r;
    steps[g]++;
  }
  const uint64_t t_issue = rdcycle();
  fence_all();
  const uint64_t t1 = rdcycle();

  int bad = 0;
  for (int p = 0; p < ATTN_NPASS; p++) {
    const int on8 = p >= DA_SPLIT;
    const uint64_t h = fnv(o_out[p], sizeof(o_out[p])), e = on8 ? EXP_DUAL_HASH_O_D8[p] : EXP_DUAL_HASH_O_D16[p];
    printf("pass %d (head %d) on %s: O hash %016llx %s\n", p, LPASS_HEAD0[p], on8 ? "8x8  " : "16x16",
           (unsigned long long) h, h == e ? "== golden" : "DIFFERS");
    bad += h != e;
  }
  const uint64_t tot = t1 - t0, macs = 2ull * ATTN_NPASS * SQ * SK * D;
  const uint64_t ideal = macs / (16 * 16 + 8 * 8);   // both meshes at E4M3 peak (320 MACs / cycle)
  printf("PERF total %llu cycles; ideal %llu (both meshes at peak) -> util %llu.%llu%%; CPU issue done at %llu; "
         "steps %d + %d\n", (unsigned long long) tot, (unsigned long long) ideal,
         (unsigned long long) (ideal * 100 / tot), (unsigned long long) (ideal * 1000 / tot % 10),
         (unsigned long long) (t_issue - t0), steps[0], steps[1]);
  printf("PERF per gemmini ideal: 16x16 %llu, 8x8 %llu cycles\n",
         (unsigned long long) (np[0] * inf[0].pass_ideal), (unsigned long long) (np[1] * inf[1].pass_ideal));
  printf("dual_attn_llama %s (%d/%d passes match)\n", bad ? "FAILED" : "PASSED", ATTN_NPASS - bad, ATTN_NPASS);
  return bad != 0;
}
