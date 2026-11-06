// dual_mlp_llama -- one llama-shaped SwiGLU MLP (M=32, D=2048, F=1024; A8W4) split across both Gemminis of the
// dual-Gemmini SoC: 16x16 (custom3) and 8x8 (custom2). See src/dual_mlp_inst.c for each Gemmini's schedule.
// The CPU only issues commands: it interleaves the two command streams (the instance with less estimated issued work
// goes next, so neither starves while the CPU blocks on the other's full queue) and fences exactly twice -- once
// at the H_full seam, once at the end. Checks H_full (codes + scales) and Y bit-exact against gen_dual_mlp_llama.py.
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "dual_mlp_llama.h"
#include "dual_mlp_api.h"

static inline uint64_t rdcycle(void) { uint64_t c; asm volatile("rdcycle %0" : "=r"(c)); return c; }
static inline void fence_all(void) { asm volatile("fence"); }   // waits for every RoCC accelerator on the core

static uint8_t h_full[DML_M * DML_F] __attribute__((aligned(64)));
static uint8_t h_sc[(DML_F / 32) * DML_M] __attribute__((aligned(64)));
static uint16_t y[DML_M * DML_D] __attribute__((aligned(64)));

typedef int (*step_fn)(void);

int main() {
  dml_bufs_t b = {DML_BLOB(DML_OFF_X), DML_BLOB(DML_OFF_X_SC), DML_BLOB(DML_OFF_WG), DML_BLOB(DML_OFF_WG_SC),
                  DML_BLOB(DML_OFF_WU), DML_BLOB(DML_OFF_WU_SC), DML_BLOB(DML_OFF_WD), DML_BLOB(DML_OFF_WD_SC),
                  h_full, h_sc, y};
  dml_info_t inf[2];
  dml_info_g16(&inf[0]);
  dml_info_g8(&inf[1]);
  for (int i = 0; i < 2; i++)
    printf("gemmini %dx%d (custom%d): F[%d:%d] N[%d:%d]  NC=%d KT=%d NC2=%d  spad %d/%d rows  ideal %llu + %llu\n",
           inf[i].dim, inf[i].dim, inf[i].opcode, inf[i].f0, inf[i].f0 + inf[i].fg, inf[i].n0, inf[i].n0 + inf[i].ng,
           inf[i].nc, inf[i].kt, inf[i].nc2, inf[i].sp_end, inf[i].sp_rows, (unsigned long long) inf[i].ideal1,
           (unsigned long long) inf[i].ideal2);
  memset(h_full, 0xa5, sizeof(h_full));
  memset(y, 0xa5, sizeof(y));
  dml_init_g16(&b);
  dml_init_g8(&b);
  step_fn step[2] = {dml_step_g16, dml_step_g8};
  fence_all();

  const uint64_t t0 = rdcycle();
  uint64_t t1 = 0, issued[2] = {0, 0};
  int state[2] = {0, 0};   // 0 issuing, 1 at the barrier, 2 done
  int phase = 1, cmds[2] = {0, 0};
  while (state[0] != 2 || state[1] != 2) {
    if (state[0] != 0 && state[1] != 0) {   // both at the barrier (phase 1 -> 2): the one fence of the run
      fence_all();
      t1 = rdcycle();
      state[0] = state[1] = 0;
      phase = 2;
      continue;
    }
    const int g = (state[1] != 0) ? 0 : (state[0] != 0) ? 1 : (issued[0] <= issued[1] ? 0 : 1);
    const int r = step[g]();
    cmds[g]++;
    if (r > 0) issued[g] += r;
    else state[g] = (r == 0) ? 1 : 2;
  }
  const uint64_t t_issue = rdcycle();
  fence_all();
  const uint64_t t2 = rdcycle();
  (void) phase;

  int hd = 0, sd = 0, yd = 0, first = -1;
  const uint8_t *h_ref = DML_BLOB(DML_OFF_H_REF), *hs_ref = DML_BLOB(DML_OFF_H_SC_REF);
  const uint16_t *y_ref = (const uint16_t *) DML_BLOB(DML_OFF_Y_REF);
  for (int i = 0; i < DML_M * DML_F; i++) hd += h_full[i] != h_ref[i];
  for (int i = 0; i < (DML_F / 32) * DML_M; i++) sd += h_sc[i] != hs_ref[i];
  for (int i = 0; i < DML_M * DML_D; i++) if (y[i] != y_ref[i]) { if (first < 0) first = i; yd++; }
  printf("H_full codes: %d/%d differ; H scales: %d/%d differ\n", hd, DML_M * DML_F, sd, (DML_F / 32) * DML_M);
  printf("Y = H @ Wd  : %d/%d differ", yd, DML_M * DML_D);
  if (first >= 0) printf(" (first @[%d][%d] got %04x exp %04x)", first / DML_D, first % DML_D, y[first], y_ref[first]);
  printf("\n");

  const uint64_t p1 = t1 - t0, p2 = t2 - t1, tot = t2 - t0;
  const uint64_t macs = 3ull * DML_M * DML_D * DML_F;
  const uint64_t ideal = macs / (2 * (16 * 16 + 8 * 8));   // both meshes at A8W4 peak (640 MACs / cycle)
  printf("PERF phase 1 (G/U + SwiGLU + requant): %llu cycles (ideal %llu | %llu per gemmini)\n", (unsigned long long) p1,
         (unsigned long long) (2ull * DML_M * DML_D * DML_F / 640), 0ull);
  printf("PERF phase 2 (down): %llu cycles (ideal %llu)\n", (unsigned long long) p2,
         (unsigned long long) (1ull * DML_M * DML_D * DML_F / 640));
  printf("PERF total %llu cycles; ideal %llu (both meshes at peak) -> util %llu.%llu%%; CPU issue done at %llu; "
         "commands issued %d + %d\n", (unsigned long long) tot, (unsigned long long) ideal,
         (unsigned long long) (ideal * 100 / tot), (unsigned long long) (ideal * 1000 / tot % 10),
         (unsigned long long) (t_issue - t0), cmds[0], cmds[1]);
  const int bad = hd + sd + yd;
  printf("dual_mlp_llama %s\n", bad ? "FAILED" : "PASSED (H_full and Y bit-exact)");
  return bad != 0;
}
