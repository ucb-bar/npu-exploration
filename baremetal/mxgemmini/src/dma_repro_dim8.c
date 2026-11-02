// Minimal repro (8x8 mesh, MxDim8Fp4E3M2E4M3VpuGemminiRocketConfig): 64-byte MX_LOAD_SCALES Gets in flight with
// mvouts corrupt memory -- whole 64-byte lines of recently read data land at wrong addresses. dma_stress_dim8's T5c,
// cut down: one setup round trip, then REPRO_ITERS x (mvout every tile, one 256-byte scale load per tile), fenced
// and checked after each pass. Stops at the first damaged line. Three printfs when clean (each costs ~30M cycles of
// HTIF on FireSim), so it is also small enough for RTL simulation with TileLink monitors.
// REPRO_SCALE_LEN 8 makes the loader issue 8-byte Gets instead, which never failed on FireSim.
#include <stdint.h>
#include <stdio.h>

#include "include/gemmini_testutils.h"

#if !defined(MX_ROCKET) && !defined(SPIKE_SIM)
int main() { printf("skipped: MX config / Spike-only test\n"); return 0; }
#else

#define REPRO_KB 256
#define REPRO_ITERS 8

#ifndef REPRO_SCALE_LEN
#define REPRO_SCALE_LEN 256   // bytes per MX_LOAD_SCALES; the loader issues 64-byte Gets for 256
#endif
#define BYTES   (REPRO_KB * 1024)
#define LD      128           // row stride (V in llama_mx_tiny)
#define VCOLS   64            // one mvin row: MAX_BYTES / DIM tiles
#define N_TILES (BYTES / (DIM * VCOLS))
#define FILL    0xA5

static uint8_t SRC[BYTES] __attribute__((aligned(64)));
static uint8_t DST[BYTES] __attribute__((aligned(64)));
static uint8_t SCL[2048] __attribute__((aligned(64)));
static uint8_t src_byte(uint32_t i) { return (uint8_t) ((i * 2654435761u) >> 13) ^ (uint8_t) (i >> 9); }

// tile t: rows [r, r+8) x cols [c, c+64) of an LD-strided matrix; in the scratchpad as 8 DIM-wide tiles from sp
#define TILE_R(t) (((t) / (LD / VCOLS)) * DIM)
#define TILE_C(t) (((t) % (LD / VCOLS)) * VCOLS)
#define TILE_SP(t) ((uint32_t) (t) * VCOLS)
static void mvout_tile(int t) {
  for (int k = 0; k < VCOLS / DIM; k++)
    gemmini_extended_mvout(DST + TILE_R(t) * LD + TILE_C(t) + k * DIM, TILE_SP(t) + k * DIM, DIM, DIM);
}

// First line of SRC (must still be its pattern) or DST (must equal SRC) that is wrong; 0 if none.
static uintptr_t first_bad(const char **which) {
  for (int i = 0; i < BYTES; i++)
    if (SRC[i] != src_byte(i)) { *which = "SRC"; return (uintptr_t) (SRC + (i & ~63)); }
  for (int i = 0; i < BYTES; i++)
    if (DST[i] != src_byte(i)) { *which = "DST"; return (uintptr_t) (DST + (i & ~63)); }
  return 0;
}
static int report(int iter) {
  const char *which;
  const uintptr_t a = first_bad(&which);
  if (!a) return 0;
  const uint8_t *p = (const uint8_t *) a;
  int from = -1;   // whose original contents the line now holds
  for (int l = 0; l < BYTES / 64 && from < 0; l++) {
    int b = 0;
    while (b < 64 && p[b] == src_byte(l * 64 + b)) b++;
    if (b == 64) from = l * 64;
  }
  printf("dma_repro FAILED at pass %d: %s line %lx (+%lx)", iter, which, (unsigned long) a,
         (unsigned long) (a - (uintptr_t) (*which == 'S' ? SRC : DST)));
  if (from >= 0) printf(" = original SRC+%x\n", from);
  else {
    printf(" now");
    for (int b = 0; b < 16; b++) printf(" %02x", p[b]);
    printf("\n");
  }
  return 1;
}

int main() {
  for (int i = 0; i < BYTES; i++) SRC[i] = src_byte(i);
  for (int i = 0; i < (int) sizeof(SCL); i++) SCL[i] = 0x7F - (i & 7);
  printf("dma_repro DIM %d: SRC %lx DST %lx SCL %lx, %d tiles x %d passes, scale loads of %d B\n", DIM,
         (unsigned long) SRC, (unsigned long) DST, (unsigned long) SCL, N_TILES, REPRO_ITERS, REPRO_SCALE_LEN);
  gemmini_flush(0);
  gemmini_config_ld(LD); gemmini_config_st(LD);
  // setup: SRC -> scratchpad, fenced; the passes only mvout it
  for (int t = 0; t < N_TILES; t++) gemmini_extended_mvin(SRC + TILE_R(t) * LD + TILE_C(t), TILE_SP(t), VCOLS, DIM);
  gemmini_fence();
  for (int it = 0; it < REPRO_ITERS; it++) {
    for (int i = 0; i < BYTES; i++) DST[i] = FILL;
    for (int t = 0; t < N_TILES; t++) {
      gemmini_mx_load_scales(SCL + (t % 8) * 256, REPRO_SCALE_LEN, t & 1);
      mvout_tile(t);
    }
    gemmini_fence();
    if (report(it)) return 1;
  }
  printf("dma_repro DIM %d PASSED (%d passes)\n", DIM, REPRO_ITERS);
  return 0;
}
#endif
