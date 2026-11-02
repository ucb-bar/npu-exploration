// Debug harness: llama_layer_mx's rms_phase alone (xn1), with per-token code hashes and the last chunk's spad
// intermediates (x, w, stats, y) hashed, to compare RTL against Spike stage by stage.
#define main lmx_full_main
#include "llama_layer_mx.c"
#undef main

static uint16_t dbg_rows[8 * NCR * VL] __attribute__((aligned(64)));

static void dump(const char *name, uint32_t sp, int rows) {
  memset(dbg_rows, 0xa5, sizeof(dbg_rows));
  mv_out(dbg_rows, sp, rows);
  gemmini_fence();
  printf("spad %-4s %016llx\n", name, (unsigned long long) fnv(dbg_rows, (size_t) rows * DIM));
}

int main() {
  lmx_rw = (uint8_t *) LMX_AT(0, uint8_t);
  __asm__ volatile("" : "+r"(lmx_rw));
  const uint16_t *H_PRE = LMX_AT(LMX_OFF_H_PRE, uint16_t);
  printf("llama_mx_rms_dbg: DIM %d, D=%d, NC %d, NCR %d, VL %d\n", DIM, LD, NC, NCR, VL);
  gemmini_flush(0);
  gemmini_fence();
  rms_phase(H_PRE, 0, 0, LMX_AT(LMX_OFF_W_IN_LN, uint16_t), xn1_c, xn1_sr);
  gemmini_fence();
  printf("hash xn1 %016llx (golden %016llx)\n", (unsigned long long) fnv(xn1_c, sizeof(xn1_c)),
         (unsigned long long) LMX_EXPECT[0]);
  for (int t = 0; t < LM / PQ; t++)
    printf("tok %2d %016llx\n", t, (unsigned long long) fnv(xn1_c + (size_t) t * LD, LD));
  for (int s = 0; s < 2; s++) {
    printf("stream %d (last chunk):\n", s);
    dump("x", RX(s), 8 * NCR);
    dump("w", RW(s), NCR);
    dump("st", RS(s), 8);
    dump("y", RY(s), 8 * NCR);
  }
  return 0;
}
