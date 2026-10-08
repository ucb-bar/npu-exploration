// Element-wise phases of llama_layer_e2e on their own (perf-model calibration, npu-exploration/planning/
// perf_model_plan.md 13.13): the layer's rms_phase (VPU + SPAD_REQUANT: h_pre -> xn1 codes + scales) and
// residual_phase (VPU only: h_out = a + b), exactly as the layer runs them, on the layer's data blob. The layer's
// file is included with its main renamed, so its functions and DRAM/scratchpad layout are used unchanged.
// ~0.3M cycles in all (FireSim, full layer: rmsnorm1 141k, residual2 178k) -- sized for a VCS run with a waveform.
// Correctness: the printed hashes (64-bit-word FNV) vs Spike.
#define main llama_layer_e2e_full_main
#include "llama_layer_e2e.c"
#undef main

// 64-bit-word FNV-1a: the byte-wise fnv() over these ~390 KB would cost the RTL core millions of cycles
static uint64_t hash64(const void *p, size_t n) {
  const uint64_t *w = (const uint64_t *) p;
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n / 8; i++) h = (h ^ w[i]) * 0x100000001b3ull;
  return h;
}

int main() {
  e2e_rw = (uint8_t *) E2E_AT(0, uint8_t);
  __asm__ volatile("" : "+r"(e2e_rw));
  const uint16_t *H_PRE = E2E_AT(E2E_OFF_H_PRE, uint16_t);
  printf("llama_e2e_elemwise: llama_layer_e2e's rmsnorm1 -> MX (VPU + SR) and a residual add (VPU), "
         "%d tokens x D=%d\n", LM, LD);
  gemmini_flush(0);
  gemmini_fence();
  const uint64_t t0 = read_cycles();
  rms_phase(H_PRE, 0, 0, E2E_AT(E2E_OFF_W_IN_LN, uint16_t), xn1_c, xn1_sr);   // as the layer's main calls it
  gemmini_fence();
  const uint64_t t1 = read_cycles();
  residual_phase(H_PRE, H_PRE, Hout);   // h_pre + h_pre: real activations, no earlier phase needed
  gemmini_fence();
  const uint64_t t2 = read_cycles();
  printf("PERF phase rmsnorm1 -> MX            %8llu cycles\n", (unsigned long long) (t1 - t0));
  printf("PERF phase residual (h_pre + h_pre)  %8llu cycles\n", (unsigned long long) (t2 - t1));
  printf("hash xn1    %016llx\n", (unsigned long long) hash64(xn1_c, sizeof(xn1_c)));
  printf("hash xn1_sr %016llx\n", (unsigned long long) hash64(xn1_sr, sizeof(xn1_sr)));
  printf("hash hout   %016llx\n", (unsigned long long) hash64(Hout, sizeof(Hout)));
  return 0;
}
