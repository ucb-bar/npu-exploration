// Repro for llama_mx_tiny_fp4_d8 (attention wrong after FP4 projections): one FP4 x FP4 quad native-loop matmul
// (results unused), then the unchanged attn_flash pass (E4M3 single). Control: attn_flash built alone.
#define main attn_flash_main
#include "attn_flash.c"
#undef main
#include "mx_native.h"

#define PM 64
#define PK 128
#define PN 64
static uint8_t  PA[PM / 2 * PK] __attribute__((aligned(64)));
static uint8_t  PB[PK * PN / 2] __attribute__((aligned(64)));
static uint16_t PC[PM * PN] __attribute__((aligned(64)));
static uint8_t  PAs[PK / 32 * PM] __attribute__((aligned(64)));
static uint8_t  PBs[PK / 32 * PN] __attribute__((aligned(64)));

int main() {
  memset(PAs, 127, sizeof(PAs)); memset(PBs, 127, sizeof(PBs));
  gemmini_flush(0);
  const int bad = mxn_matmul_fp4(PA, PK, PB, PN / 2, PC, PN, PAs, PM, PBs, PN, PM, PK, PN);
  gemmini_fence();
  printf("attn_after_fp4: FP4 %dx%dx%d matmul %s, then attention\n", PM, PK, PN, bad ? "REJECTED" : "done");
  return attn_flash_main();
}
