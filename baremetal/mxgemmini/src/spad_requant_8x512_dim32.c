// llama_layer_mx's RMSNorm SPAD_REQUANT on the 32x32 mesh: two 8x512 flat SRs back to back, each reading row 0 and
// writing row 272 of its own bank (banks 1 / 3), as the kernel's two streams.
#define M 8
#define N 512
#define SR_FLAT_ONLY
#define SR_TWO
#define SP_SRC   (1 * BANK_ROWS)
#define SP_FLAT  (1 * BANK_ROWS + 272)
#define SP_SRC2  (3 * BANK_ROWS)
#define SP_FLAT2 (3 * BANK_ROWS + 272)
#include "../../../../software/gemmini-rocc-tests/bareMetalC/spad_requant.c"
