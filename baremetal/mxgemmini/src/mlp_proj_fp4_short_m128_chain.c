// mlp_proj_fp4_short_chain at M = 128 tokens (2x the work per weight byte). A is 8192 rows per matmul: gate's at 0, the
// B buffers at 8192, down's from 12288 wrapping onto gate's blocks 0, 1 (loaded under gate's second N-chunk).
// Data: gen_mlp_proj_fp4.py --ng 256 --kd 2048 --m 128 --tag _short_m128.
#define MP_HEADER "mlp_proj_fp4_short_m128.h"
#define MP_ARES 1
#define MP_CHAIN 1
#define MP_CHAIN_DOWN_A 12288
#define MP_CHAIN_B 8192
#include "mlp_proj_fp4.c"
