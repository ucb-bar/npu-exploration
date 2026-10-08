// mlp_proj_fp4 at about a third of the cycles for RTL simulation: gate 2048 -> 256, down K 2048 -> 512 (two 256-column
// chunks). The weights still stream cold from DRAM; h is re-read from the L2 rather than evicted (the full-size effect
// needs mlp_proj_fp4). Data: gen_mlp_proj_fp4.py --ng 256 --kd 2048 --tag _short.
#define MP_HEADER "mlp_proj_fp4_short.h"
#include "mlp_proj_fp4.c"
