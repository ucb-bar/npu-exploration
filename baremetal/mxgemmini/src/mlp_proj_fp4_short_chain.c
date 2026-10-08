// mlp_proj_fp4_short_ares with gate and down issued back to back (no fence between, disjoint A regions): the down
// projection's A / B loads overlap the gate's last compute. Times both together.
#define MP_ARES 1
#define MP_CHAIN 1
#include "mlp_proj_fp4_short.c"
