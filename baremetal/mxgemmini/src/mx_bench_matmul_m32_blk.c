// mx_bench_matmul_m32 with B PRE-BLOCKED (each loop's 128x512 tile contiguous in DRAM): same shape, loops
// and checksum as _m32 -- isolates the DRAM access pattern.
#define BENCH_M 32
#define BENCH_BLOCKED 1
#include "mx_bench_matmul.c"
