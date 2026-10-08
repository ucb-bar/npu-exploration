// Projection-shaped, waveform-sized variant of mx_bench_matmul (perf-model calibration, npu-exploration/planning/
// perf_model_plan.md 13.9): M = 64 tokens gives mxn_matmul the same loop tile as llama_layer_e2e's projections
// (NC = 256, Kt = 256 at 512 accumulator rows: loops of 64 x 256 x 256), here K = 1024 x N = 512 = 8 loops
// (~131k ideal cycles) instead of the layer's 2048-5632. One pass, no load-only stream, so a VCS run with a
// waveform stays short. Correctness: CHECKSUM vs Spike (sum=1090220149 xor=3a8e659f66e70833).
#define BENCH_M 64
#define BENCH_K 1024
#define BENCH_N 512
#define BENCH_SKIP_STREAM
#define BENCH_ONE_PASS
#include "mx_bench_matmul.c"
