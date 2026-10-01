// The same stacked-model kernel as `llama_model.c`, at a 64-token context instead of 32.
//
// Everything is planned from LLAMA_M at compile time, so this is only a different data header.
// 64 is the largest context the current plan holds: the attention projections contract the full
// D in one loop_ws, and their A-side scale window is M*D/512 rows -- 256 at M=64, the ceiling.
//
// Regenerate the data with:
//   .venv/bin/python3 -m kernels.captures.llama_model --seq 64 --out out/model_capture_m64
//   cd gen && ../../../.venv/bin/python3 gen_llama_model.py --capture ../../../out/model_capture_m64 --tag _m64
#define LLAMA_MODEL_HEADER "llama_model_m64.h"
#include "llama_model.c"
