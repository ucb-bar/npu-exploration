// llama_model_m64.c over the FIRST TWO layers: the 64-token bring-up build (see llama_model_l2.c
// for why the truncated stack's perplexity is meaningless and what it still gates).
//
//   cd gen && ../../../.venv/bin/python3 gen_llama_model.py --capture ../../../out/model_capture_m64 --layers 2 --tag _m64l2
#define LLAMA_MODEL_HEADER "llama_model_m64l2.h"
#include "llama_model.c"
