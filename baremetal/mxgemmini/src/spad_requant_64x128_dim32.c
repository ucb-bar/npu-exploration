// llama_layer_mx's O requant shape on the 32x32 mesh: SPAD_REQUANT 64x128 E4M3, flat and operand-A tiled, mvout
// issued right behind the requant (no fence), as lmx_store_o.
#define M 64
#define N 128
#include "../../../../software/gemmini-rocc-tests/bareMetalC/spad_requant.c"
