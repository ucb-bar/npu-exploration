# baremetal — hand-written application kernels

Whole-application kernels written by hand in C, one directory per hardware target. This is the
track that runs *real model code* on real silicon models; the compiler track
([`../compiler/`](../compiler/README.md), driven by [`../run_kernel.py`](../README.md)) is the one
that *generates* such code from PyTorch. They meet at the same ISA and the same reference.

| target | contents |
|---|---|
| [`mxgemmini/`](mxgemmini/README.md) | TinyLlama MLP and attention on MxGemmini, fp8 e4m3 + E8M0 |

## Why these are not in `gemmini-rocc-tests/bareMetalC`

That directory is the **ISA-level test suite**: one test per instruction behaviour, each
self-contained, each runnable by `run_mx_vcs.sh`. A TinyLlama decoder layer is not that — it carries
captured model weights, a generator, and an fp32 host runtime, none of which has anything to do with
the instruction under test. Keeping proof-of-concept ISA tests there and applications here is the
split; see [`../planning/llama_layer_hw_plan.md`](../planning/llama_layer_hw_plan.md) §9.2.

What a target here still borrows from `gemmini-rocc-tests` is exactly what it should: the ISA
headers (`gemmini.h`, `gemmini_params.h`, `gemmini_testutils.h`), the riscv-tests baremetal runtime,
and `gen_matmul_llama.py` — which stays over there because it generates the `matmul_*.h` data for
the ISA tests themselves.

See [`../README.md`](../README.md) for install and the run command.
