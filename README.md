# npu-exploration

Define a model in PyTorch, compile it through merlin to MxGemmini RoCC instructions, run it on
spike, and grade it against MXQuant. **One ELF per kernel, any MX datatype.**

```bash
source scripts/env.sh /path/to/chipyard
.venv/bin/python run_kernel.py --list
.venv/bin/python run_kernel.py --kernel llama_mlp
.venv/bin/python run_kernel.py --kernel linear --dtype fp4_e2m1
```

## Status

Nine kernels, every one a single ELF, every one **bit-identical to MXQuant under `rtl_exact`**:

| kernel | shape | steps | vs MXQuant | vs fp32 |
|---|---|---|---|---|
| `linear` | 64³ | 1 mesh | 4096/4096 | 5.91% |
| `mlp2` … `mlp8` | 64³ | 2–8 mesh, fused chain | 4096/4096 | 8.97% … 22.7% |
| `attention` | 64³ | 6 mesh + 1 host | 4096/4096 | 13.62% |
| **`llama_mlp`** | 32×2048 | 3 mesh + 2 host | **65536/65536** | **11.5928%** |
| **`llama_attention`** | 32×2048 | 6 mesh + 4 host | **65536/65536** | **13.4157%** |

The two llama kernels are a **real TinyLlama decoder layer** — real captured activations and
weights, `d_model = 2048` kept full — and reproduce the hand-written `bareMetalC/llama_*.c`
reference kernels to within **5 ppm** (11.5933% / 13.4161%).

**All six MX formats are bit-identical to MXQuant under `rtl_exact`** — as single matmuls and as
fused chains up to eight stages deep (4096/4096 in every cell):

| | `fp8_e4m3` | `fp8_e4m3_quad` | `fp8_e5m2` | `fp6_e3m2` | `fp6_e2m3` | `fp4_e2m1` |
|---|---|---|---|---|---|---|
| single | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2-stage chain | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 8-stage chain | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Each is additionally gated against its shipped baremetal test's golden. **Two of those goldens are
currently stale**: spike moved to RNE rounding on 2026-09-10, so `matmul_fp8_64x64_chain.h` and
`matmul_fp6_64x64_chain.h` need regenerating (the old datapath reproduces them exactly; the new one
differs by 250 and 56 adjacent codes, all `odd → even`). `selftest_formats` reports those two as
FAIL until they are. Everything above is measured against the current spike.

**Not verified:** nothing has been run on RTL. The C compiles for `-DMX_ROCKET` and its instruction
stream matches the reference tests', but no simulator is built here.

## Grading

The verdict is **bit-identity against MXQuant**, not a tolerance against fp32:

| tier | compares against | answers |
|---|---|---|
| `mxquant_rtl_exact` | MXQuant + `rtl_exact/rtl_datapath.install()` | is the hardware correct? |
| `mxquant_shipped` | MXQuant as normally configured | how far is the model from the silicon? |
| `fp32` | `torch.matmul` | what MX costs at all — context only |

## Layout

| dir | what |
|---|---|
| `app/` | quantization (MXQuant only), host ops, format table, codebooks, interface MLIR, llama capture |
| `kernels/` | the kernel registry — a kernel is data, not code |
| `compiler/targets/mx_gemmini_rocket/` | the out-of-tree merlin target: contract, two emitters, C runtime, runner |
| `grade/` | run, compare against MXQuant, record |
| `rtl_exact/` | the MXQuant config that is bit-identical to the hardware |
| `MXQuant/`, `merlin/` | the quantization framework and the compiler framework |
| `tests/` | seven self-test suites — including a C oracle compiled from the datapath's own `mx_fp_math.h` |

Two emitters, on purpose: `mxgemm_emit` fuses a straight **chain** and keeps intermediates in the
scratchpad (no host round trip); `mxgraph_emit` handles any **graph** — multiple live values,
computed operands, host ops — with every edge through host fp32 memory, which is what the hardware
requires wherever a host op sits in a seam.

## What lives here

Everything workload-side. As of 2026-09-09 a graded run imports **zero** Python from
`gemmini-rocc-tests`:

| was | now |
|---|---|
| `llama_operands.build_luts` | `app/mxlut.py` |
| `fp8_matmul_model.py`, `fp4_matmul_model.py` | `app/mxmesh/` + `app/mxarith.py` (mechanically extracted, drift-tested) |
| `include/mx_host.h` | `compiler/.../backend/runtime/mx_host.h` |
| `gen_llama_layer.py`'s kernels | `kernels/registry.py` |
| the MX quantizer | `app/mxq_golden.py` (byte-identical to the shipped headers) |
| `verify_vs_pytorch.py` | `grade/` |

Still external, deliberately:

- **`include/gemmini.h`** + `rocc-software/`, `riscv-tests/` — the accelerator's ABI and the
  baremetal build environment. Owned by the hardware, like `libgemmini`.
- **the shipped `matmul_*.h` headers** — read by `tests/selftest_{quantizer,formats}.py` as the
  reference goldens we gate against. A dependency of the tests, not of the compiler.
- **`software/libgemmini`** — spike's functional model of the datapath.

Not ported, because nothing here needs them: the remaining per-format bit-exact mesh models
(`lut_*_model.py`) and the header generators (`gen_matmul_llama.py`, `lut_mapping_demo.py`,
`gen_fp6_chain.py`). `app/mxquant.py` remains as a deprecated shim so those generators still import.

## Tests

```bash
.venv/bin/python tests/selftest_grade.py          # metrics, telemetry, reporting
.venv/bin/python tests/selftest_quantizer.py      # our quantizer == the baremetal headers, byte for byte
.venv/bin/python tests/selftest_formats.py        # all 6 formats + 3 chains vs their shipped goldens
.venv/bin/python tests/selftest_mx_host.py        # the C runtime == its Python twin
.venv/bin/python tests/selftest_requant.py       # the chained requantizer, per step, vs a C oracle
.venv/bin/python tests/selftest_extracted.py      # app/mxmesh/ == the models it was extracted from
.venv/bin/python tests/selftest_mx_rocket_build.py  # one C source, builds for spike AND MX_ROCKET
.venv/bin/python rtl_exact/verify_rtl_exact.py    # MXQuant under rtl_exact == the hardware
```

## Plans

`planning/merlin_glue_port_plan.md` is the live one — decisions, every step's gate, and the
measurements behind them. `planning/chain_seam_hw_notes.md` and
`planning/llama_layer_hw_plan.md` carry the hardware findings the port depends on.
