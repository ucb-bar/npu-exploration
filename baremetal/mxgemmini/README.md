# baremetal/mxgemmini — TinyLlama on MxGemmini, by hand

Seven kernels, each one ELF, each carrying real captured TinyLlama weights and activations. The mesh
does the matmuls in MX fp8 (e4m3 + E8M0); everything else — RMSNorm, RoPE, softmax, SwiGLU, the
residual — runs in fp32 on the Rocket scalar core, because the target contract declares one compute
unit, `mx_systolic_mesh`, `ops: [matmul]`, and there is no reduction hardware.

| kernel | shape | what it proves |
|---|---|---|
| `llama_mlp` | `M=32 D=2048 F=64` | the MLP chain, gate/up → SwiGLU → down |
| `llama_mlp_small` | same, `D` sliced to 256 | the same chain, a 0.7 MB header instead of 5 MB |
| `llama_mlp_tiny` | same, `D` sliced to 64 | one K-tile, one output chunk, a header you can read by eye |
| `llama_attention` | `M=32 D=2048 H=64`, 1 of 32 heads | the resident `P@V → o_proj` seam: the intermediate never leaves the scratchpad |
| `llama_attention_small` | same, `D` sliced to 256 | the same six matmuls at ~8 min of VCS instead of ~45 |
| `llama_attention_tiny` | same, `D` sliced to 64 | the smallest D the E8M0 block (32) and the mvin tile (DIM) allow |
| `llama_attention_full` | **all 32 heads**, nothing truncated | the real attention output, gradeable against TinyLlama itself |

Plus the **MX bisection ladder**, `mxl0`…`mxl9` — ten tiny self-checking ELFs for finding out *why*
one of the above fails on hardware. See [The ladder](#the-ladder) below.

| directory | contents |
|---|---|
| `src/` | the kernels, plus `mx_ladder.c` and its ten `mxlN.c` rungs |
| `include/mx_host.h` | the fp32 host runtime: bf16↔float, e4m3 encode, E8M0 block quantize, rmsnorm, silu, softmax, rope |
| `include/mx_mesh.h` | the mesh calling convention: operand mvin, output drain, one `mesh_matmul` |
| `gen/` | PyTorch capture → quantize → mesh golden → data (`gen_llama_layer.py`, `gen_llama_attn_full.py`, `gen_mx_ladder.py`) |
| `data/` | generated headers + the full-attention blob. **Gitignored** — run `make data` |

## Using it

```bash
make              # every kernel, for spike and the RTL path
make spike        # -DSPIKE_SIM (real RoCC)
make mx_rocket    # -DMX_ROCKET
make run          # build for spike and run each one
make ladder       # just the bisection ladder, both targets
make run-ladder   # the ladder on spike, in bisection order
make data         # regenerate data/ from the capture npz
```

ELFs land in `../../out/baremetal/{spike,mx_rocket}/`. A kernel passes when **every mesh matmul is
bit-exact** against a golden from `fp8_matmul_model.tiled_matmul_hwlike` — an independent
implementation of the same precision schedule, so agreement is a real gate rather than a tautology.

`make data` needs a capture first (once, needs the venv + HF cache):

```bash
cd ../.. && .venv/bin/python3 -m app.capture_llama_layer              # single head
          .venv/bin/python3 -m app.capture_llama_layer --all-heads    # for llama_attention_full
```

## The scratchpad is a compile-time constraint, not an assumption

Every kernel checks its scratchpad map against `BANK_NUM * BANK_ROWS` with `LLAMA_REQUIRE`, so a
config it does not fit is a **build error naming the region that overflowed** — not a run that
returns plausible wrong numbers. This is load-bearing: at `BANK_ROWS 2048`, `llama_mlp` used to
complete and report 69,569 wrong elements.

`llama_attention_full` goes further and **replans itself**: its projection width, head batching and
o_proj chunking are all chosen at compile time from the scratchpad size. Measured, same ELF source:

| `BANK_ROWS` | plan | result |
|---|---|---|
| 4096 (16384 rows) | `proj N=64 \| 32 heads \| o_proj 2 chunks of 1024` | PASS |
| 2048 (8192 rows) | `proj N=16 \| 32 heads \| o_proj 4 chunks of 512` | PASS, **bit-identical** |

The data blob is independent of all of it — an output column depends only on its own column of B —
so changing scratchpad size replans the C without regenerating a byte.

**Every kernel here replans**, not just the full one. `llama_mlp` and `llama_attention` derive both
their K-tiling (splitting the D-deep contraction when the weight tile will not fit beside the
activations) and their output chunking from the scratchpad size:

| kernel | 16384 rows | 8192 rows |
|---|---|---|
| `llama_mlp` | 1 K-tile of 2048, 2 chunks of 1024 | 2 K-tiles of 1024, 4 chunks of 512 |
| `llama_attention` | 1 K-tile of 2048, 2 chunks of 1024 | 2 K-tiles of 1024, 4 chunks of 512 |

All four pass at both sizes, and every graded number is identical across them — splitting a
2048-deep reduction into two accumulating 1024-deep matmuls is exact. At 16384 rows the projections
pick one K-tile spanning all of D, so the larger config runs the original schedule unchanged.

## One thing worth knowing before trusting a number

`llama_attention_full` depends on a fix to the functional model: `mx_loop_ws_spad` now honours
`loop_ws`'s `ex_accumulate` bit (rs1 bit 0), which it previously discarded. Whether the MX **RTL**
honours that bit is unverified, so treat that kernel as a spike result until it is checked —
[`../../planning/llama_layer_hw_plan.md`](../../planning/llama_layer_hw_plan.md) §10.3.

See [`../../README.md`](../../README.md) for install, and [`../README.md`](../README.md) for why
these live here rather than in the ISA test suite.

## The ladder

`llama_attention` PASSES on spike and fails on the FPGA with **every mesh stage wrong, including
the first projection** — while `matmul_tiled_fp8_64x64_chain` and the 128x128 fp8 test PASS on the
same bitstream. So the problem is not "MX fp8 on RTL"; it is something llama's matmuls do that the
ISA tests do not.

`Q = Xn @ Wq` is `[32,2048] x [2048,64]` → `I=2, J=4, TK=128`, with a 4096-byte B-side scale
window. The passing ISA tests are square (`I=J`), at most 8 k-tiles deep, and load 128 bytes of B
scales. The ladder walks from one to the other **one delta at a time**:

| rung | the one delta | isolates |
|---|---|---|
| `mxl0` | none — 64x64x64, through llama's own helpers | the calling convention, not the shapes |
| `mxl1` | M=32 → `I=2, J=4` | a **non-square tile grid** |
| `mxl2` | K=256 → `TK=16` | a deeper reduction |
| `mxl3` | K=1024 → `TK=64`, 2 KB B scales | depth and scale-window size |
| `mxl4` | K=2048 → `TK=128`, 4 KB B scales | **exactly `Q`** |
| `mxl5` | `mxl4` as two accumulating K-tiles | **`ex_accumulate` on RTL** — unverified, see the plan §10.3 |
| `mxl6` | two matmuls into one `C_spad` | **output-region reuse** / the shadow accumulator |
| `mxl7` | requant→spad tiled + resident scales | the `P@V → o_proj` seam at a non-square shape |
| `mxl8` | A as a column slice, strided mvin | the DMA row pitch |
| `mxl9` | no mesh at all — `mx_host.h` vs golden | the Rocket's fp32 / `expf` |

Run `out/baremetal/mx_rocket/mxl0` … `mxl9` **in order**. The first that fails names the feature;
everything below it is then a consequence, not a separate bug. Each prints its own `DIM`, the
scratchpad it planned against, its spad map, PASS/FAIL, and the first eight mismatching elements —
the *pattern* is the diagnosis, and a bare count cannot tell all-wrong from one-tile-wrong.

Two rungs test a hypothesis rather than just reporting it. `mxl5` says outright that failing while
`mxl4` passes means the RTL discards `ex_accumulate`; `mxl6` counts how many outputs equal
`bf16(golden#1 + golden#2)`, a high count being the signature of a shadow accumulator that was
never cleared.

**All ten PASS on spike**, which is the precondition for reading anything into an FPGA failure — a
rung failing there would be a bug in the ladder, not in the hardware.

### `gemmini_params.h` is shared with the ISA suite, and it moves

`DIM` is overridden by every kernel here, so a `DIM 32` header is harmless. **`BANK_ROWS` is not**,
and it silently replans every kernel: at 2048 the projections split into two accumulating K-tiles
and `o_proj` into four chunks, so a DIM=16 build made against the dim32 header depends on
`ex_accumulate` for no reason. The ladder `#warning`s on a `DIM` mismatch; nothing warns on
`BANK_ROWS`, so check it before building for hardware.
