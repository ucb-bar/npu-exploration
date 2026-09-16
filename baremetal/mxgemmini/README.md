# baremetal/mxgemmini — TinyLlama on MxGemmini, by hand

Four kernels, each one ELF, each carrying real captured TinyLlama weights and activations. The mesh
does the matmuls in MX fp8 (e4m3 + E8M0); everything else — RMSNorm, RoPE, softmax, SwiGLU, the
residual — runs in fp32 on the Rocket scalar core, because the target contract declares one compute
unit, `mx_systolic_mesh`, `ops: [matmul]`, and there is no reduction hardware.

| kernel | shape | what it proves |
|---|---|---|
| `llama_mlp` | `M=32 D=2048 F=64` | the MLP chain, gate/up → SwiGLU → down |
| `llama_attention` | `M=32 D=2048 H=64`, 1 of 32 heads | the resident `P@V → o_proj` seam: the intermediate never leaves the scratchpad |
| `llama_attention_small` | same, `D` sliced to 256 | the same six matmuls at ~8 min of VCS instead of ~45 |
| `llama_attention_full` | **all 32 heads**, nothing truncated | the real attention output, gradeable against TinyLlama itself |

| directory | contents |
|---|---|
| `src/` | the kernels |
| `include/mx_host.h` | the fp32 host runtime: bf16↔float, e4m3 encode, E8M0 block quantize, rmsnorm, silu, softmax, rope |
| `gen/` | PyTorch capture → quantize → mesh golden → data (`gen_llama_layer.py`, `gen_llama_attn_full.py`) |
| `data/` | generated headers + the full-attention blob. **Gitignored** — run `make data` |

## Using it

```bash
make              # every kernel, for spike and the RTL path
make spike        # -DSPIKE_SIM (real RoCC)
make mx_rocket    # -DMX_ROCKET
make run          # build for spike and run each one
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
