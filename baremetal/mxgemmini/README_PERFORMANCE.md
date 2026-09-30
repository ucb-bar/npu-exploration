# High-performance MxGemmini kernels

Which kernels here are the **fast path** (native DRAM loops) and which are the original scratchpad-flow
baselines, what each measures, and what it achieved on RTL. The programming guide behind them is
[`../../../MXGEMMINI_PERFORMANCE.md`](../../../MXGEMMINI_PERFORMANCE.md).

All fast-path kernels use native DRAM loops (`gemmini_loop_ws_mx`): A and B stream from DRAM, C (BF16) is
stored straight to DRAM, and each loop loads its own E8M0 scale slices. The large ones go through one
helper, `mxn_matmul` in [`include/mx_native.h`](include/mx_native.h): N-chunks × K-tiles of loops, each
loop in its own scratchpad half, so the next loop's operands load under the current compute.
Every mesh stage is still checked **bit-exact** against its golden.

## Utilization benchmarks: start here

One large matmul, random fp8 data from a DRAM-cold blob (`gen/gen_mx_bench.py` → `data/mx_bench.{bin,h}`).

| kernel | shape (M × K × N) | what it measures | RTL result |
|---|---|---|---|
| `mx_bench_matmul` | 128 × 4096 × 1024 | **latency hiding**: compute-bound (~4.2 B/c demand) | **99.3%** (2,111,015 / 2,097,152 ideal) |
| `mx_bench_matmul_m32` | 32 × 4096 × 1024 | llama-shaped, memory-bound (~8.9 B/c demand) | **72.1%** (726,406 / 524,288), FireSim |
| `mx_bench_matmul_m32_blk` | same, B pre-blocked | whether the DRAM layout of the weights matters | 71.6%: it doesn't |

Each prints three passes:
- **pass 0:** a load-only 4 MB stream, the platform's DMA ceiling;
- **pass 1:** `PERF … util=`, traffic and B/c, plus `PERF_HW`, the mesh busy/idle split;
- **pass 2:** `PERF_LITTLE`, with reads in flight, latency per 64 B, and the % of cycles with the request
  table full (`xact_full`), the bus refusing requests (`tl_wait`), or the mvin tracker full (`ld_cmd_full`).

`CHECKSUM` must match Spike:

| kernel | Spike checksum |
|---|---|
| M = 128 | `sum=4391224426 xor=5a410dab13820ef0` |
| M = 32, both variants | `sum=1094338120 xor=2623395b490a2320` |

## Llama kernels: fast path vs baseline

Each native kernel is the same source as its baseline with `-DMLP_NATIVE`, `-DATTN_NATIVE` or `-DLAYER_NATIVE`
set by a small shim in `src/`. Cycles are RTL mesh cycles; lower is better.

| fast path | baseline | shape | native | baseline | ideal |
|---|---|---|---|---|---|
| **`llama_layer_full_native_gh`** | `llama_layer_full` | one full layer: 32 heads, F = 5632, host stages golden | **7,833,998 (70%)** | — | 5,521,408 |
| `llama_layer_full_native` | `llama_layer_full` | same, host stages computed | same mesh schedule | — | 5,521,408 |
| `llama_mlp_native` | `llama_mlp` | M32 D2048 F64 | 81,781 (60%) | — | 49,152 |
| `llama_attention_native` | `llama_attention` | M32 D2048, 1 head | 109,387 (60%) | — | 66,048 |
| `llama_mlp_small_native` | `llama_mlp_small` | M32 D256 F64 | **9,905** | 20,771 (**2.1×**) | 6,144 |
| `llama_mlp_tiny_native` | `llama_mlp_tiny_db` | M32 D64 F64 | **3,373** | 6,807 (**2.0×**) | 1,536 |
| `llama_attention_tiny_native` | `llama_attention_tiny` | M32 D64, 1 head | **5,934** | 10,158 (**1.7×**) | 2,560 |
| `llama_attention_small_native` | `llama_attention_small` | M32 D256, 1 head | not yet run on RTL | | 8,704 |

Notes:
- **`_gh`** (`-DLLAMA_GOLDEN_HOST`) replaces RMSNorm, RoPE, softmax, SwiGLU and residual 1 with their goldens
  from the blob. The RTL run is then mostly mesh time, and every mesh stage is still checked. The full-layer
  kernels print a per-phase table (QKV, S, O, o_proj, G/U, down) with measured cycles, ideal cycles and utilization.
- **Attention** keeps `P@V` as the resident scratchpad-requant loop. O and its scales feed o_proj in place
  in the D ≤ 2048 kernels; in the full layer they are concatenated across heads for one K = 2048 o_proj.
- **Other scratchpad-flow variants** (`llama_mlp_tiny`, `_tiny_t1`, `_tiny_db`, `llama_attention_tiny_t1`,
  the `_full` kernels, the ladder) are kept as baselines and for debugging.
- `llama_mlp_tiny_native_ua` is an alignment A/B test only.

## Why M = 32 tops out near 70%

The compute-bound benchmark proves the schedule hides load latency (99.3%). At M = 32 each weight byte feeds
only 32 rows, so a fed mesh needs ~8.9 B/c, while the memory system behind Gemmini sustains ~6 B/c. The
FireSim counters show why: `tl_wait` 80% (the bus is back-pressuring), `xact_full` 1% (Gemmini's
request cap is not the limit), and Little's law gives 6.05 B/c. The ways past it are fp4 weights, more
tokens per pass, or a wider or second memory channel.

## Building and running

```bash
export RISCV=/bwrcq/scratch/nicorakela/radiance-cy-dev/.conda-env/riscv-tools PATH=$RISCV/bin:$PATH
O=$(realpath ../../out/baremetal)                       # targets must be absolute paths
make $O/mx_rocket/mx_bench_matmul $O/spike/mx_bench_matmul
make $O/mx_rocket/llama_layer_full_native_gh
python3 gen/gen_mx_bench.py                             # regenerate the benchmark blob

# Spike with the local MX model (its "cycles" are instruction counts, not timing)
spike --extlib=../../../software/libgemmini/libgemmini.so --extension=gemmini $O/spike/mx_bench_matmul
```

- **Hardware:** `MxGemminiRocketConfig` (DIM 16, `acc_banks = 2`, 512-bit system bus) with the loop-managed
  scales, the cross-loop WAR hold in `LoopMatmul`, and the narrow-Put fix.
- **Platforms:** compare bandwidth numbers only within one platform (VCS or FireSim). Their DRAM models differ.
