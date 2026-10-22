# llama_layer_mx — one decoder layer of 7–13B models on MxGemmini

`src/llama_layer_mx.c` is `llama_layer_e2e.c` (the TinyLlama layer, every stage on the device) generalized over
the **model**, the **projection precision** and the **mesh size**. One ELF per combination, 45 in all:

| precision | projections (Q/K/V, o_proj, gate/up, down) and their inputs | attention + KV cache | meshes |
|---|---|---|---|
| `fp8` | E4M3 (single) | E4M3 | 16×16, 8×8, 32×32 |
| `fp4` | E2M1 (quad mesh, 4× the E4M3 rate) | **E4M3** | 16×16, 8×8, 4×4 |
| `fp6` | E3M2 through 16-entry LUTs, G = 4 (quad mesh) | **E4M3** | 16×16, 8×8, 4×4 |

| model | D | F | heads / kv heads | head dim | notes |
|---|---|---|---|---|---|
| `llama31_8b` Llama-3.1-8B | 4096 | 14336 | 32 / 8 | 128 | RoPE base 500000, llama3 rope scaling |
| `llama2_7b` Llama-2-7B | 4096 | 11008 | 32 / 32 | 128 | |
| `llama2_13b` Llama-2-13B | 5120 | 13824 | 40 / 40 | 128 | |
| `qwen25_7b` Qwen2.5-7B | 3584 | 18944 | 28 / 4 | 128 | Q/K/V biases, RMSNorm eps 1e-6, RoPE base 1e6 |
| `mistral_7b` Mistral-7B-v0.1 | 4096 | 14336 | 32 / 8 | 128 | (the 4096 sliding window is wider than the 2112 keys) |
| `tiny` (bring-up) | 512 | 1536 | 4 / 2 | 128 | every code path of the above at ~1% of the cost |

Each ELF runs exactly one layer for a **64-token prefill chunk after a 2048-token KV cache** (causal over 2112 keys),
as `llama_layer_e2e` does, and checks itself against a bit-exact golden.

## The layer

```
VPU+SR  xn1 = Q(rmsnorm(h_pre) * w_in)              8-token streams x2, column chunks of LMX_RMS_NC
mesh    Q, K, V = xn1 @ Wq / Wk / Wv                 native DRAM loops (mx_native.h)
VPU     (+ bq, bk, bv)                               Qwen2.5 only
VPU+SR  Q, K = rope(Q), rope(K) -> E4M3; V^T -> E4M3 the chunk's K / V appended to the cache (keys 2048..2111)
attn    one flash pass per query head (attn_flash.c), E4M3, online softmax on the VPU; O -> Q(.)
mesh    Yattn = O @ Wo
VPU+SR  h_mid = h_pre + Yattn;  xn2 = Q(rmsnorm(h_mid) * w_post)
mesh    G, U = xn2 @ Wg / Wu
VPU+SR  H = Q(silu(G) * U)
mesh    Ymlp = H @ Wd
VPU     h_out = h_mid + Ymlp
```

`Q(.)` is the projection format's SPAD_REQUANT: E4M3, FP4 (E2M1), or FP6 (BF16 → E3M2, then the requantizer's
nearest-finder picks the index in the tensor's 16-entry table). The CPU only issues commands, reorders the small
scale arrays, transposes V and appends the chunk to the cache, exactly as in `llama_layer_e2e`.

What had to change from the TinyLlama kernel, and why:

* **Head dim 128, one query head per pass.** TinyLlama packed two GQA heads (head dim 64) per flash pass; at 128 the
  scratchpad holds one. Wq / bq carry 1/sqrt(128), so attention needs no scaling pass.
* **Column chunking.** A 4096–5120-wide row of 8 tokens no longer fits a bank pair, so RMSNorm, RoPE, SwiGLU and the
  residual work in column chunks. RMSNorm's sum of squares becomes RSUM per chunk with the partial sums added in BF16;
  that changes the result, so the chunk width (`LMX_RMS_NC`) is chosen by the golden and compiled into the kernel.
* **Every mesh size.** `attn_flash.c` is now DIM-generic (`-DATTN_PARAMS_DIM` takes DIM / BANK_ROWS / ACC_ROWS from the
  mesh's params header; VPU rows are DIM/2 BF16 lanes). Unchanged at 16×16: `llama_layer_e2e`,
  `attn_flash_llama7b_fused` and `attn_flash_llama_2h_fused_fp4` give identical hashes and identical cycles.
  The 4×4 mesh's 128 KB scratchpad takes 64-key blocks and the packed layout with V double-buffered.
* **The VPU's 10-bit row counter.** A reduction's row length is at most 1024 scratchpad rows (2048 BF16 at DIM 4), so
  RSUM runs per chunk there too.

## Validation: every stage bit-exact against `rtl_exact`

`gen/golden_llama_layer_mx.py` recomputes the whole layer in Python, independently of Spike, in the kernel's order of
operations:

* matmuls through `rtl_exact` (`rtl_datapath.py`'s truncated product, both-addends-quantized lane accumulation, bf16
  cross-tile), with **that mesh's** accumulator: `mxgemmini_rtl.json` (16), `_dim8.json`, `_dim4.json`, and the new
  `_dim32.json` (window 32: lanes 0–14 the 16×16 ramp, 15–31 bf16, from `gemmini.cc`'s `acc_e / acc_m`);
* the VPU through `golden_llama_layer_e2e`'s port of `vpu_ref.h`, with the RSUM lane tree over DIM/2 lanes;
* SPAD_REQUANT for E4M3, FP4 and FP6-LUT (an independent port of `bf16_bits_to_fp6_e3m2_code` and the finder).

It writes `data/llama_mx_<model>_<fmt>_d<dim>_expect.h`: 22 stage hashes (xn1 + scales, Q/K/V, the four RoPE outputs,
V^T, O + scales, Yattn, h_mid, xn2 + scales, G, U, H + scales, Ymlp, h_out) and the two plan constants that change
numbers (RMSNorm chunk, key blocks). The kernel hashes the same 22 buffers on Spike and prints `PASSED` only when
**all 22 equal the golden**.

## Performance: Spike's perf model

`GEMMINI_MODE=both GEMMINI_PERF_CONFIG=e4m3_vpu` (bit-exact math and the timing model; VPU ×2). The model's structural
parameters default to the 16×16 build even in `libgemmini_dim{4,8,32}.so`, so `run_llama_mx.py` sets them per mesh
with `GEMMINI_PERF_SET`: `mesh.dim`, `spad.bank_rows`, `acc.total_rows / bank_rows`, `loop.max_block_len`,
`dma.spad_write_bytes_per_cycle`, `st.elems_per_acc_read`, `sreq.cycles_per_block`. Measured latencies keep their
16×16 calibration; numbers at other meshes are model estimates, not yet checked against RTL.

**A perf-model fix this needed** (`software/libgemmini/perf`): the VPU and SPAD_REQUANT command decoders assumed 14-bit
scratchpad addresses. The 4×4 and 8×8 configs have 32K-row scratchpads and use 15-bit fields (`SPAD_ROW_ADDR_BITS 15`,
as `gemmini.cc`'s functional decode already did), so the timing model read every VPU command with ~8× its row count:
8×8 flash attention went from 29% to 96% mesh utilization with the fix. 16×16 and 32×32 timing is unchanged
(`llama_layer_e2e`: 15,195,700 cycles before and after).

## Results: all 45 pass, every stage bit-exact

Spike, `e4m3_vpu` preset, per-mesh parameters as above. Each cell: **layer cycles (M)** · mesh utilization (the layer's mesh-ideal cycles at the format's rate / measured). All 45 print `PASSED (22/22 stages match the rtl_exact golden)`. Full table with accuracy: `results/llama_mx/summary.md`.

| model | fp8 16×16 | fp8 8×8 | fp8 32×32 | fp6 16×16 | fp6 8×8 | fp6 4×4 | fp4 16×16 | fp4 8×8 | fp4 4×4 |
|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | **63.9** · 92% | **246.8** · 95% | **38.2** · 38% | **27.5** · 65% | **86.3** · 83% | **311.5** · 92% | **25.2** · 71% | **82.8** · 87% | **309.6** · 93% |
| Llama-2-7B | **63.0** · 87% | **234.1** · 94% | **40.6** · 34% | **29.3** · 58% | **85.2** · 80% | **299.2** · 91% | **27.6** · 61% | **82.0** · 83% | **297.3** · 91% |
| Llama-2-13B | **95.0** · 89% | **357.3** · 95% | **62.4** · 34% | **42.4** · 59% | **124.2** · 81% | **439.4** · 92% | **40.6** · 62% | **118.8** · 85% | **436.3** · 92% |
| Qwen2.5-7B | **66.8** · 93% | **259.7** · 96% | **43.1** · 36% | **28.1** · 65% | **88.4** · 83% | **318.2** · 92% | **27.0** · 68% | **84.4** · 87% | **315.8** · 93% |
| Mistral-7B-v0.1 | **63.9** · 92% | **246.8** · 95% | **38.2** · 38% | **27.5** · 65% | **86.3** · 83% | **311.5** · 92% | **25.2** · 71% | **82.8** · 87% | **309.6** · 93% |

Accuracy of the layer update (h_out − h_pre) against the fp64 layer, within ~1 pt across meshes: **fp8 12.3–13.5%, fp6 41–43%, fp4 76–79%** (see the caveat below).

Where the time goes (share of layer cycles; Llama-3.1-8B is GQA with 8 kv heads, Llama-2-7B full MHA with 32):

| model | fmt | mesh | rms1 | Q/K/V | bias+rope+cache | attention | o_proj | res1+rms2 | gate/up | swiglu | down | res2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | fp8 | 16×16 | 0% | 10% | 2% | 8% | 7% | 1% | 46% | 2% | 23% | 0% |
| Llama-3.1-8B | fp8 | 32×32 | 0% | 10% | 3% | 6% | 6% | 1% | 45% | 2% | 25% | 1% |
| Llama-3.1-8B | fp4 | 16×16 | 1% | 8% | 6% | 19% | 5% | 3% | 35% | 5% | 18% | 1% |
| Llama-3.1-8B | fp4 | 4×4 | 0% | 8% | 1% | 24% | 5% | 1% | 38% | 2% | 19% | 1% |
| Llama-2-7B | fp8 | 16×16 | 0% | 20% | 8% | 8% | 7% | 1% | 36% | 2% | 18% | 1% |
| Llama-2-7B | fp4 | 16×16 | 1% | 13% | 17% | 18% | 4% | 2% | 27% | 4% | 13% | 1% |

* **FP8 on 16×16 and 8×8 is mesh-bound:** every projection runs at 98–99% of the mesh ideal and the layer at 87–96%.
* **FP8 on 32×32 is weight-bandwidth-bound (34–38%):** at 64 tokens a 32×32 E4M3 mesh wants ~16 B/cycle of weights,
  about twice what the modelled memory system sustains. The layer is still the fastest FP8 build (38–62 M cycles).
* **FP4 / FP6 run the projections at 4× the E4M3 rate,** so the E4M3 attention grows to 15–25% of the layer. At 16×16
  the quad projections are themselves bandwidth-limited (FP6 also loads its LUTs per loop), hence 58–71%; at 4×4 they are
  compute-bound and the layer reaches 91–93%.
* **Full-MHA models pay for the cache append on the CPU:** Llama-2-7B / 13B (32 / 40 kv heads) spend 8–17% (16×16) in
  "bias + rope + cache", whose CPU part (the V transpose and the per-key cache writes) scales with the kv width. It is
  2–6% for the GQA models.
* Llama-3.1-8B and Mistral-7B-v0.1 have identical shapes, so identical cycles (the timing is data-independent; their
  hashes differ).


## Using it

```bash
export RISCV=/bwrcq/scratch/nicorakela/radiance-cy-dev/.conda-env/riscv-tools PATH=$RISCV/bin:$PATH
PY=../../.venv/bin/python3
$PY run_llama_mx.py --model all --fmt all                   # data -> goldens -> build -> Spike -> report (~1 h)
$PY run_llama_mx.py --model llama2_7b --fmt fp4 --dim 8     # one build
$PY run_llama_mx.py --model tiny                            # the bring-up shape, all nine, ~3 min
make llama-mx                                               # just the ELFs (spike + mx_rocket), given the data
```

| step | what | where |
|---|---|---|
| `gen` | synthetic layer at the model's shapes, fp64 reference, quantized weights / cache / LUTs | `gen/gen_llama_layer_mx.py` → `data/llama_mx_<model>_<fmt>.{bin,h}` (0.13–0.36 GB) |
| `golden` | the rtl_exact golden per mesh | `gen/golden_llama_layer_mx.py` → `data/..._d<dim>_expect.h` (30–120 s on 250 cores) |
| `build` | `make` the Spike (and with `--rocket`, MX_ROCKET) ELFs | `out/baremetal/{spike,mx_rocket}/llama_mx_<model>_<fmt>_d<dim>` |
| `run` | Spike, `libgemmini_dim<N>.so`, all in parallel (5–25 min each) | `out/llama_mx/<elf>.log` |
| `report` | the table above | `results/llama_mx/summary.{md,json}` |

The `mx_rocket` ELFs build against the params headers of `MxE4M3VpuGemminiRocketConfig` (16×16, the default header),
`dim8/e3m2_vpu` (`MxDim8E3M2VpuGemminiRocketConfig`), `dim4/vpu` (`MxDim4{Fp4,E3M2}VpuGemminiRocketConfig`) and
`dim32/vpu` (`MxDim32E4M3VpuGemminiRocketConfig`); those headers fix only DIM and the memory sizes. The kernel needs a
VPU built with `expSub` / `expSum` (as `llama_layer_e2e`) plus the projection format's datapath (FP6: the LUTs). Which
RTL config implements which combination has not been checked here, and none of these 45 has run on RTL yet.

## Things to know before trusting a number

* **The weights are synthetic.** The checkpoints are not in this tree, so every tensor is seeded random at the model's
  exact shapes: N(0, 1/fan_in) weights, RMSNorm gains around 1, hidden states N(0, 1) with four channels ×20 (the
  "massive activation" pattern), Qwen's biases N(0, 0.5) on Q/K. The cache is the fp64 layer's own K / V of 2048
  earlier tokens. Bit-exactness and timing do not depend on the values; the accuracy column does.
* **FP4 / FP6 accuracy is low by construction here.** Every SPAD_REQUANT and the weights use the shipped scale
  convention (`log2_pmax = 0`, decided 2026-09-09: `planning/merlin_glue_port_plan.md` §4.11), which puts a block's max
  in [1, 2): E2M1 then uses only ±{0, 0.5, 1, 1.5, 2}, and the ×20 outlier channels flatten everything else in their
  blocks. FP6's activation tables are one per tensor, k-means over the fp64 reference activation's E3M2 codes (a static
  calibration on the same input); weight tables one per 16 columns (`mxq.lut`).
* **"llama7b" / "llama13b" are Llama-2-7B / Llama-2-13B** (LLaMA-1 has the same shapes; only RMSNorm eps differs).
* Attention is not causal-masked inside the cache (every cached key precedes every query), only on the chunk's own
  block, exactly as `llama_layer_e2e`.
