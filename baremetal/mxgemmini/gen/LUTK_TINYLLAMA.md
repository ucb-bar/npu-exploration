# TinyLlama per-K-block LUT kernels on MX-Gemmini: shapes, data, checks

Real TinyLlama operands for `src/lutk_proj.c`, taken from the MXQuant study's weights-only LUT runs
(`runs/tinyllama_var_u32x0_tkblock_w[_gG]_s0.jsonl`). The masks, codebook fit and scale rule are the ones
behind the perplexity numbers.

## Kernel shapes

C[M×N] (BF16) = A[M×K] (MXFP8 E4M3, E8M0 per 32 along K) · W[K×N].

Each 32-row K-block is either all MXFP8 (mode 8) or all LUT (4-bit index into a 16-entry E4M3 codebook). One codebook covers 32 K-rows × 2^G output columns.

All kernels use layer 10, M = 64 tokens (16–79 of WikiText-2 calibration sequence 0), the full K of the layer, and N = 256 (output columns 0–255). LUT K-blocks are given per budget as b5 / b6 / b7; the `all` variant uses every K-block.

| tag | linear | K | K-blocks | G=1 | G=3 | G=5 | G=7 | G=8 |
|---|---|---|---|---|---|---|---|---|
| `tl10_o_g<G>` | self_attn.o_proj | 2048 | 64 | 25/15/8 | 26/14/8 | 26/14/9 | 27/16/8 | 27/16/9 |
| `tl10_gate_g<G>` | mlp.gate_proj | 2048 | 64 | 64/63/61 | 64/62/55 | 63/62/51 | 62/61/51 | 62/59/50 |
| `tl10_up_g<G>` | mlp.up_proj | 2048 | 64 | 63/48/0 | 61/46/0 | 61/44/0 | 59/43/0 | 58/41/0 |
| `tl10_down_g<G>` | mlp.down_proj | 5632 | 176 | 173/103/1 | 172/102/3 | 171/104/7 | 167/109/10 | 171/111/11 |

Codebooks per K-block = 256 >> G: 128, 32, 8, 2 and 1. N = 256 is the smallest slice that still covers G = 8.

The ELF variants are the same as `gen_lutk_proj.py`:
- `fp8` (`LK_VAR=-1`): all MXFP8.
- `lut0`–`lut3`: b5, b6, b7, all.
- `fp4_<i>`: B4 is only a placeholder (nearest E2M1 on the MXFP8 scales), not the study's FP4 ablation. Skip it.

### Kernel changes needed for G > 1
`lutk_proj.c` assumes G = 1:
- It hard-codes `#define LK_G 1`. The data header has `LK_DATA_G`; use it.
- The codebook stride is `N / 2` (`LUT + (kb * (N / 2) + ((c * NC) >> LK_G)) * LK_LUT_BYTES`). For G > 1 the data is `BLUT[LK_NKB][LK_N >> G][16]`, so the stride must be `N >> G`. The same applies to the `BLUTC<i>` copies and `lk_cbc`.
- Pass `G` to `gemmini_mxquant_config_mvout(..., LK_G)`.
- `LK_LUT_HALF / (NC >> LK_G)` already scales with G.

Everything else matches the generator: `A`, `AS`, `B8`, `BL`, `BS`, `BLUTC<i>`, and the goldens
`FP8_D<d>`, `PHI/PLUT/CLUT/CSL/CML<i>_D<d>`.

## Generate the data
Run on CPU. The script uses two environments and switches between them itself.

```bash
cd /tools/scratch/nicorakela/radiance-cy-dev/generators/gemmini/npu-exploration/baremetal/mxgemmini
PY=/scratch/nicorakela/envs/mxquant/bin/python
for G in 1 3 5 7 8; do
  for t in o:self_attn.o_proj gate:mlp.gate_proj up:mlp.up_proj down:mlp.down_proj; do
    CUDA_VISIBLE_DEVICES= $PY gen/gen_lutk_tinyllama.py --layer model.layers.10.${t#*:} \
        --m 64 --n 256 --g $G --tag tl10_${t%%:*}_g$G --dims 16,32
  done
done
```

Each run writes three files to `data/`:
- `lutk_<tag>.bin`: the blob.
- `lutk_<tag>.h`: shape, offsets and masks.
- `lutk_<tag>_ref.npz`: the study's fp32 outputs, codebooks and indices.

Run time: down_proj (K=5632) takes 24 min with `--dims 16,32`. Most of that is the bit-accurate goldens, so use `--dims 16` to roughly halve it. The K=2048 layers take about 9 min each. `lutk_tl10_down_g1` is already generated.

Build exactly like the random data: `make lutk LUTK_TAGS="tl10_down_g1 ..."`.

## Check and compare
1. **On chip:** the kernel's own golden check should PASS bit-exact. The goldens come from the same model (`fp8_matmul_model.tiled_matmul_hwlike`) on the real operands. Pick the golden that matches the schedule:
   - CML: mixed-K, one accumulation in K order.
   - CSL: shared accumulator, LUT K-blocks first.
   - CLUT: two partials plus a CPU add.
2. **Against the study:** dump C (BF16 hex words, row-major) from the sim log, then run:
   ```bash
   python gen/compare_lutk_tinyllama.py --tag tl10_down_g1 --var b6 --dim 16 --golden CML --rtl sim.log
   python gen/compare_lutk_tinyllama.py --tag tl10_down_g1 --var b6 --dim 16 --self-test   # plumbing check
   ```
   It reports:
   - bit-exactness against the golden;
   - the RTL's relative error against the study's fp32 output (the arithmetic the perplexities used), which should equal the golden's;
   - the LUT's own effect (study vs all-MXFP8).

### Expected numbers (`tl10_down_g1`, relative Frobenius error)
| variant | LUT K-blocks | hw golden vs study fp32 | study vs all-MXFP8 (LUT effect) |
|---|---|---|---|
| b5 | 173/176 | 2.2e-2 (d16), 1.65e-2 (d32) | 7.0e-2 |
| b6 | 103/176 | 2.2e-2 / 1.66e-2 | 5.3e-2 |
| b7 | 1/176 | 2.25e-2 / 1.67e-2 | 4.5e-3 |
| all | 176/176 | 2.2e-2 / 1.65e-2 | 7.1e-2 |

The BF16 accumulation over K = 5632 differs from the study's fp32 GEMM by about 2%. That is larger than the LUT's effect at the b7 budget (0.45%). The perplexity numbers assume fp32 accumulation, so for a long-K layer the chip's BF16 accumulator, not the LUT, sets the error floor.
