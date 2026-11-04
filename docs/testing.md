# Testing

Each test is a plain script that defends one claim: exit `0` on pass, non-zero on failure. Run the one
whose claim you are about to depend on, or all of them before a commit:

```bash
for t in tests/selftest_*.py tests/test_recipe_drift.py; do .venv/bin/python "$t" || echo "FAIL $t"; done
```

| step | test | claim |
|---|---|---|
| inputs | `test_recipe_drift.py` | `baseline.json` and the compile-path constants equal libgemmini, the Chisel source, the emitters and `rtl_exact/` |
| inputs | `selftest_trace.py` | every traced model's fp32 reference equals `module(x)` |
| quantize | `selftest_scheme.py` | `config/scheme.py` builds the extracted hardware model's arithmetic from each config |
| quantize | `selftest_mxquant.py` | the quantized model follows the config and degrades honestly when it cannot |
| quantize | `selftest_block.py` | the block quantizer equals MXQuant's bit for bit (frozen fixture) |
| quantize | `selftest_workload.py`, `selftest_torchao.py`, `selftest_codebook_mxq.py` | the perplexity path gives the same bits as the kernel path |
| compile | `selftest_lower.py` | `lower.py` builds the command buffers merlin's parser built |
| compile | `selftest_compile.py` | the compile step writes the same C as the evaluate step, and refuses what it cannot build |
| compile | `selftest_formats.py`, `selftest_quantizer.py`, `selftest_requant.py`, `selftest_mx_host.py`, `selftest_lut_layer.py` | operand bytes, the requantizer and the host runtime equal their references |
| compile | `selftest_mx_rocket_build.py` | one C source builds for both spike and `MX_ROCKET` |
| evaluate | `selftest_grade.py` | the verdict, the fp32 fallback, NO VERDICT, and the run record |
| evaluate | `selftest_timing.py` | the cycle model leaves the bits unchanged |
| evaluate | `selftest_attend_spike.py` | mxq's attention core equals the chip's on spike |
| evaluate | `selftest_extracted.py` | `rtl_exact/mxmesh/` still equals the models it was extracted from |
| cost | `selftest_ppa.py`, `selftest_perf.py` | the cost models are driven with the workspace's own tokens and parse back |
| — | `verify_rtl_exact.py` | MXQuant under `rtl_exact/` equals the hardware on a real llama MLP (needs the MXQuant clone) |

Fixtures live in `tests/oracle/` and `tests/fixtures/`.
