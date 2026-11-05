# Step 4: Cost

The cost step estimates what your hardware costs and how the model runs on it. Both models come from
`MxGemmini-workspace/ppa` (cloned next to this repo by `setup.sh`, or `MX_PPA_ROOT`), called as black
boxes and never reimplemented.

```bash
npu.py --hw baseline --model m.py:Block --only cost
.venv/bin/python -m models.ppa.ppa   --hw baseline [--run fp4_e2m1] [--json]
.venv/bin/python -m models.perf.perf --hw baseline --m 64 --k 64 --n 64 [--json]
```

## PPA: area and power (`models/ppa/ppa.py`)

Maps the hardware config and the operand format onto the post-synthesis area and power model. It is
calibrated at 16x16 in tstech16c at tt0p8v25c. The clock and utilization come from `implementation`.
The config's LUT unit picks which measured machine is priced:

| `mx.lut.projFormat` | machine priced |
|---|---|
| `LutFP6E3M2` (baseline) | the tapeout: `mxgemmini` PE, FP6 LUT requantizer |
| `LutFP8E4M3` | MxAll: quad PE, FP8 E4M3 LUT, all finders |
| other / none | not measured: reported as unavailable |

Nothing is extrapolated. A format the workspace never ran on that machine is reported unavailable,
with the missing point named.

## PERF: cycles and energy (`models/perf/perf.py`)

An RTL-calibrated analytical timeline of each GEMM stage on the configured hardware. When evaluate also
ran, the `PERF` line puts spike's **measured** cycle-model numbers first (cycles, time at the
configured clock, mesh utilization, energy over that window), with the analytical prediction under
`estimate`:

```
PERF  <cycles> cycles (spike cycle model)  <us> us  mesh util <pct>%  <uJ> uJ (<pJ/op> pJ/op) | estimate <cycles> cycles
```

Energy is the Gemmini power from `compose_gemmini --system radiance` at the measured utilization,
times the measured time. Without evaluate, only the estimate is reported.

SRAM macro area and energy (`memory`) need the PDK's SRAM compiler tables; without them that field
reads `{"available": false}` and nothing else changes.
