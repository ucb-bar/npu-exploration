# MicroMX results and handover

Real TinyLlama-1.1B, Qwen2.5-7B and Llama-2-7B layer benchmarks run on the
supplied DIM16 BF16 FPGA image on `sohumpc`. Projections use mixed FP4/FP6/FP8;
decode attention runs on the scalar core.

## Measured gains

These gains are against the previous **MicroMX** implementations on the same
image. The timed precision mix is 50% FP4, 25% FP6 and 25% FP8.

| Workload | Change | Layer speedup | Including cache initialization |
|---|---|---:|---:|
| TinyLlama, 64-token prefill | Reuse packed Q/K/V and gate/up activations | 1.10× | — |
| TinyLlama, batch-one decode | Tiled grouped-query attention | 1.22× | 1.20× |
| Qwen2.5-7B, batch-one decode | Tiled grouped-query attention | 1.18× | 1.17× |
| Llama-2-7B, batch-one decode | Tiled attention and blocked cache initialization | 1.16× | 1.17× |

Each result uses five measured trials after warmup. Decode trials replay eight
teacher-captured tokens after a 2,048-token prefix. Checked BF16 stage outputs
match their independent same-mix references exactly; the FP8 reference runs
remain unchanged. This is one transformer layer, not full-model generation or
a perplexity evaluation.

The Qwen 62.5/25/12.5 row-layout specialization also passed FPGA functional
checks: FP4 issues 32 rows and FP6/FP8 issue 16, retaining 32-row storage and
scale pitches. It remains opt-in; **no speedup or timing profile is claimed**.

Exact cycle counts, validation counts and image/firmware/archive hashes are in
[the results metadata](MICROMIX_RESULTS.json). Timings refer to those measured
firmware binaries; rebuilding the cleaned source requires a new FPGA timing run.

## Code and review branches

All four repositories use the branch `kernels/micro-mx`:

| Repository | Contents |
|---|---|
| [ucb-bar/npu-exploration](https://github.com/ucb-bar/npu-exploration/tree/kernels/micro-mx) | Main kernel, packing, model-layer replay and test changes; based on `compiler-dev` |
| [copparihollmann/gemmini](https://github.com/copparihollmann/gemmini/tree/kernels/micro-mx) | Integration checkout, build instructions and pinned submodules |
| [copparihollmann/libgemmini](https://github.com/copparihollmann/libgemmini/tree/kernels/micro-mx) | Full-BF16 and direct E3M2 Spike model corrections |
| [copparihollmann/gemmini-rocc-tests](https://github.com/copparihollmann/gemmini-rocc-tests/tree/kernels/micro-mx) | Shared-accumulator instruction wrapper |

The native package follows `baremetal/mxgemmini/{include,src,gen,tests,docs}`.
The cleanup removes unused decoder implementations, formats the code, and
keeps experimental row layouts separate from measured defaults. Host tests
cover packing, numerical checks, prepared-buffer lifetimes and command shapes;
Spike tests execute projections and real-layer replays. Model assets and raw
run archives stay outside the source repositories.

## Next work

CPU attention and other scalar layer operations remain the main decode cost.
Priorities are timed row-layout trials, further attention/cache improvements,
and qualification of the remaining 62.5% FP4 profiles. See
[the kernel guide](micromix.md) and [decode replay instructions](MICROMIX_DECODE.md)
for build and validation commands.
