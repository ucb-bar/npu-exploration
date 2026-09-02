# kernels — what to run

A kernel is **data, not code**: a PyTorch model plus an input, flattened into an ordered
list of matmul `Stage`s. The backend lowers exactly one matmul per command buffer, so a
model of N `nn.Linear` layers becomes N stages.

| file | role |
|---|---|
| `spec.py` | `Stage`, `KernelSpec`, `from_module()` — the machinery; rarely edited |
| `registry.py` | the catalogue: name → builder. **Add kernels here.** |

Adding one is a few lines in `registry.py` and no other change — see "Adding a kernel" in
the root `README.md`.

`KernelSpec.reference()` computes the FP32 answer for the whole chain, and
`KernelSpec.validate()` checks shape legality *before* anything is built, mirroring
`mxgemm_emit._validate` so failures are cheap and name the offending stage.

`from_module()` raises on bias and on activations rather than approximating them: both
need a COMMIT epilogue, which the backend explicitly refuses. That is also why attention
is not expressible yet — its two matmuls are fine, but softmax is neither a matmul nor an
allowed epilogue.
