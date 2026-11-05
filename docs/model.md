# Model

The second input is a plain PyTorch `nn.Module` at full precision:

```bash
npu.py --hw baseline --model examples/mlp.py:MLP
npu.py --hw baseline --model my_model.py:Block --input x.npy     # a real [M][K] input
```

`Name` is a class (instantiated with no arguments after `torch.manual_seed(--seed)`) or an instance.
The input is `randn(m, k)` (`--m`, `--k`) unless `--input` names a 2-D `.npy` file.

## What a model may contain

`kernels/trace.py` traces the module with torch.fx into an ordered list of stages:

| op | runs on |
|---|---|
| bias-free `nn.Linear`, `torch.matmul` (an rhs transpose is folded in) | the accelerator mesh |
| scaled / causal softmax, add, `silu(g) * u`, `nn.RMSNorm`, `kernels.trace.RoPE` | the Rocket host |

Anything else is refused, never approximated. Bias and fused activations need an epilogue the
backend does not have. A custom module can be lowered whole by registering its type in
`kernels.trace.TRANSLATORS`. Every traced model satisfies `reference() == module(x)` in fp32
(`tests/selftest_trace.py`).

## Shape rules

Checked before anything is built (`KernelSpec.validate`):

- `M`, `K`, `N` are multiples of the mesh dimension (16)
- `K` is a multiple of the scale group (32)
- in a chain, each stage's `K` equals the previous stage's `N`
- a stage that feeds another needs `N` a multiple of 32 (one E8M0 scale per 32 output columns)

## Built-in example models

`--kernel NAME` runs a registered model instead of a file (`npu.py --list` shows them): `linear`,
`mlp2` ... `mlp8`, `attention`, and `llama_mlp` / `llama_attention` once a TinyLlama capture exists
(`python -m kernels.captures.llama_layer`, needs the MXQuant clone). They are defined in
`kernels/registry.py`; `--m --k --h --n` set their shapes.
