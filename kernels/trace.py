"""torch.fx -> KernelSpec: trace a PyTorch module into the kernel registry's IR.

``spec.from_module`` deliberately handles only ``nn.Linear`` chains; this module
is the graph-shaped frontend on top of the same contract. It symbolically traces
a module and translates each FX node into the registry's vocabulary:

  * ``nn.Linear`` (bias-free)                  -> mesh ``Stage(weight=...)``
  * ``torch.matmul(a, b)`` / ``a @ b``         -> mesh ``Stage(lhs=..., rhs=...)``,
    a ``.t()``/``.transpose(-2,-1)`` on the rhs folded into the ``"name.T"`` spelling
  * ``F.softmax(z, dim=-1)``                   -> ``HostStage(op="softmax")``, with a
    preceding scalar ``mul``/``div`` folded into ``scale=`` and a constant
    ``masked_fill`` recognized as ``causal=True``
  * ``a + b``                                  -> ``HostStage(op="add")``
  * ``silu(g) * u``                            -> ``HostStage(op="swiglu")``
  * any module type in ``TRANSLATORS``         -> whatever its translator returns
    (register your RMSNorm/RoPE here; ``linear_translator``/``rmsnorm_translator``
    are provided)

Everything else RAISES, naming the FX node -- same stance as ``spec.py``:
anything this cannot express fails loudly instead of silently lowering
something different from what PyTorch would compute. In particular a
standalone ``silu`` or ``transpose`` is refused (the C graph emitter has no
branch for them yet), and bias/activations are refused exactly as
``from_module`` refuses them.

The strongest property, and the one the selftest leans on: for every traced
module, ``spec.reference()`` computes the same fp32 result as ``module(x)``.
"""
from __future__ import annotations

import math
import operator
from typing import Callable

import torch
import torch.fx as fx
from torch import nn

from .spec import INPUT, AnyStage, HostStage, KernelSpec, Stage


class TraceError(ValueError):
    """The module contains something this frontend cannot express. Fail closed."""


def _fail(node: fx.Node, why: str) -> TraceError:
    return TraceError(f"cannot lower FX node {node.op} {node.target!r} (as {node.name!r}): {why}")


# ---- translators for custom module types ----------------------------------------------

#: type -> fn(stage_name, submodule, src_names: tuple[str, ...]) -> Stage | HostStage.
#: Extend for your own modules: ``TRANSLATORS[MyRMSNorm] = my_translator``.
TRANSLATORS: dict[type, Callable] = {}


def linear_translator(name: str, mod: nn.Linear, srcs: tuple[str, ...]) -> Stage:
    """The one built-in lowering: bias-free nn.Linear, same convention as from_module."""
    if mod.bias is not None:
        raise TraceError(f"{name}: nn.Linear has a bias; build it with bias=False "
                         "(bias needs a COMMIT epilogue, unsupported on this datapath)")
    return Stage(name=name, weight=mod.weight.detach().T.contiguous().float(), lhs=srcs[0])


def rmsnorm_translator(name: str, mod: nn.Module, srcs: tuple[str, ...]) -> HostStage:
    """For any module exposing ``weight`` (a 1-D tensor) and ``eps`` -- e.g. LlamaRMSNorm."""
    w = getattr(mod, "weight", None)
    eps = getattr(mod, "eps", None) or getattr(mod, "variance_epsilon", None)
    if w is None or eps is None:
        raise TraceError(f"{name}: rmsnorm_translator needs .weight and .eps/.variance_epsilon "
                         f"on {type(mod).__name__}")
    return HostStage(name, op="rmsnorm", src=srcs[0],
                     params={"weight": w.detach().float().numpy(), "eps": float(eps)})


TRANSLATORS[nn.Linear] = linear_translator


# ---- FX-graph helpers ------------------------------------------------------------------

_MATMULS = (torch.matmul, operator.matmul, torch.mm)
_ADDS = (torch.add, operator.add)
_MULS = (torch.mul, operator.mul)
_DIVS = (torch.div, operator.truediv)


def _is_matmul(node: fx.Node) -> bool:
    return node.target in _MATMULS or (node.op == "call_method" and node.target == "matmul")


def _is_softmax(node: fx.Node) -> bool:
    t = node.target
    return t is torch.softmax or t == "softmax" or getattr(t, "__name__", "") == "softmax"


def _scalar_scale(node: fx.Node) -> bool:
    return (node.op == "call_function" and node.target in (*_MULS, *_DIVS)
            and _is_scalar(node.args[1]))


def _folds_downstream(node: fx.Node) -> bool:
    """True when a later stage will absorb this node, so the main loop must skip it.

    FX iterates topologically, so a transpose (folded by the matmul that consumes
    it) and the masked_fill / scalar mul / div below a softmax are visited BEFORE
    their consumer; without this check they would be rejected as standalone ops.
    """
    if not _only_user(node):
        return False
    user = next(iter(node.users))
    if _is_last2_transpose(node):
        return _is_matmul(user) and len(user.args) > 1 and user.args[1] is node
    if _scalar_scale(node) or (node.op == "call_method" and node.target == "masked_fill"):
        while _scalar_scale(user) or (user.op == "call_method" and user.target == "masked_fill"):
            if not _only_user(user):
                return False
            user = next(iter(user.users))
        return _is_softmax(user)
    return False


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float)) or (isinstance(v, torch.Tensor) and v.numel() == 1)


def _only_user(node: fx.Node) -> bool:
    return len(node.users) == 1


def _is_last2_transpose(node: fx.Node) -> bool:
    """x.t() / x.transpose(-2,-1) (any spelling of the last two dims of a 2-D tensor)."""
    if node.op == "call_method" and node.target == "t":
        return True
    if node.op in ("call_method", "call_function") and (
            node.target == "transpose" or node.target is torch.transpose):
        dims = tuple(node.args[1:]) or (node.kwargs.get("dim0"), node.kwargs.get("dim1"))
        return sorted(d % 2 if isinstance(d, int) and d >= 0 else d for d in dims) in (
            [-2, -1], [0, 1])
    return False


def _const_tensor(gm: fx.GraphModule, node: fx.Node) -> torch.Tensor | None:
    """The tensor behind a get_attr node (a buffer/parameter/attr), else None."""
    if node.op != "get_attr":
        return None
    obj = gm
    for part in str(node.target).split("."):
        obj = getattr(obj, part)
    return obj if isinstance(obj, torch.Tensor) else None


def _is_causal_mask(mask: torch.Tensor) -> bool:
    """True for the strictly-upper-triangular boolean mask (mask==True is dropped)."""
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        return False
    n = mask.shape[0]
    return bool(torch.equal(mask, torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)))


# ---- the tracer ------------------------------------------------------------------------

def trace(module: nn.Module, x: torch.Tensor, *, name: str = "traced",
          validate: bool = True, dim: int = 16, block: int = 32) -> KernelSpec:
    """Trace ``module`` on symbolic input shaped like ``x`` into a KernelSpec.

    ``x`` is the real input tensor the spec will carry ([M][K] fp32). Raises
    :class:`TraceError` on anything the datapath + host-op vocabulary cannot
    express. With ``validate=True`` the spec's shape legality is checked too.
    """
    gm = fx.symbolic_trace(module)
    modules = dict(gm.named_modules())

    stages: list[AnyStage] = []
    env: dict[fx.Node, str] = {}          # fx node -> spec value name ("x", stage, "stage.T")
    consumed: set[fx.Node] = set()        # nodes folded into another stage (transposes, scales)
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    if len(placeholders) != 1:
        raise TraceError(f"expected exactly one input, module takes {len(placeholders)}")

    def ref(node: fx.Node, *, side: str) -> str:
        """The spec name for an operand node, folding a transpose on the rhs."""
        if side == "rhs" and _is_last2_transpose(node) and _only_user(node):
            src = node.args[0]
            consumed.add(node)
            return f"{env[src]}.T"
        if node not in env:
            raise _fail(node, "operand was not produced by a lowerable stage")
        return env[node]

    def softmax_src(node: fx.Node) -> tuple[str, float, bool]:
        """Unwrap masked_fill (-> causal) and scalar mul/div (-> scale) below a softmax."""
        scale, causal = 1.0, False
        while True:
            if node.op == "call_method" and node.target == "masked_fill" and _only_user(node):
                mask = _const_tensor(gm, node.args[1]) if isinstance(node.args[1], fx.Node) else None
                fill = node.args[2]
                if mask is None or not _is_causal_mask(mask):
                    raise _fail(node, "masked_fill whose mask is not a constant strictly-upper-"
                                      "triangular bool tensor; only causal masking is expressible")
                if not (isinstance(fill, float) and math.isinf(fill) and fill < 0):
                    raise _fail(node, "masked_fill value must be -inf for a softmax mask")
                consumed.add(node)
                if isinstance(node.args[1], fx.Node):
                    consumed.add(node.args[1])
                causal, node = True, node.args[0]
            elif node.op == "call_function" and node.target in _MULS and _only_user(node) \
                    and _is_scalar(node.args[1]):
                consumed.add(node)
                scale, node = scale * float(node.args[1]), node.args[0]
            elif node.op == "call_function" and node.target in _DIVS and _only_user(node) \
                    and _is_scalar(node.args[1]):
                consumed.add(node)
                scale, node = scale / float(node.args[1]), node.args[0]
            else:
                return env_name_or_fail(node), scale, causal

    def env_name_or_fail(node: fx.Node) -> str:
        if node not in env:
            raise _fail(node, "operand was not produced by a lowerable stage")
        return env[node]

    pending_silu: dict[fx.Node, fx.Node] = {}     # silu node -> its src, awaiting a mul

    for node in gm.graph.nodes:
        if node in consumed or _folds_downstream(node):
            continue

        if node.op == "placeholder":
            env[node] = INPUT

        elif node.op == "get_attr":
            continue    # only legal when folded by a translator/mask; using it raises later

        elif node.op == "call_module":
            mod = modules[str(node.target)]
            fn = TRANSLATORS.get(type(mod))
            if fn is None:
                for klass, f in TRANSLATORS.items():
                    if isinstance(mod, klass):
                        fn = f
                        break
            if fn is None:
                if isinstance(mod, nn.SiLU):
                    pending_silu[node] = node.args[0]
                    continue
                raise _fail(node, f"module type {type(mod).__name__} has no translator; "
                                  "add one to kernels.trace.TRANSLATORS")
            srcs = tuple(env_name_or_fail(a) for a in node.args if isinstance(a, fx.Node))
            st = fn(node.name, mod, srcs)
            stages.append(st)
            env[node] = st.name

        elif node.op in ("call_function", "call_method"):
            t = node.target
            if getattr(t, "__name__", "") == "linear":
                # F.linear(x, W, bias) -- how a root-level nn.Linear (or a direct
                # functional call) traces. W is a get_attr parameter node.
                w = _const_tensor(gm, node.args[1]) if isinstance(node.args[1], fx.Node) else None
                if w is None:
                    raise _fail(node, "F.linear with a computed weight; only a parameter is "
                                      "lowerable (a computed rhs should be a matmul)")
                if len(node.args) > 2 and node.args[2] is not None:
                    raise _fail(node, "F.linear has a bias; build it with bias=False "
                                      "(bias needs a COMMIT epilogue, unsupported)")
                consumed.add(node.args[1])
                stages.append(Stage(name=node.name, weight=w.detach().T.contiguous().float(),
                                    lhs=env_name_or_fail(node.args[0])))
                env[node] = node.name
            elif t in _MATMULS or (node.op == "call_method" and t == "matmul"):
                a, b = node.args[0], node.args[1]
                lhs = env_name_or_fail(a)
                rhs = ref(b, side="rhs")
                stages.append(Stage(name=node.name, lhs=lhs, rhs=rhs))
                env[node] = node.name
            elif (t is torch.softmax or t == "softmax"
                  or getattr(t, "__name__", "") == "softmax"):
                if node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else -1) not in (-1, 1):
                    raise _fail(node, "only row softmax (dim=-1) exists on the host")
                src, scale, causal = softmax_src(node.args[0])
                stages.append(HostStage(node.name, op="softmax", src=src,
                                        params={"scale": scale, "causal": causal}))
                env[node] = node.name
            elif t in _ADDS:
                a, b = node.args[0], node.args[1]
                if not (isinstance(a, fx.Node) and isinstance(b, fx.Node)):
                    raise _fail(node, "add with a scalar has no host op; only tensor+tensor")
                stages.append(HostStage(node.name, op="add",
                                        src=(env_name_or_fail(a), env_name_or_fail(b))))
                env[node] = node.name
            elif t in _MULS:
                a, b = node.args[0], node.args[1]
                sil = next((n for n in (a, b) if isinstance(n, fx.Node) and n in pending_silu), None)
                if sil is not None:
                    other = b if sil is a else a
                    stages.append(HostStage(node.name, op="swiglu",
                                            src=(env_name_or_fail(pending_silu.pop(sil)),
                                                 env_name_or_fail(other))))
                    env[node] = node.name
                else:
                    raise _fail(node, "bare multiply: only silu(g)*u (-> swiglu) and a scalar "
                                      "scale folded into softmax are expressible")
            elif getattr(t, "__name__", t) in ("silu",):
                pending_silu[node] = node.args[0]
            elif _is_last2_transpose(node):
                raise _fail(node, "standalone transpose: fold it into a matmul rhs as '<name>.T' "
                                  "(the C graph emitter has no transpose branch)")
            else:
                raise _fail(node, "no lowering for this op")

        elif node.op == "output":
            out = node.args[0]
            if isinstance(out, (tuple, list)):
                raise TraceError("module returns multiple outputs; the datapath computes one")
            if not stages or env.get(out) != stages[-1].name:
                raise TraceError(f"output {getattr(out, 'name', out)!r} is not the last stage; "
                                 "reorder the module so the returned value is computed last")

    if pending_silu:
        n = next(iter(pending_silu))
        raise _fail(n, "standalone silu is not graph-emittable; only silu(g)*u (swiglu) is")

    spec = KernelSpec(name=name, x=x.detach().float(), stages=stages)
    if validate:
        spec.validate(dim=dim, block=block)
    return spec
