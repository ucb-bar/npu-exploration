"""The two recipes: the hardware one (a design point) and the run one (how it is driven).

``config/hardware/<name>.json`` is the machine: mesh, product format, accumulator ladder, product
flush, block size, scratchpad, clock and utilization. Every field except ``name``, ``description``
and ``provenance`` is hashed into ``build_id``, which keys the spike build, the perplexity cache and
every record. ``config/run/<name>.json`` is how that machine is driven: operand format, rounding,
scale floor, reducer and the grading tolerance, and, for a LUT format, how its LUTs are made
(``lut``); ``run_id`` hashes it the same way. A run recipe never triggers a build.
The hardware recipe says what LUT unit was built (``mx.lut``, the RTL's GemminiLUTConfig); the run recipe
says how it is used (``lut``). Every LUT setting is written in one of the two files: the code holds no
default for any of them (planning/LUT_integration.md). The run recipe's optional ``vector`` block says at
what precision the perplexity path computes the vector ops between the matmuls (softmax, RMSNorm); absent,
they run as transformers computes them.

Every key is read by something; an unknown key is refused by name. ``check(hw, run, path)`` refuses
the combinations a path cannot follow (the kernel path runs on spike and the chip's requantizer, the
perplexity path only on mxq).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG = Path(__file__).resolve().parent
HARDWARE_DIR = CONFIG / "hardware"
RUN_DIR = CONFIG / "run"

#: Fields that name or explain a recipe and change no number: left out of build_id / run_id.
_LABELS = {"name", "description", "provenance"}

_HW_TOP = {"name", "description", "provenance", "array", "types", "mx", "accumulator", "scratchpad",
           "implementation"}
_HW_SECTIONS = {
    "array": {"meshRows", "meshColumns", "tileRows", "tileColumns"},
    "types": {"meshProdPrecisionList", "meshAccPrecisionList", "prodFloor"},
    "mx": {"scaleSize", "scaleSizeOut", "enable_lut", "lut"},
    "accumulator": {"acc_read_full_width", "acc_read_small_width"},
    "scratchpad": {"banks", "rows"},
    "implementation": {"clock_ns", "utilization"},
}
_MXFLOAT = {"expWidth", "sigWidth", "count", "isRecoded", "pad"}
_RUN_KEYS = {"name", "description", "operand_fmt", "rounding", "scale_floor", "reduce",
             "allow_lossy_chain", "fp32_tol", "lut", "scale", "vector"}
#: Optional run keys: absent means today's behaviour, and leaves run_id unchanged.
_RUN_OPTIONAL = {"description", "lut", "scale", "vector"}
#: Where the block scale puts the block maximum: "mxgemmini" (in [1, 2): the chip's requantizer, MXQuant, every
#: record so far) or "ocp" (at the format's maximum, OCP MX v1.0; mxq.block.ocp, perplexity path only).
SCALES = ("mxgemmini", "ocp")

#: The run recipe's vector block: every op written, each one of these (mxq.nn.patch(vector=...)). null = as
#: transformers computes it (fp32 inside, bf16 out); "bf16" = every step's result rounded to bf16.
VECTOR_OPS = ("softmax", "rmsnorm")
VECTOR_CHOICES = (None, "bf16")

#: mx.lut: the RTL's GemminiLUTConfig field names (MxConfigFragments.scala:48), so one JSON describes both.
_HW_LUT_KEYS = {"projFormat", "rdataWidth", "raddrWidth", "numEntries", "numBits", "lutUpdateRegularityWidth",
                "actCodeWidth", "weiCodeWidth"}
#: The LUT formats each projection's requantizer can index: the nearest-entry finders QuantLut builds for
#: it (QuantLut.scala:105-155, gemmini 04d7502). A fact of the RTL, looked up by mx.lut.projFormat.
LUT_SERVES = {
    "LutFP6E3M2": ("fp6_e3m2",),
    "LutFP6E2M3": ("fp6_e2m3",),
    "LutFP8E5M2": ("fp8_e5m2", "fp6_e3m2"),
    "LutFP8E4M3": ("fp8_e4m3_quad", "fp8_e5m2", "fp6_e3m2", "fp6_e2m3"),
}
_FP8_PROJECTIONS = ("LutFP8E4M3", "LutFP8E5M2")     # GemminiLUTConfig.isFp8Proj: requires rdataWidth == 8

#: The run recipe's lut block. Every key is required; each accepts the values the compiler implements
#: today. The others the plan names (top16, calibrated, given tables, finder pick) are refused by name.
_LUT_KEYS = {"group", "weights", "activations", "outputs", "pick", "fit"}
_FIT_KEYS = {"method", "init", "max_iters"}
LUT_CHOICES = {
    "weights": ("data",),           # B tables: k-means over the weights' own codes
    "activations": ("data",),       # A tables: k-means over each input's own codes, on the host
    "outputs": ("estimate",),       # C tables: k-means over an fp32 run of this input (compiler/lower.py)
    "pick": ("host",),              # A and B indices: the host's nearest entry by value
}
FIT_CHOICES = {"method": ("kmeans",), "init": ("quantile",)}

#: What the kernel path can follow. Each is a fact of spike, the emitters or the chip's own
#: requantizer, not a preference: a recipe asking for anything else is refused by check().
KERNEL_DIM = 16             # the emitters' tile plan and libgemmini's DIM
KERNEL_BLOCK = 32           # mx_host.h MX_BLOCK, compiler/formats.BLOCK, spike's GROUP
KERNEL_SCRATCHPAD = (4, 4096)   # libgemmini gemmini_params.h BANK_NUM, BANK_ROWS at DIM 16
KERNEL_ROUNDING = "rne"     # the requantizer (gemmini.cc) and mx_host.h round to nearest even
KERNEL_SCALE_FLOOR = 2.0 ** -23     # the fp8 requantizer floors the block max at FLT_EPSILON


class RecipeError(ValueError):
    """A recipe we refuse to run. Raised instead of silently falling back to a
    default, because a recipe that is quietly ignored produces a grade against a
    machine we did not build."""


@dataclass(frozen=True)
class MxFloatSpec:
    """One ``{expWidth, sigWidth}`` entry. ``sigWidth`` counts the implicit leading bit, so the
    mantissa is ``sigWidth - 1`` (term for term the ``acc_e``/``acc_m`` of gemmini.cc)."""
    expWidth: int
    sigWidth: int
    count: int
    isRecoded: bool = False
    pad: bool = True

    @property
    def e(self) -> int:
        return self.expWidth

    @property
    def m(self) -> int:
        return self.sigWidth - 1

    @classmethod
    def parse(cls, obj: dict, where: str) -> "MxFloatSpec":
        if not isinstance(obj, dict):
            raise RecipeError(f"{where}: expected an object, got {type(obj).__name__}")
        _check_keys(obj, _MXFLOAT, where)
        for req in ("expWidth", "sigWidth", "count"):
            if req not in obj:
                raise RecipeError(f"{where}.{req} is required")
        return cls(int(obj["expWidth"]), int(obj["sigWidth"]), int(obj["count"]),
                   bool(obj.get("isRecoded", False)), bool(obj.get("pad", True)))


@dataclass(frozen=True)
class LutUnit:
    """``mx.lut``: the LUT unit as built (GemminiLUTConfig). Field for field the RTL's own."""
    projection: str             # projFormat: which finders the requantizer has (LUT_SERVES)
    entry_bits: int             # rdataWidth: bits per LUT entry
    index_bits: int             # raddrWidth: log2 entries per LUT
    tables: tuple[int, ...]     # numEntries: LUTs each table holds, by MX_LOAD_LUT sel (0 B, 1 A, 2 C)
    word_bits: tuple[int, ...]  # numBits: one LUT's write word
    group_bits: int             # lutUpdateRegularityWidth: the width of the G register
    act_code_bits: int          # actCodeWidth: 0 = rdataWidth (asymmetric builds set it)
    wei_code_bits: int          # weiCodeWidth: 0 = rdataWidth

    @property
    def serves(self) -> tuple[str, ...]:
        return LUT_SERVES[self.projection]


@dataclass(frozen=True)
class Hardware:
    """A validated hardware recipe."""
    name: str
    path: Path
    raw: dict
    dim: int
    prod: tuple[MxFloatSpec, ...]
    acc: tuple[MxFloatSpec, ...]
    prod_floor: int | None      # types.prodFloor: a product below 2^prodFloor is zero; None = no flush
    block: int                  # mx.scaleSize      -> gemmini.cc GROUP
    block_out: int              # mx.scaleSizeOut   -> gemmini.cc GROUP_OUT
    enable_lut: bool            # mx.enable_lut: a copy of the RTL field, recorded; nothing decides from it
    lut: LutUnit | None         # mx.lut: the LUT unit, or None for a build without one
    banks: int                  # scratchpad.banks  -> the emitters' bank_num
    rows: int                   # scratchpad.rows   -> the emitters' bank_rows
    clock_ns: float             # implementation.clock_ns    -> ppa, perf
    utilization: float          # implementation.utilization -> ppa
    description: str = ""
    provenance: dict = field(default_factory=dict)

    @property
    def prod_e(self) -> int:
        return self.prod[0].e

    @property
    def prod_m(self) -> int:
        return self.prod[0].m

    @property
    def acc_e(self) -> tuple[int, ...]:
        return tuple(s.e for s in self.acc)

    @property
    def acc_m(self) -> tuple[int, ...]:
        return tuple(s.m for s in self.acc)

    def hardware(self) -> dict:
        """Every field that describes the machine: the file without its labels."""
        return {k: v for k, v in self.raw.items() if k not in _LABELS}

    def build_id(self) -> str:
        return _digest(self.hardware())

    def geometry(self) -> dict:
        """The scratchpad plan the emitters read (command buffer ``params``)."""
        return {"dim": self.dim, "bank_num": self.banks, "bank_rows": self.rows}

    def ladder(self) -> list[dict]:
        """The precision ladder, one entry per column: a run record states it outright."""
        return [{"col": i, "prod": f"e{pr.e}m{pr.m}", "acc": f"e{ac.e}m{ac.m}"}
                for i, (pr, ac) in enumerate(zip(self.prod, self.acc))]

    def ladder_lines(self) -> list[str]:
        """``ladder()`` run-length collapsed: one line per contiguous acc format."""
        lines, start = [], 0
        for i in range(1, self.dim + 1):
            same = (i < self.dim
                    and (self.acc[i].e, self.acc[i].m) == (self.acc[start].e, self.acc[start].m))
            if same:
                continue
            cols = f"col {start}" if i - start == 1 else f"col {start}-{i - 1}"
            lines.append(f"{cols:<12} acc=e{self.acc[start].e}m{self.acc[start].m}"
                         f"  prod=e{self.prod[start].e}m{self.prod[start].m}")
            start = i
        return lines

    def describe(self) -> str:
        uniform_acc = len(set(zip(self.acc_e, self.acc_m))) == 1
        acc = (f"{self.acc_e[0]}e{self.acc_m[0]}m x{self.dim}" if uniform_acc
               else f"e{self.acc_e[0]}..{self.acc_e[-1]} m{self.acc_m[0]}..{self.acc_m[-1]}")
        flush = "none" if self.prod_floor is None else f"2^{self.prod_floor}"
        return (f"{self.name}  dim={self.dim}  prod=e{self.prod_e}m{self.prod_m} flush<{flush}  "
                f"acc[{acc}]  block={self.block}")


@dataclass(frozen=True)
class Fit:
    """How one table's 16 entries are fitted to its group's codes (compiler/codebook.build_codebooks)."""
    method: str                 # "kmeans": weighted 1-D k-means over the distinct codes
    init: str                   # "quantile": seeds spread by mass
    max_iters: int              # Lloyd passes at most


@dataclass(frozen=True)
class Lut:
    """How a LUT format's tables are made: the run recipe's ``lut`` block, every field written.

    group        G: one LUT per 2**G rows of A, columns of B, rows of C (gemmini_mxquant_config_mvout)
    weights      where B tables come from
    activations  where A tables come from (inputs the host quantizes)
    outputs      where C tables come from (outputs the chip requantizes, in a chain)
    pick         who picks A and B indices
    fit          how a table is fitted

    The values each accepts are ``LUT_CHOICES`` / ``FIT_CHOICES``: what the compiler implements.
    """
    group: int
    weights: str
    activations: str
    outputs: str
    pick: str
    fit: Fit


@dataclass(frozen=True)
class Vector:
    """The run recipe's ``vector`` block: the precision of each vector op on the perplexity path.

    softmax   attention's scale, mask and softmax (mxq.nn.attend; an attention module with no core rule gets
              mxq's exact core so its softmax can be rounded)
    rmsnorm   every RMSNorm module
    None = as transformers computes it; "bf16" = each step rounded to bf16 (mxq/nn/_vector.py)."""
    softmax: str | None
    rmsnorm: str | None

    def mxq(self) -> dict:
        """What mxq.nn.patch takes as ``vector``."""
        return {"softmax": self.softmax, "rmsnorm": self.rmsnorm}


@dataclass(frozen=True)
class Run:
    """A validated run recipe. The defaults are ``config/run/default.json``: what the chip does."""
    name: str = "default"
    operand_fmt: str = "fp8_e4m3"
    rounding: str = KERNEL_ROUNDING
    scale_floor: float = KERNEL_SCALE_FLOOR
    reduce: str = "hardware"
    allow_lossy_chain: bool = False
    fp32_tol: float = 0.15
    lut: Lut | None = None      # None: no lut block (the kernel path's LUTs are the compiler's; perplexity is full grid)
    scale: str = "mxgemmini"    # SCALES; "mxgemmini" is absent from run_id so every recipe without it keeps its id
    vector: Vector | None = None    # None: no vector block (softmax and RMSNorm as transformers computes them)
    description: str = ""
    path: Path | None = None

    def fields(self) -> dict:
        """Every field that changes a number. ``lut`` and ``vector`` only when set, so a recipe without them
        keeps its run_id."""
        return {k: v for k, v in asdict(self).items()
                if k not in _LABELS and k != "path" and not (k in ("lut", "vector") and v is None)
                and not (k == "scale" and v == "mxgemmini")}

    def run_id(self) -> str:
        return _digest(self.fields())

    def describe(self) -> str:
        lut = "" if self.lut is None else (
            f"  lut G={self.lut.group} B {self.lut.weights} A {self.lut.activations} C {self.lut.outputs}"
            f" pick {self.lut.pick} {self.lut.fit.method}/{self.lut.fit.init}x{self.lut.fit.max_iters}")
        scale = "" if self.scale == "mxgemmini" else f"  scale {self.scale}"
        vec = "" if self.vector is None else (
            "  vector " + " ".join(f"{op} {getattr(self.vector, op) or 'hf'}" for op in VECTOR_OPS))
        return (f"{self.name}  {self.operand_fmt}  {self.rounding}  floor {self.scale_floor:g}  "
                f"reduce {self.reduce}{lut}{scale}{vec}")


def _digest(obj) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _check_keys(obj: dict, allowed: set[str], where: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise RecipeError(f"unrecognized key(s) in {where}: {', '.join(sorted(extra))}. "
                          f"allowed: {', '.join(sorted(allowed))}")


def _read(path: str | Path, folder: Path, what: str) -> tuple[dict, Path]:
    p = Path(path)
    if not p.exists() and not p.suffix:
        p = folder / f"{p.name}.json"
    if not p.exists():
        raise RecipeError(f"no such {what} recipe: {path} (looked in {folder})")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecipeError(f"{p}: invalid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise RecipeError(f"{p}: expected an object")
    return raw, p


def load_hardware(path: str | Path) -> Hardware:
    """A name in ``config/hardware/`` or a path to a .json."""
    raw, p = _read(path, HARDWARE_DIR, "hardware")
    return parse_hardware(raw, path=p)


def parse_hardware(raw: dict, *, path: Path | None = None) -> Hardware:
    _check_keys(raw, _HW_TOP, "hardware recipe")
    sec = {}
    for key, allowed in _HW_SECTIONS.items():
        s = raw.get(key, {})
        if not isinstance(s, dict):
            raise RecipeError(f"{key}: expected an object, got {type(s).__name__}")
        _check_keys(s, allowed, key)
        sec[key] = s
    for key in ("array", "types", "mx", "scratchpad", "implementation"):
        if key not in raw:
            raise RecipeError(f"{key} is required")

    array, types, mx, spad, impl = (sec[k] for k in ("array", "types", "mx", "scratchpad", "implementation"))
    rows = int(array.get("meshRows", 16))
    cols = int(array.get("meshColumns", 16))
    if rows != cols:
        raise RecipeError(f"array: meshRows ({rows}) != meshColumns ({cols}); the backend "
                          "plans a square PE tile (mxgemm_emit uses a single `dim`)")
    prod = _parse_list(types, "meshProdPrecisionList", cols)
    acc = _parse_list(types, "meshAccPrecisionList", cols)
    if len({(s.e, s.m) for s in prod}) != 1:
        raise RecipeError("types.meshProdPrecisionList: entries differ. The spike model "
                          "declares ONE scalar prod_e/prod_m (gemmini.cc:1145), so a "
                          "non-uniform product precision cannot be represented on spike")
    if "prodFloor" not in types:
        raise RecipeError("types.prodFloor is required: the exponent below which a product is "
                          "flushed to zero (MxFPMul: -16), or null for no flush")
    floor = types["prodFloor"]
    for k in ("banks", "rows"):
        if k not in spad:
            raise RecipeError(f"scratchpad.{k} is required")
    for k in ("clock_ns", "utilization"):
        if k not in impl:
            raise RecipeError(f"implementation.{k} is required")
    for k in ("enable_lut", "lut"):
        if k not in mx:
            raise RecipeError(f"mx.{k} is required (mx.lut: the LUT unit as built, the RTL's GemminiLUTConfig, "
                              "or null for none)")
    return Hardware(
        name=raw.get("name") or (path.stem if path else "unnamed"),
        path=path or Path("<inline>"),
        raw=raw,
        dim=cols,
        prod=prod,
        acc=acc,
        prod_floor=None if floor is None else int(floor),
        block=int(mx.get("scaleSize", 32)),
        block_out=int(mx.get("scaleSizeOut", mx.get("scaleSize", 32))),
        enable_lut=_bool(mx["enable_lut"], "mx.enable_lut"),
        lut=_parse_lut_unit(mx["lut"]),
        banks=int(spad["banks"]),
        rows=int(spad["rows"]),
        clock_ns=float(impl["clock_ns"]),
        utilization=float(impl["utilization"]),
        description=raw.get("description", ""),
        provenance=raw.get("provenance", {}),
    )


def load_run(path: str | Path) -> Run:
    """A name in ``config/run/`` or a path to a .json."""
    raw, p = _read(path, RUN_DIR, "run")
    return parse_run(raw, path=p)


def parse_run(raw: dict, *, path: Path | None = None) -> Run:
    _check_keys(raw, _RUN_KEYS, "run recipe")
    missing = sorted(_RUN_KEYS - _RUN_OPTIONAL - set(raw))
    if missing:
        raise RecipeError(f"run recipe: {', '.join(missing)} required (write every field; "
                          "config/run/default.json is the template)")
    scale = str(raw.get("scale", "mxgemmini"))
    if scale not in SCALES:
        raise RecipeError(f"run recipe: scale {scale!r}; choose from {', '.join(SCALES)}")
    return Run(name=raw["name"], operand_fmt=str(raw["operand_fmt"]), rounding=str(raw["rounding"]),
               scale_floor=float(raw["scale_floor"]), reduce=str(raw["reduce"]),
               allow_lossy_chain=bool(raw["allow_lossy_chain"]), fp32_tol=float(raw["fp32_tol"]),
               lut=_parse_lut(raw.get("lut"), path), scale=scale, vector=_parse_vector(raw.get("vector")),
               description=raw.get("description", ""), path=path)


def _parse_vector(obj) -> Vector | None:
    """The run recipe's ``vector`` block: every op written, each null or "bf16"."""
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise RecipeError(f"vector: an object with {', '.join(VECTOR_OPS)}, each null or \"bf16\"")
    _check_keys(obj, set(VECTOR_OPS), "vector")
    missing = sorted(set(VECTOR_OPS) - set(obj))
    if missing:
        raise RecipeError(f"vector: {', '.join(missing)} required (null = as transformers computes it)")
    for op in VECTOR_OPS:
        if obj[op] not in VECTOR_CHOICES:
            raise RecipeError(f"vector.{op} {obj[op]!r}: null or \"bf16\"")
    return Vector(**{op: obj[op] for op in VECTOR_OPS})


def _bool(v, where: str) -> bool:
    if not isinstance(v, bool):
        raise RecipeError(f"{where} {v!r}: true or false")
    return v


def _int(v, where: str, lo: int = 0) -> int:
    if not isinstance(v, int) or isinstance(v, bool) or v < lo:
        raise RecipeError(f"{where} {v!r}: an integer >= {lo}")
    return v


def _parse_lut_unit(obj) -> LutUnit | None:
    """``mx.lut``: every GemminiLUTConfig field written, held to the RTL's own ``require``s."""
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise RecipeError("mx.lut: an object (GemminiLUTConfig's fields) or null")
    _check_keys(obj, _HW_LUT_KEYS, "mx.lut")
    missing = sorted(_HW_LUT_KEYS - set(obj))
    if missing:
        raise RecipeError(f"mx.lut: {', '.join(missing)} required")
    proj = obj["projFormat"]
    if proj not in LUT_SERVES:
        raise RecipeError(f"mx.lut.projFormat {proj!r}; choose from {', '.join(LUT_SERVES)}")
    entry = _int(obj["rdataWidth"], "mx.lut.rdataWidth", 1)
    index = _int(obj["raddrWidth"], "mx.lut.raddrWidth", 1)
    lists = {}
    for k in ("numEntries", "numBits"):
        v = obj[k]
        if not isinstance(v, list) or len(v) != 3:
            raise RecipeError(f"mx.lut.{k} {v!r}: three integers, one per table (B, A, C)")
        lists[k] = tuple(_int(x, f"mx.lut.{k}", 1) for x in v)
    if proj in _FP8_PROJECTIONS and entry != 8:
        raise RecipeError(f"mx.lut: {proj} requires rdataWidth 8, got {entry} (MxConfigFragments.scala:66)")
    if any(b != (1 << index) * entry for b in lists["numBits"]):
        raise RecipeError(f"mx.lut.numBits {list(lists['numBits'])}: each must be 2**raddrWidth * rdataWidth = "
                          f"{(1 << index) * entry} (MxConfigFragments.scala:70)")
    return LutUnit(projection=proj, entry_bits=entry, index_bits=index, tables=lists["numEntries"],
                   word_bits=lists["numBits"],
                   group_bits=_int(obj["lutUpdateRegularityWidth"], "mx.lut.lutUpdateRegularityWidth", 1),
                   act_code_bits=_int(obj["actCodeWidth"], "mx.lut.actCodeWidth"),
                   wei_code_bits=_int(obj["weiCodeWidth"], "mx.lut.weiCodeWidth"))


def _parse_lut(obj, path: Path | None) -> Lut | None:
    """The run recipe's ``lut`` block: every key written, each one of the values the compiler implements."""
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise RecipeError(f"lut: expected an object with {', '.join(sorted(_LUT_KEYS))}")
    _check_keys(obj, _LUT_KEYS, "lut")
    missing = sorted(_LUT_KEYS - set(obj))
    if missing:
        raise RecipeError(f"lut: {', '.join(missing)} required")
    for k, choices in LUT_CHOICES.items():
        if obj[k] not in choices:
            raise RecipeError(f"lut.{k} {obj[k]!r}: implemented today: {', '.join(choices)} "
                              "(planning/LUT_integration.md lists the rest)")
    fit = obj["fit"]
    if not isinstance(fit, dict):
        raise RecipeError(f"lut.fit: an object with {', '.join(sorted(_FIT_KEYS))}")
    _check_keys(fit, _FIT_KEYS, "lut.fit")
    missing = sorted(_FIT_KEYS - set(fit))
    if missing:
        raise RecipeError(f"lut.fit: {', '.join(missing)} required")
    for k, choices in FIT_CHOICES.items():
        if fit[k] not in choices:
            raise RecipeError(f"lut.fit.{k} {fit[k]!r}: implemented today: {', '.join(choices)}")
    return Lut(group=_int(obj["group"], "lut.group"), weights=obj["weights"], activations=obj["activations"],
               outputs=obj["outputs"], pick=obj["pick"],
               fit=Fit(method=fit["method"], init=fit["init"], max_iters=_int(fit["max_iters"], "lut.fit.max_iters", 1)))


def check(hw: Hardware, run: Run, path: str) -> None:
    """Refuse what ``path`` cannot follow: "kernel" (spike, the emitters, the chip's requantizer) or
    "perplexity" (mxq alone)."""
    from config import scheme                   # the format and reducer vocabulary
    scheme.mxq_format(run.operand_fmt)
    if run.reduce not in scheme.REDUCERS:
        raise RecipeError(f"run {run.name}: reduce {run.reduce!r}; choose from {', '.join(scheme.REDUCERS)}")
    codebook = scheme.is_codebook(run.operand_fmt)
    if run.lut is not None and not codebook:
        raise RecipeError(f"run {run.name}: a lut block for {run.operand_fmt}, which is not a LUT format "
                          "(the LUT formats are fp8_e4m3_quad, fp8_e5m2, fp6_e3m2, fp6_e2m3)")
    if run.lut is not None:
        _check_lut(hw, run)
    if run.scale != "mxgemmini" and run.lut is not None:
        raise RecipeError(f"run {run.name}: scale {run.scale} with a lut block; the chip's tables index codes "
                          "placed the mxgemmini way")
    if path == "perplexity":
        return                  # a codebook format without a lut block: quantized straight to its grid (LUT off)
    if run.vector is not None:
        raise RecipeError(f"run {run.name}: a vector block on the kernel path, which grades matmul kernels; the "
                          "vector ops' precision is the perplexity path's (python -m models.mxquant)")
    if path != "kernel":
        raise ValueError(f"path {path!r}: 'kernel' or 'perplexity'")
    refusals = []
    if codebook and run.lut is None:
        refusals.append(f"{run.operand_fmt} without a lut block: the chip's requantizer sends it through the LUT "
                        f"(config/run/{run.operand_fmt}.json is the template)")
    if hw.dim != KERNEL_DIM:
        refusals.append(f"mesh {hw.dim}x{hw.dim}: the emitters and libgemmini are built for {KERNEL_DIM}")
    if hw.block != KERNEL_BLOCK:
        refusals.append(f"mx.scaleSize {hw.block}: spike, mx_host.h and the compiler block in {KERNEL_BLOCK}s")
    if (hw.banks, hw.rows) != KERNEL_SCRATCHPAD:
        refusals.append(f"scratchpad {hw.banks}x{hw.rows}: libgemmini is built with "
                        f"BANK_NUM {KERNEL_SCRATCHPAD[0]}, BANK_ROWS {KERNEL_SCRATCHPAD[1]}")
    if run.rounding != KERNEL_ROUNDING:
        refusals.append(f"rounding {run.rounding}: the chip's requantizer and mx_host.h round to nearest even")
    if run.scale != "mxgemmini":
        refusals.append(f"scale {run.scale}: the chip's requantizer places the block maximum in [1, 2)")
    if run.scale_floor != KERNEL_SCALE_FLOOR:
        refusals.append(f"scale_floor {run.scale_floor:g}: the chip's requantizer floors the block max at 2^-23")
    if run.reduce != "hardware":
        refusals.append(f"reduce {run.reduce}: the kernel path grades the chip, whose reducer is the recipe's ladder")
    if refusals:
        raise RecipeError(f"{hw.name} + run {run.name} on the kernel path: " + "; ".join(refusals)
                          + " (the perplexity path, python -m models.mxquant, runs these)")


def _check_lut(hw: Hardware, run: Run) -> None:
    """A LUT format on both paths: the build must have a LUT unit that serves it, the run must say how."""
    fmt, unit = run.operand_fmt, hw.lut
    if run.lut is None:
        raise RecipeError(f"run {run.name}: {fmt} is a LUT format, so the run recipe needs a lut block "
                          f"(config/run/{fmt}.json is the template)")
    if unit is None:
        raise RecipeError(f"{hw.name}: mx.lut is null, a build without a LUT unit, so it cannot run {fmt}")
    if fmt not in unit.serves:
        raise RecipeError(f"{hw.name}: its {unit.projection} LUT unit serves {', '.join(unit.serves)}, not {fmt} "
                          "(QuantLut.scala's finders); config/hardware/ has a build for each LUT format")
    from compiler import formats
    if unit.index_bits != formats.get(fmt, where="check").bits:
        raise RecipeError(f"{hw.name}: mx.lut.raddrWidth {unit.index_bits}, but {fmt} sends "
                          f"{formats.get(fmt, where='check').bits}-bit indices")
    width = formats.get(fmt, where="check").entry_bits
    if width != unit.entry_bits:
        raise RecipeError(f"{hw.name}: {fmt} on {unit.entry_bits}-bit LUT entries ({unit.projection}): the compiler "
                          f"and mxq.lut hold {width}-bit {fmt} entries, and wider entries are not modelled")
    if unit.act_code_bits or unit.wei_code_bits:
        raise RecipeError(f"{hw.name}: mx.lut.actCodeWidth/weiCodeWidth set (an asymmetric LUT build), "
                          "which no model follows yet")
    if run.lut.group >= 1 << unit.group_bits:
        raise RecipeError(f"run {run.name}: lut.group {run.lut.group} does not fit the "
                          f"{unit.group_bits}-bit G register (mx.lut.lutUpdateRegularityWidth)")


def lut_settings(hw: Hardware, run: Run):
    """The compiler's codebook settings (``compiler.codebook.Settings``) from the two recipes, or None for a
    direct format. ``check`` has already held them to the build."""
    if run.lut is None:
        return None
    from compiler.codebook import Settings
    return Settings(group=run.lut.group, max_iters=run.lut.fit.max_iters)


def emitter_params(hw: Hardware, run: Run) -> dict:
    """``cb["params"]``: the scratchpad plan, and for a LUT format G and each table's capacity."""
    params = hw.geometry()
    if run.lut is not None:
        params |= {"lut_group": run.lut.group, "lut_tables": list(hw.lut.tables)}
    return params


def _parse_list(types: dict, key: str, expect: int) -> tuple[MxFloatSpec, ...]:
    if key not in types:
        raise RecipeError(f"types.{key} is required (it IS the quantization ladder)")
    items = types[key]
    if not isinstance(items, list):
        raise RecipeError(f"types.{key}: expected an array of {expect} MxFloat objects")
    if len(items) != expect:
        raise RecipeError(f"types.{key}: length {len(items)} must equal meshColumns ({expect}) "
                          "— one entry per column of the systolic array")
    return tuple(MxFloatSpec.parse(o, f"types.{key}[{i}]") for i, o in enumerate(items))


def _listing(folder: Path, load, describe) -> dict[str, str]:
    out = {}
    for p in sorted(folder.glob("*.json")):
        try:
            out[p.stem] = describe(load(p))
        except RecipeError as exc:
            out[p.stem] = f"INVALID: {exc}"
    return out


def list_hardware() -> dict[str, str]:
    return _listing(HARDWARE_DIR, load_hardware, Hardware.describe)


def list_runs() -> dict[str, str]:
    return _listing(RUN_DIR, load_run, Run.describe)


#: Flags the run recipe replaced, and the field that carries each now.
REMOVED_FLAGS = {"--dtype": "operand_fmt", "--rounding-mode": "rounding", "--scale-floor": "scale_floor",
                 "--reduce": "reduce", "--tol": "fp32_tol", "--allow-lossy-chain": "allow_lossy_chain",
                 "--seam": None}


def removed_flag(argv: list[str]) -> str | None:
    """The refusal for a flag the run recipe replaced, or None."""
    for a in argv:
        flag = a.split("=", 1)[0]
        if flag in REMOVED_FLAGS:
            field = REMOVED_FLAGS[flag]
            if field is None:
                return f"{flag} is gone: it chose a workaround for a requantizer behaviour that no longer exists"
            return (f"{flag} is now the run recipe's '{field}' field: copy config/run/default.json, set "
                    f"{field}, and pass --run <that file> (config/run/ has default, exact, bf16_tiles, fp4_e2m1)")
    return None
