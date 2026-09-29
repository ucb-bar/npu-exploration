"""The two recipes: the hardware one (a design point) and the run one (how it is driven).

``config/hardware/<name>.json`` is the machine: mesh, product format, accumulator ladder, product
flush, block size, scratchpad, clock and utilization. Every field except ``name``, ``description``
and ``provenance`` is hashed into ``build_id``, which keys the spike build, the perplexity cache and
every record. ``config/run/<name>.json`` is how that machine is driven: operand format, rounding,
scale floor, reducer and the grading tolerance; ``run_id`` hashes it the same way. A run recipe
never triggers a build.

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
    "mx": {"scaleSize", "scaleSizeOut", "enable_lut"},
    "accumulator": {"acc_read_full_width", "acc_read_small_width"},
    "scratchpad": {"banks", "rows"},
    "implementation": {"clock_ns", "utilization"},
}
_MXFLOAT = {"expWidth", "sigWidth", "count", "isRecoded", "pad"}
_RUN_KEYS = {"name", "description", "operand_fmt", "rounding", "scale_floor", "reduce",
             "allow_lossy_chain", "fp32_tol"}

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
    enable_lut: bool
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
class Run:
    """A validated run recipe. The defaults are ``config/run/default.json``: what the chip does."""
    name: str = "default"
    operand_fmt: str = "fp8_e4m3"
    rounding: str = KERNEL_ROUNDING
    scale_floor: float = KERNEL_SCALE_FLOOR
    reduce: str = "hardware"
    allow_lossy_chain: bool = False
    fp32_tol: float = 0.15
    description: str = ""
    path: Path | None = None

    def fields(self) -> dict:
        """Every field that changes a number."""
        return {k: v for k, v in asdict(self).items() if k not in _LABELS and k != "path"}

    def run_id(self) -> str:
        return _digest(self.fields())

    def describe(self) -> str:
        return (f"{self.name}  {self.operand_fmt}  {self.rounding}  floor {self.scale_floor:g}  "
                f"reduce {self.reduce}")


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
        enable_lut=bool(mx.get("enable_lut", False)),
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
    missing = sorted(_RUN_KEYS - {"description"} - set(raw))
    if missing:
        raise RecipeError(f"run recipe: {', '.join(missing)} required (write every field; "
                          "config/run/default.json is the template)")
    return Run(name=raw["name"], operand_fmt=str(raw["operand_fmt"]), rounding=str(raw["rounding"]),
               scale_floor=float(raw["scale_floor"]), reduce=str(raw["reduce"]),
               allow_lossy_chain=bool(raw["allow_lossy_chain"]), fp32_tol=float(raw["fp32_tol"]),
               description=raw.get("description", ""), path=path)


def check(hw: Hardware, run: Run, path: str) -> None:
    """Refuse what ``path`` cannot follow: "kernel" (spike, the emitters, the chip's requantizer) or
    "perplexity" (mxq alone)."""
    from config import scheme                   # the format and reducer vocabulary
    scheme.mxq_format(run.operand_fmt)
    if run.reduce not in scheme.REDUCERS:
        raise RecipeError(f"run {run.name}: reduce {run.reduce!r}; choose from {', '.join(scheme.REDUCERS)}")
    if path == "perplexity":
        return
    if path != "kernel":
        raise ValueError(f"path {path!r}: 'kernel' or 'perplexity'")
    refusals = []
    if hw.dim != KERNEL_DIM:
        refusals.append(f"mesh {hw.dim}x{hw.dim}: the emitters and libgemmini are built for {KERNEL_DIM}")
    if hw.block != KERNEL_BLOCK:
        refusals.append(f"mx.scaleSize {hw.block}: spike, mx_host.h and the compiler block in {KERNEL_BLOCK}s")
    if (hw.banks, hw.rows) != KERNEL_SCRATCHPAD:
        refusals.append(f"scratchpad {hw.banks}x{hw.rows}: libgemmini is built with "
                        f"BANK_NUM {KERNEL_SCRATCHPAD[0]}, BANK_ROWS {KERNEL_SCRATCHPAD[1]}")
    if run.rounding != KERNEL_ROUNDING:
        refusals.append(f"rounding {run.rounding}: the chip's requantizer and mx_host.h round to nearest even")
    if run.scale_floor != KERNEL_SCALE_FLOOR:
        refusals.append(f"scale_floor {run.scale_floor:g}: the chip's requantizer floors the block max at 2^-23")
    if run.reduce != "hardware":
        refusals.append(f"reduce {run.reduce}: the kernel path grades the chip, whose reducer is the recipe's ladder")
    if refusals:
        raise RecipeError(f"{hw.name} + run {run.name} on the kernel path: " + "; ".join(refusals)
                          + " (the perplexity path, python -m models.mxquant, runs these)")


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
