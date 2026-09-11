"""Load, validate and fingerprint a hardware recipe.

The recipe answers one question: WHICH MX-Gemmini are we running the kernel on. Its
``array``/``types``/``mx``/``accumulator`` sections are the hardware itself and are
hashed into a ``build_id``; ``software``/``runtime`` are not, because they change no
gate and no line of the C model.

Field names and nesting mirror ``JsonGemminiConfig.scala`` exactly. That file is the
Chisel-side parser and it rejects unknown keys, so a recipe that loads here is a
recipe Verilator can elaborate — no translation layer to drift.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

RECIPES_DIR = Path(__file__).resolve().parent / "recipes"

#: Sections that describe the machine. Only these are hashed into the build id.
HARDWARE_SECTIONS = ("array", "types", "mx", "accumulator")

_ALLOWED_TOP = {"name", "description", "array", "types", "mx", "accumulator",
                "supported_backends", "software", "runtime", "provenance", "formats"}
_ALLOWED_ARRAY = {"meshRows", "meshColumns", "tileRows", "tileColumns"}
_ALLOWED_MXFLOAT = {"expWidth", "sigWidth", "count", "isRecoded", "pad"}
_ALLOWED_MX = {"scaleSize", "scaleSizeOut", "enable_lut", "lut"}
_ALLOWED_SOFTWARE = {"block", "target_code_exp", "seam", "intermediate_dtype"}
_ALLOWED_RUNTIME = {"operand_fmt", "out_dtype", "use_lut"}

#: Operand formats the emitter and libgemmini both know (mxgemm_emit.DTYPE_TO_FMT).
_OPERAND_FMT = {"fp8": "f8E4M3FN", "fp6": "f6E3M2FN", "fp4": "f4E2M1FN"}

#: Element (exponent, mantissa) bits per operand format (OCP MX element formats).
_ELEMENT_EM = {"fp8": (4, 3), "fp6": (3, 2), "fp4": (2, 1)}

_ALLOWED_FORMAT = {"tile", "codes_per_byte", "prod_frac_bits", "via_lut"}


class RecipeError(ValueError):
    """A recipe we refuse to run. Raised instead of silently falling back to a
    default, because a recipe that is quietly ignored produces a grade against a
    machine we did not build."""


@dataclass(frozen=True)
class MxFloatSpec:
    """One ``MxFloat`` from the Chisel config.

    ``sigWidth`` counts the implicit leading bit, so the mantissa field the C model
    uses is ``sigWidth - 1`` (verified: ConfigsFP.scala's 16-entry lists map term for
    term onto gemmini.cc's ``acc_e``/``acc_m``).
    """
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
        _check_keys(obj, _ALLOWED_MXFLOAT, where)
        for req in ("expWidth", "sigWidth", "count"):
            if req not in obj:
                raise RecipeError(f"{where}.{req} is required")
        return cls(int(obj["expWidth"]), int(obj["sigWidth"]), int(obj["count"]),
                   bool(obj.get("isRecoded", False)), bool(obj.get("pad", True)))


@dataclass(frozen=True)
class FormatSpec:
    """Per-operand-format execution geometry, DERIVED from the hashed sections.

    [COMPILE] — no JsonGemminiConfig.scala counterpart yet. A recipe may spell a
    ``formats`` block out explicitly, but only to CONFIRM the derivation: a value
    that contradicts it is a RecipeError, so ``build_id`` (which does not hash this
    section) can never disagree with behavior.
    """
    tile: tuple[int, int, int]      # hardware tile (M, N, K)
    codes_per_byte: int             # operand packing density
    prod_frac_bits: int             # exact product fraction width: 2*m + 1
    via_lut: bool = False           # decode goes through the LUT SRAMs


def derive_format(fmt: str, dim: int) -> FormatSpec:
    """Geometry/packing for one operand format, from mesh size + element width.

    Grounded in the reference models, not invented here:
      fp8: gemmini-rocc-tests/fp8_matmul_model.py  TILE=16, PROD_MANT_BITS=7
      fp4: gemmini-rocc-tests/fp4_matmul_model.py  TILE_M=TILE_N=32, TILE_K=16,
           PROD_MANT_BITS=3, two codes per byte
    tests/test_recipe_drift.py holds this derivation to those files.
    """
    if fmt == "fp6":
        raise RecipeError(
            "formats.fp6 is unpinned: fp6 decodes through QuantLut, which is being "
            "reworked upstream (gemmini-mx-cleanup WIP commits touch exactly this). "
            "Pin tile/packing/via_lut from lut_golden_model.py once it settles")
    if fmt not in _ELEMENT_EM:
        raise RecipeError(f"unknown operand format {fmt!r}; known: {sorted(_ELEMENT_EM)}")
    _, m = _ELEMENT_EM[fmt]
    pack = 2 if fmt == "fp4" else 1
    # Packed operands widen one tile's M/N footprint; reduction depth stays the
    # mesh depth.
    return FormatSpec(tile=(dim * pack, dim * pack, dim),
                      codes_per_byte=pack,
                      prod_frac_bits=2 * m + 1)


@dataclass(frozen=True)
class Recipe:
    """A validated recipe. ``build_id`` identifies the machine, nothing else."""
    name: str
    path: Path
    raw: dict
    dim: int
    prod: tuple[MxFloatSpec, ...]
    acc: tuple[MxFloatSpec, ...]
    block: int              # mx.scaleSize      -> gemmini.cc GROUP
    block_out: int          # mx.scaleSizeOut   -> gemmini.cc GROUP_OUT
    operand_fmt: str        # fp8 | fp6 | fp4
    out_dtype: str
    target_code_exp: int
    seam: str
    intermediate_dtype: str
    supported_backends: tuple[str, ...]
    description: str = ""
    provenance: dict = field(default_factory=dict)
    formats: dict = field(default_factory=dict)   # fmt -> FormatSpec, derived (not hashed)

    # --- the three things the models actually need -------------------------------

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

    @property
    def operand_mlir_dtype(self) -> str:
        return _OPERAND_FMT[self.operand_fmt]

    @property
    def format_spec(self) -> FormatSpec:
        """Execution geometry for the active ``runtime.operand_fmt``."""
        return self.formats[self.operand_fmt]

    def hardware(self) -> dict:
        """Just the sections that describe the machine, canonically ordered."""
        return {k: self.raw[k] for k in HARDWARE_SECTIONS if k in self.raw}

    def build_id(self) -> str:
        """sha256 over the hardware sections only.

        Two recipes differing only in ``software``/``runtime`` share a build, which is
        what makes an fp8-vs-fp4 sweep free: those are instruction fields, not gates.
        """
        blob = json.dumps(self.hardware(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def ladder(self) -> list[dict]:
        """The precision ladder expanded one entry per column.

        ``types.*`` is stored the way ConfigsFP.scala spells it -- a 16-element list
        whose ``count`` field is a Chisel literal, not a repeat factor -- so the JSON
        reads as 16 near-identical blocks and the banner squashes it to a range. A
        run record has to state which column carries which format outright, because
        that ladder IS the experiment.
        """
        return [
            {"col": i,
             "prod": f"e{pr.e}m{pr.m}",
             "acc": f"e{ac.e}m{ac.m}"}
            for i, (pr, ac) in enumerate(zip(self.prod, self.acc))
        ]

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
        return (f"{self.name}  dim={self.dim}  operand={self.operand_fmt}->{self.out_dtype}  "
                f"prod=e{self.prod_e}m{self.prod_m}  acc[{acc}]  block={self.block}")


def _check_keys(obj: dict, allowed: set[str], where: str) -> None:
    extra = set(obj) - allowed
    if extra:
        raise RecipeError(f"unrecognized key(s) in {where}: {', '.join(sorted(extra))}. "
                          f"allowed: {', '.join(sorted(allowed))}")


def _section(raw: dict, key: str, allowed: set[str]) -> dict:
    sec = raw.get(key, {})
    if not isinstance(sec, dict):
        raise RecipeError(f"{key}: expected an object, got {type(sec).__name__}")
    _check_keys(sec, allowed, key)
    return sec


def load(path: str | Path) -> Recipe:
    """Read and validate a recipe file."""
    p = Path(path)
    if not p.exists() and not p.suffix:
        p = RECIPES_DIR / f"{p.name}.json"
    if not p.exists():
        raise RecipeError(f"no such recipe: {path} (looked in {RECIPES_DIR})")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RecipeError(f"{p}: invalid JSON: {exc}") from exc
    return parse(raw, path=p)


def parse(raw: dict, *, path: Path | None = None) -> Recipe:
    if not isinstance(raw, dict):
        raise RecipeError("recipe root: expected an object")
    _check_keys(raw, _ALLOWED_TOP, "recipe root")

    array = _section(raw, "array", _ALLOWED_ARRAY)
    mx = _section(raw, "mx", _ALLOWED_MX)
    software = _section(raw, "software", _ALLOWED_SOFTWARE)
    runtime = _section(raw, "runtime", _ALLOWED_RUNTIME)

    types = raw.get("types", {})
    if not isinstance(types, dict):
        raise RecipeError("types: expected an object")

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

    fmt = runtime.get("operand_fmt", "fp8")
    if fmt not in _OPERAND_FMT:
        raise RecipeError(f"runtime.operand_fmt: {fmt!r} not one of {sorted(_OPERAND_FMT)}")
    if fmt != "fp8":
        # The field is the right shape -- these ARE runtime instruction bits, and
        # libgemmini implements all three formats -- but the SOFTWARE side stops at
        # fp8: app/mxquant.py encodes e4m3 only, so a recipe asking for fp4 would
        # quietly send fp8 codes and report an fp8 result under an fp4 label.
        raise RecipeError(
            f"runtime.operand_fmt={fmt!r} is declared but not wired. app/mxquant.py "
            "encodes FP8 E4M3 only (no e2m1/e3m2 encoder, and no 2-per-byte packing), "
            "so this recipe would silently run fp8. Wire the encoder first")
    seam = software.get("seam", "weight")
    if seam not in ("weight", "rescale"):
        raise RecipeError(f"software.seam: {seam!r} not one of 'weight', 'rescale'")

    formats_raw = raw.get("formats", {})
    if not isinstance(formats_raw, dict):
        raise RecipeError("formats: expected an object")
    derived = {f: derive_format(f, cols) for f in ("fp8", "fp4")}
    for f, spec in formats_raw.items():
        if f not in derived:
            derive_format(f, cols)  # fp6/unknown: raises with the right message
        if not isinstance(spec, dict):
            raise RecipeError(f"formats.{f}: expected an object")
        _check_keys(spec, _ALLOWED_FORMAT, f"formats.{f}")
        d = derived[f]
        want = {"tile": list(d.tile), "codes_per_byte": d.codes_per_byte,
                "prod_frac_bits": d.prod_frac_bits, "via_lut": d.via_lut}
        for k, v in spec.items():
            v2 = list(v) if isinstance(v, (list, tuple)) else v
            if v2 != want[k]:
                raise RecipeError(
                    f"formats.{f}.{k} = {v!r} contradicts the derivation ({want[k]!r}). "
                    "An explicit formats block may only confirm derived geometry; "
                    "fix the value or delete the block")

    backends = tuple(raw.get("supported_backends", ("spike",)))
    for b in backends:
        if b not in ("spike", "verilator"):
            raise RecipeError(f"supported_backends: unknown backend {b!r}")

    r = Recipe(
        name=raw.get("name") or (path.stem if path else "unnamed"),
        path=path or Path("<inline>"),
        raw=raw,
        dim=cols,
        prod=prod,
        acc=acc,
        block=int(mx.get("scaleSize", 32)),
        block_out=int(mx.get("scaleSizeOut", mx.get("scaleSize", 32))),
        operand_fmt=fmt,
        out_dtype=runtime.get("out_dtype", "bf16"),
        target_code_exp=int(software.get("target_code_exp", 2)),
        seam=seam,
        intermediate_dtype=software.get("intermediate_dtype", "f8E4M3FN"),
        supported_backends=backends,
        description=raw.get("description", ""),
        provenance=raw.get("provenance", {}),
        formats=derived,
    )
    _validate_backends(r)
    return r


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


def _validate_backends(r: Recipe) -> None:
    """Reject a recipe a declared backend physically cannot build.

    Better here than as a mismatch discovered after a build: a backend that silently
    ignores a field grades the kernel against a machine we never made.
    """
    if "spike" in r.supported_backends:
        if r.dim != 16:
            raise RecipeError(
                f"spike: dim={r.dim} unsupported. libgemmini declares `float A_col[16], "
                "B_row[16]` (gemmini.cc:1164) and `acc_e[16]/acc_m[16]`, so the C model is "
                "hard-wired to 16. Drop 'spike' from supported_backends or set dim=16")
        if r.block not in (32,):
            raise RecipeError(f"spike: mx.scaleSize={r.block}; only 32 is wired through "
                              "mxgemm_emit.BLOCK_SCALE_GROUP today")


def list_recipes() -> dict[str, str]:
    out = {}
    for p in sorted(RECIPES_DIR.glob("*.json")):
        try:
            out[p.stem] = load(p).describe()
        except RecipeError as exc:
            out[p.stem] = f"INVALID: {exc}"
    return out
