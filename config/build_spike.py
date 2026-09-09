"""Build (and cache) the spike model for a recipe.

libgemmini's MX precision ladder is compile-time: ``prod_e``/``prod_m``, the 16-entry
``acc_e``/``acc_m`` tables and the block-scale group are C literals in
``gemmini.cc``, not runtime fields. So a recipe that changes them needs its own
``libgemmini.so``, and running it against the stock one would fail in the worst
possible way -- silently, with the golden honouring the recipe, the device ignoring
it, and the report blaming the hardware for a mismatch we caused.

We never edit the submodule. The four sources are copied into
``out/builds/<build_id>/src`` and patched there; spike is pointed at the result
through the ``MX_LIBGEMMINI`` override the runner already honours
(``runner.py:84`` -> ``--extlib=``).

    python -m config.build_spike --config flat_acc4
    python -m config.build_spike --list

Cost: about 30 seconds on a miss, zero on a hit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO / "out" / "builds"

#: The four files that make up the model.
SOURCES = ("gemmini.cc", "gemmini.h", "gemmini_params.h", "mx_fp_math.h")

#: NOT `g++` from PATH. scripts/env.sh puts chipyard's conda bin first, and conda's
#: g++ (13.2) links a libstdc++ newer than the one spike's DT_RPATH pins (spack 12.2,
#: max GLIBCXX_3.4.30). DT_RPATH outranks LD_LIBRARY_PATH, so such a build fails at
#: dlopen with no useful message. Resolve the compiler explicitly.
DEFAULT_GXX = "/orcd/software/core/001/spack/pkg/gcc/12.2.0/yt6vabm/bin/g++"


class BuildError(RuntimeError):
    pass


def chipyard_root() -> Path:
    cy = os.environ.get("MERLIN_CHIPYARD") or os.environ.get("CHIPYARD_ROOT")
    if not cy:
        cy = str(Path.home() / "orcd/scratch/npu-exploration/chipyard-graphics")
    return Path(cy)


def upstream_dir() -> Path:
    d = chipyard_root() / "generators/gemmini/software/libgemmini"
    if not (d / "gemmini.cc").exists():
        raise BuildError(f"libgemmini sources not found at {d}; set MERLIN_CHIPYARD")
    return d


def _sub_once(text: str, pattern: str, repl: str, expect: int, what: str) -> str:
    """Substitute with an asserted match count.

    A count that has moved means upstream libgemmini changed shape underneath us. Far
    better to stop here than to half-patch the model and grade a kernel against a
    machine that is part one recipe and part another.
    """
    out, n = re.subn(pattern, repl, text)
    if n != expect:
        raise BuildError(
            f"{what}: expected {expect} match(es) of /{pattern}/ in gemmini.cc, found {n}. "
            "libgemmini has changed; re-check gemmini.cc:1145-1148 and the GROUP_OUT sites "
            "before trusting any build")
    return out


def patch(src: str, recipe) -> tuple[str, list[str]]:
    """Rewrite the precision literals to the recipe's. Returns (text, changes)."""
    changes = []
    acc_e = ",".join(str(v) for v in recipe.acc_e)
    acc_m = ",".join(str(v) for v in recipe.acc_m)

    src = _sub_once(src, r"const int prod_e = \d+, prod_m = \d+;",
                    f"const int prod_e = {recipe.prod_e}, prod_m = {recipe.prod_m};",
                    1, "product precision")
    changes.append(f"prod_e={recipe.prod_e} prod_m={recipe.prod_m}")

    src = _sub_once(src, r"const int8_t acc_e\[16\] = \{[^}]*\};",
                    f"const int8_t acc_e[16] = {{{acc_e}}};", 1, "accumulator exponents")
    src = _sub_once(src, r"const int8_t acc_m\[16\] = \{[^}]*\};",
                    f"const int8_t acc_m[16] = {{{acc_m}}};", 1, "accumulator mantissas")
    changes.append(f"acc_e=[{acc_e}]")
    changes.append(f"acc_m=[{acc_m}]")

    src = _sub_once(src, r"const int GROUP = \d+;", f"const int GROUP = {recipe.block};",
                    1, "input block-scale group")
    # Three sites, one per operand format path (fp8 / fp4 / fp6).
    src = _sub_once(src, r"const int GROUP_OUT = \d+;",
                    f"const int GROUP_OUT = {recipe.block_out};", 3, "output block-scale group")
    changes.append(f"GROUP={recipe.block} GROUP_OUT={recipe.block_out}")
    return src, changes


def _glibcxx_max(so: Path) -> str | None:
    try:
        out = subprocess.run(["objdump", "-p", str(so)], capture_output=True, text=True,
                             timeout=60).stdout
    except Exception:
        return None
    vs = sorted({tuple(int(x) for x in v.split("_")[1].split("."))
                 for v in re.findall(r"GLIBCXX_[0-9.]+", out)})
    return ".".join(str(x) for x in vs[-1]) if vs else None


def build(recipe, *, force: bool = False, gxx: str | None = None, quiet: bool = False) -> Path:
    """Return the path to this recipe's libgemmini.so, building it if absent."""
    bid = recipe.build_id()
    out = BUILD_ROOT / bid
    so = out / "libgemmini.so"

    if so.exists() and not force:
        if not quiet:
            print(f"[cache hit ] {bid}  {so}")
        return so

    up = upstream_dir()
    gxx = gxx or os.environ.get("MX_HOST_GXX") or DEFAULT_GXX
    if not Path(gxx).exists():
        raise BuildError(f"compiler not found: {gxx}. Set MX_HOST_GXX to a g++ whose "
                         "libstdc++ is no newer than the one spike's DT_RPATH points at")

    srcdir = out / "src"
    srcdir.mkdir(parents=True, exist_ok=True)
    for f in SOURCES:
        shutil.copy(up / f, srcdir / f)

    text = (srcdir / "gemmini.cc").read_text(encoding="utf-8")
    patched, changes = patch(text, recipe)
    (srcdir / "gemmini.cc").write_text(patched, encoding="utf-8")

    riscv = os.environ.get("RISCV") or str(chipyard_root() / ".conda-env/riscv-tools")
    cmd = [gxx, "-L", f"{riscv}/lib", f"-Wl,-rpath,{riscv}/lib", "-shared",
           "-o", str(so), "-std=c++17", "-I", f"{riscv}/include", "-I", str(srcdir),
           "-fPIC", "-O3", str(srcdir / "gemmini.cc")]
    if not quiet:
        print(f"[build     ] {bid} ({recipe.name})")
        for c in changes:
            print(f"             {c}")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        raise BuildError(f"g++ failed:\n{r.stderr[-3000:]}")

    # The failure this guards against is a dlopen error with no diagnostic, so compare
    # against the stock model rather than guessing an acceptable version.
    stock = up / "libgemmini.so"
    ours, theirs = _glibcxx_max(so), (_glibcxx_max(stock) if stock.exists() else None)
    if ours and theirs and tuple(map(int, ours.split("."))) > tuple(map(int, theirs.split("."))):
        raise BuildError(
            f"built .so needs GLIBCXX_{ours} but the working stock model needs only "
            f"GLIBCXX_{theirs}. spike has a DT_RPATH that outranks LD_LIBRARY_PATH, so this "
            f"would fail to load with no message. Build with a g++ matching {theirs}")

    (out / "recipe.json").write_text(json.dumps(recipe.raw, indent=2) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({
        "build_id": bid,
        "recipe_name": recipe.name,
        "built": datetime.now().isoformat(timespec="seconds"),
        "compiler": gxx,
        "changes": changes,
        "upstream": str(up),
        "upstream_gemmini_cc_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
        "so_sha256": hashlib.sha256(so.read_bytes()).hexdigest()[:16],
        "glibcxx_max": ours,
    }, indent=2) + "\n", encoding="utf-8")

    if not quiet:
        print(f"[built     ] {so} ({so.stat().st_size} B, needs GLIBCXX_{ours})")
    return so


def resolve(recipe, *, quiet: bool = True) -> Path | None:
    """The .so to run this recipe on, or None to use the stock model.

    A recipe whose hardware matches the stock build needs nothing built: the shipped
    model already IS that machine.
    """
    from .recipe import load
    if recipe.build_id() == load("baseline").build_id():
        return None
    return build(recipe, quiet=quiet)


def main() -> int:
    from .recipe import RECIPES_DIR, RecipeError, load
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="recipe name or path")
    ap.add_argument("--all", action="store_true", help="build every recipe in config/recipes/")
    ap.add_argument("--list", action="store_true", help="show the build cache")
    ap.add_argument("--force", action="store_true", help="rebuild even on a cache hit")
    ap.add_argument("--gxx", default=None, help=f"compiler (default {DEFAULT_GXX})")
    a = ap.parse_args()

    if a.list:
        if not BUILD_ROOT.exists():
            print(f"no builds yet ({BUILD_ROOT})")
            return 0
        for d in sorted(BUILD_ROOT.iterdir()):
            meta = d / "meta.json"
            if meta.exists():
                m = json.loads(meta.read_text())
                print(f"  {m['build_id']}  {m['recipe_name']:14s} built {m['built']}  "
                      f"{', '.join(m['changes'])}")
        return 0

    names = ([p.stem for p in sorted(RECIPES_DIR.glob("*.json"))] if a.all
             else [a.config or "baseline"])
    for n in names:
        try:
            r = load(n)
            if r.build_id() == load("baseline").build_id():
                print(f"[stock     ] {n}: hardware identical to baseline, no build needed")
                continue
            build(r, force=a.force, gxx=a.gxx)
        except (RecipeError, BuildError) as exc:
            print(f"ERROR {n}: {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
