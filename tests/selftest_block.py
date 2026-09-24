"""models/mxquant/block.py must equal MXQuant's quantize_mx_block32(round_mode="even"), bit for bit.

The operand encoder (app/mxq_golden.py) used to call MXQuant; it now calls this module. The claim
that nothing changed is settled here, on a frozen fixture generated FROM MXQuant, so the test runs on
a machine without the MXQuant clone. When the clone is present the live comparison runs as well.

Inputs cover what a random tensor does not: exact ties, element subnormals, all-zero blocks, blocks
whose max is below the 2^-23 floor (where mxq's default 1e-38 floor would differ), a block max below
1e-38, ragged shapes (not multiples of 32), both blocking axes, every operand format.

    .venv/bin/python tests/selftest_block.py            # check against tests/oracle/block_fixture.npz
    .venv/bin/python tests/selftest_block.py --update   # regenerate the fixture from MXQuant (needs the clone)
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

FIXTURE = REPO / "tests" / "oracle" / "block_fixture.npz"
FORMATS = ("MXFP8_E4M3", "MXFP8_E5M2", "MXFP6_E3M2", "MXFP6_E2M3", "MXFP4")
AXES = ("row", "col")

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not cond:
        FAILURES.append(name)


def cases() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    wide = torch.randn(96, 64, generator=g) * torch.logspace(-3, 3, 64)
    ties = torch.arange(1.0, 2.0, 1 / 64).repeat(64, 1)                      # every tie of every grid <= 1/32
    sub = torch.randn(64, 64, generator=g) * 1e-3
    sub[0, :] = 1.0                                                            # block max 1 -> values hit subnormals
    zero = torch.zeros(64, 64)
    zero[0, :] = torch.randn(64, generator=g)                                  # one live row among zero blocks
    tiny = torch.full((64, 64), 2.0 ** -30)                                    # block max under the 2^-23 floor
    tiny[:, 0] = 1e-39                                                         # and under mxq's 1e-38 floor
    ragged = torch.randn(40, 50, generator=g)                                  # R, C not multiples of 32
    return {"wide": wide, "ties": ties, "subnormal": sub, "zero_blocks": zero, "tiny": tiny, "ragged": ragged}


def mxquant_reference():
    """MXQuant's quantize_mx_block32, or None when the clone is absent."""
    root = REPO / "MXQuant"
    if not (root / "end_to_end_linear" / "mx_block_quant.py").exists():
        return None
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from end_to_end_linear.mx_block_quant import quantize_mx_block32  # type: ignore
    return quantize_mx_block32


def update() -> int:
    ref = mxquant_reference()
    if ref is None:
        print(f"cannot update: MXQuant not found at {REPO / 'MXQuant'}")
        return 1
    out: dict[str, np.ndarray] = {}
    skipped = []
    for cname, V in cases().items():
        for fmt in FORMATS:
            for axis in AXES:
                try:
                    o = ref(V, fmt=fmt, axis=axis, round_mode="even")
                except Exception as exc:                      # a shape MXQuant does not accept
                    skipped.append(f"{cname}|{fmt}|{axis}: {type(exc).__name__}")
                    continue
                out[f"{cname}|{fmt}|{axis}|P"] = o.P.float().numpy()
                out[f"{cname}|{fmt}|{axis}|X"] = o.X.float().numpy()
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **out)
    print(f"wrote {FIXTURE} ({len(out) // 2} cases, {FIXTURE.stat().st_size / 1024:.0f} KB)")
    for s in skipped:
        print(f"  skipped {s}")
    return 0


def main() -> int:
    if "--update" in sys.argv:
        return update()
    import models
    if not models.paths():
        print(f"SKIP: {models.mxq_missing()}")
        return 0
    from models.mxquant import block

    print("\n[1] against the frozen MXQuant fixture ----------------------------")
    if not FIXTURE.exists():
        check("fixture present", False, f"{FIXTURE} missing -- run with --update where MXQuant is present")
    else:
        fx = np.load(FIXTURE)
        keys = sorted({k.rsplit("|", 1)[0] for k in fx.files})
        n_bad = 0
        allc = cases()
        for key in keys:
            cname, fmt, axis = key.split("|")
            o = block.quantize_mx_block32(allc[cname], fmt=fmt, axis=axis, round_mode="even")
            P_ok = torch.equal(o.P, torch.from_numpy(fx[key + "|P"]))
            X_ok = torch.equal(o.X, torch.from_numpy(fx[key + "|X"]))
            if not (P_ok and X_ok):
                n_bad += 1
                check(f"{key}", False, f"P {'ok' if P_ok else 'DIFF'}  X {'ok' if X_ok else 'DIFF'}")
        check(f"{len(keys)} fixture cases bit-identical (P and X)", n_bad == 0, f"{n_bad} differ")

    print("\n[2] against live MXQuant (when present) ---------------------------")
    ref = mxquant_reference()
    if ref is None:
        print("  skip  MXQuant clone not present; the fixture above is the evidence")
    else:
        n_bad = n_all = 0
        for cname, V in cases().items():
            for fmt in FORMATS:
                for axis in AXES:
                    try:
                        o_ref = ref(V, fmt=fmt, axis=axis, round_mode="even")
                    except Exception:
                        continue
                    n_all += 1
                    o = block.quantize_mx_block32(V, fmt=fmt, axis=axis, round_mode="even")
                    if not (torch.equal(o.P, o_ref.P.float()) and torch.equal(o.X, o_ref.X.float())):
                        n_bad += 1
                        check(f"{cname}|{fmt}|{axis}", False)
        check(f"{n_all} live cases bit-identical (P and X)", n_bad == 0, f"{n_bad} differ")

    print("\n[3] the encoder round-trips without MXQuant -----------------------")
    from app import mxq_golden
    g = torch.Generator().manual_seed(1)
    V = (torch.randn(64, 64, generator=g) * 3).numpy()
    codes, scales, books = mxq_golden.quantize_operand(V, side="a", dtype="fp8_e4m3")
    check("quantize_operand runs on the mxq-backed quantizer", codes.shape == (64, 64) and scales.shape == (2, 64))
    check("a rejected round_mode raises", _raises(lambda: block.quantize_mx_block32(torch.ones(32, 32), "MXFP8_E4M3", round_mode="floor")))

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}: {', '.join(FAILURES)}")
        return 1
    print("ALL CHECKS PASSED -- the mxq-backed block quantizer is MXQuant's, bit for bit.")
    return 0


def _raises(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


if __name__ == "__main__":
    raise SystemExit(main())
