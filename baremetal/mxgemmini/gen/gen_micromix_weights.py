#!/usr/bin/env python3
"""Pack user-supplied one-layer FP32 KxN NPZ weights; never download a checkpoint.
Required arrays: W_q,W_k,W_v,W_o,W_g,W_u,W_d. Q is unscaled; Q/K
columns are optionally reordered to all rotary first halves then second halves.
Biases, RoPE and RMS parameters belong to the caller's independent capture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from micromix_formats import quant

NAMES = ("q", "k", "v", "o", "g", "u", "d")


def allocation(k, mix):
    if k % 32:
        raise ValueError("K must be divisible by 32")
    nb = k // 32
    if mix == "fp8":
        return (0, 0, nb)
    counts = (nb // 2, nb // 4, nb // 4) if mix == "mix50" else (5 * nb // 8, nb // 4, nb // 8)
    if sum(counts) != nb:
        raise ValueError("Fixed allocation requires an integral K32 partition")
    return counts


def export(raw, prefix, mix, head_dim, permute=True, column_tile=128):
    prefix = Path(prefix)
    paths = [prefix.with_suffix("." + e) for e in ("h", "bin", "json")]
    if any(p.exists() or p.is_symlink() for p in paths):
        raise FileExistsError("Refusing existing output prefix")
    if not prefix.name.replace("_", "").isalnum():
        raise ValueError("Output basename must be a C identifier")
    if head_dim < 2 or head_dim % 2:
        raise ValueError("Even head dimension required")
    if column_tile < 2 or column_tile % 2:
        raise ValueError("Even positive column tile required")
    with np.load(raw, allow_pickle=False) as arrays:
        blob = bytearray()
        entries = []
        records = []

        def add(x):
            blob.extend(bytes((-len(blob)) % 64))
            off = len(blob)
            blob.extend(np.ascontiguousarray(x).tobytes())
            return off

        for name in NAMES:
            x = arrays["W_" + name]
            if x.dtype != np.float32 or x.ndim != 2 or not np.isfinite(x).all():
                raise ValueError("Finite FP32 KxN required: " + name)
            k, n = x.shape
            if n % 32:
                raise ValueError("N must be divisible by32")
            if permute and name in ("q", "k"):
                if n % head_dim:
                    raise ValueError("Q/K width must contain whole heads")
                idx = np.arange(n).reshape(-1, head_dim)
                idx = np.concatenate(
                    [idx[:, : head_dim // 2].ravel(), idx[:, head_dim // 2 :].ravel()]
                )
                x = x[:, idx]
            counts = allocation(k, mix)
            pointers = []
            scales = []
            kb = 0
            for f, count in zip((4, 6, 8), counts):
                if not count:
                    pointers.append(None)
                    continue
                packed = np.empty((count * 32, n // 2 if f == 4 else n), np.uint8)
                scale_rows = np.empty((count, n), np.uint8)
                for col in range(0, n, column_tile):
                    end = min(n, col + column_tile)
                    code, scale = quant(
                        x[kb * 32 : (kb + count) * 32, col:end].reshape(count, 32, end - col), f
                    )
                    if f == 8 and np.any((code & 127) > 120):
                        raise ValueError("FP8 code outside supplied physical image finite domain")
                    scale_rows[:, col:end] = scale
                    code = code.reshape(count * 32, end - col)
                    if f == 4:
                        packed[:, col // 2 : end // 2] = code[:, 0::2] | (code[:, 1::2] << 4)
                    else:
                        packed[:, col:end] = code
                scales.append(scale_rows)
                pointers.append(add(packed))
                kb += count
            so = add(np.concatenate(scales))
            entries.append((k, n, counts, pointers, so))
            records.append(
                dict(
                    projection=name,
                    k=k,
                    n=n,
                    k32_groups=counts,
                    group_offsets=pointers,
                    scale_offset=so,
                    source_array_sha256=hashlib.sha256(arrays["W_" + name].tobytes()).hexdigest(),
                )
            )
    symbol = "_binary_" + prefix.name + "_bin_start"
    lines = [
        "/* Generated direct MicroMX weights. Link binary object with data alignment64. */",
        "#pragma once",
        '#include "mx_micromix.h"',
        f"extern const uint8_t {symbol}[];",
        "static const mxmm_weight mxmm_weights[7]={",
    ]
    for k, n, counts, pointers, so in entries:
        ps = ["NULL" if v is None else f"{symbol}+{v}" for v in pointers]
        lines.append(
            " {"
            + ",".join(map(str, (k, n, *counts)))
            + ","
            + ",".join(ps)
            + f",{symbol}+{so}"
            + "},"
        )
    lines += ["};", ""]
    manifest = dict(
        schema_version=1,
        mix=mix,
        source_npz_sha256=hashlib.sha256(Path(raw).read_bytes()).hexdigest(),
        blob_sha256=hashlib.sha256(blob).hexdigest(),
        bytes=len(blob),
        alignment=64,
        k_block=32,
        scale="floor(log2(amax)), log2_pmax=0",
        rounding="RNE, even magnitude code on ties",
        q_scaled=False,
        qk_half_permutation=permute,
        head_dim=head_dim,
        column_tile=column_tile,
        physical_fp8_domain="code magnitude<=120; fixed block normalization",
        projections=records,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for p, data in zip(
        paths,
        ("\n".join(lines).encode(), bytes(blob), (json.dumps(manifest, indent=2) + "\n").encode()),
    ):
        with p.open("xb") as f:
            f.write(data)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output-prefix", type=Path, required=True)
    p.add_argument("--mix", choices=("fp8", "mix50", "mix625"), required=True)
    p.add_argument("--head-dim", type=int, required=True)
    p.add_argument("--no-qk-permutation", action="store_true")
    p.add_argument("--column-tile", type=int, default=128)
    a = p.parse_args()
    print(
        json.dumps(
            export(
                a.raw, a.output_prefix, a.mix, a.head_dim, not a.no_qk_permutation, a.column_tile
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
