"""Capture large CONTIGUOUS (activation, weight) tiles from a real TinyLlama forward pass.

``MXQuant/end_to_end_linear/systolic_simulation/data_evalrun_01`` was produced by MXQuant's
``log_pairs_from_eval.py`` with ``--N 32``, so every logged tile is a 32x32 window taken at a
*random* offset (``log_pairs_from_eval.py:183-195``). Eight tiles of one projection are eight
independent windows, not neighbours -- stitching them into a 64x64 grid, as
``gen_matmul_fp8_64x64_llama.py`` does, yields a matrix whose every *value* is real but whose
*structure* is a collage. It also does not scale: a 128x512 operand needs 64 tiles, more than any
one projection has.

This module re-runs the same capture at ``--N 512`` so each test shape can be sliced out of ONE
contiguous tile:

    A = A_square[:M, :K]        real activations, tokens x in-features
    B = W_square[:N, :K].T      real weights,     out x in-features -> [K][N]

``A_square`` and ``W_square`` of a pair share the in-feature offset ``i0``, so ``A @ B`` is a
genuine sub-block of that projection's real output.

It reuses MXQuant's own machinery -- ``eval_simquant.get_model`` / ``get_loaders`` and
``log_pairs_from_eval.PairLogger`` -- and differs from running ``log_pairs_from_eval.py`` directly
in one respect only: it stops after a single forward pass rather than evaluating the whole
wikitext2 test split, since ``max_pairs_per_target=1`` means every tile has been written by then.

Run it with the npu-exploration venv:

    .venv/bin/python3 -m app.capture_llama_tiles
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

MXQ_ROOT = Path(__file__).resolve().parent.parent / "MXQuant"

#: Tile edge. 512 covers every shape the matmul_tiled tests need (max M=128, K=512, N=256).
DEFAULT_N = 512
#: TinyLlama's k_proj/v_proj have out_features = 4*64 = 256 (num_key_value_heads=4), below N, so
#: PairLogger skips them (``log_pairs_from_eval.py:176``). These five have out_features >= 2048.
EXPECTED_TARGETS = ["attn.q_proj", "attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


#: ``mxquant/datautils.py:10`` asks for the bare dataset id ``wikitext``, which datasets>=4 rejects
#: ("Repository id must be 'namespace/name'"). Same corpus, canonical id.
WIKITEXT2 = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def _wikitext2_testenc(model_id: str):
    """The test split, tokenized exactly as ``mxquant.datautils.get_wikitext2`` does."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    testdata = load_dataset(*WIKITEXT2, split="test")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    return tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=DEFAULT_N, help="tile edge (tiles are N x N)")
    ap.add_argument("--out", type=Path,
                    default=MXQ_ROOT / "end_to_end_linear" / "systolic_simulation" / "data_evalrun_512")
    ap.add_argument("--model-id", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2"])
    ap.add_argument("--seqlen", type=int, default=None,
                    help="tokens per sample; defaults to --N (the tile needs N rows of tokens)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads; 0 = leave default")
    args = ap.parse_args()

    seqlen = args.seqlen or args.N

    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    if str(MXQ_ROOT) not in sys.path:
        sys.path.insert(0, str(MXQ_ROOT))
    esq = _load("eval_simquant", MXQ_ROOT / "eval_simquant.py")
    lpe = _load("log_pairs_from_eval", MXQ_ROOT / "end_to_end_linear" / "log_pairs_from_eval.py")

    print(f"[load] {args.model_id} (seqlen={seqlen}) via eval_simquant.get_model")
    model = esq.get_model(args.model_id, seqlen, seqlen, gpu=0)
    model.eval()

    print(f"[load] {args.dataset} test split")
    testenc = _wikitext2_testenc(args.model_id)

    logger = lpe.PairLogger(pairs_root=args.out.resolve(), N=args.N, max_pairs_per_target=1)
    logger.attach(model)

    # One forward pass over a single sample. max_pairs_per_target=1 means every hook has fired and
    # written its tile by the time this returns, so the remaining test batches would be pure cost.
    ids = testenc.input_ids[:, :model.seqlen]
    if ids.shape[1] < args.N:
        raise SystemExit(f"sample is {ids.shape[1]} tokens, need >= N={args.N}")
    print(f"[run] one forward pass, input_ids {tuple(ids.shape)}")
    with torch.no_grad():
        model(input_ids=ids.to(model.device), use_cache=False, return_dict=True)
    logger.clear()

    written = sorted(p.relative_to(args.out) for p in args.out.glob("layer*/*/A_square.npz"))
    print(f"\n[done] {len(written)} (A,W) pairs of {args.N}x{args.N} under {args.out}")
    got = sorted({p.parts[1] for p in written})
    print(f"  projections captured: {got}")
    missing = [t for t in EXPECTED_TARGETS if t not in got]
    if missing:
        print(f"  WARNING: expected projections not captured: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
