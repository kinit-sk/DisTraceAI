#!/usr/bin/env python3
"""Self-quantize the DisTraceAI generators to AWQ 4-bit (W4A16).

Why this exists
---------------
AWQ-4bit is the portable precision: one artifact runs on RTX 3090 / A100 / H200
(all Ampere+). Community AWQ checkpoints exist for the larger models but not the
small ones, so we build them ourselves — a few minutes each on a single GPU.

Tooling
-------
We use GPTQModel (the maintained successor to the now-stale AutoAWQ) for the AWQ
algorithm. GPTQModel supports the Gemma 4 and Qwen 3.5 architectures and emits
AWQ-Marlin-compatible weights that vLLM loads with quantization="awq_marlin".
(AutoAWQ does not support these newer architectures.)

Reproducibility
---------------
The produced weights depend on (a) the GPTQModel + transformers versions (pinned
in setup_quantize.sh) and (b) the calibration set + sample count (pinned below).
Commit this script and the calibration spec; anyone can regenerate the weights.
We do NOT commit the weights as the source of truth — this recipe is.

Source repos and output paths come from core.models._CATALOGUE (single source of
truth), so the catalogue and the quantizer never drift.

NOTE: Context-1 (20B gpt-oss MoE) is intentionally NOT quantized here. Its
upstream low-bit checkpoint is produced by quantization-AWARE distillation; a
naive post-training AWQ of the bf16 weights would be lower quality. Use an
MXFP4/4-bit community/official checkpoint, or run bf16 on a large GPU.

Usage
-----
    python quantize.py --all                  # all six generators
    python quantize.py --models qwen3.5-2b gemma4-12b
    python quantize.py --all --out models/awq
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# transformers 5.12 + kernels 0.15 incompatibility: transformers' hub_kernels
# integration constructs LayerRepository(...) without a version/revision, which
# the installed `kernels` rejects ("Either a revision or a version must be
# specified") at import time — crashing before any work. We don't need hub
# kernels (GPTQModel does its own quant kernels), so disable the mapping. Must
# be set before transformers is imported (GPTQModel imports it transitively).
os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")

# Pinned calibration config — part of the reproducibility contract.
CALIB_DATASET = "allenai/c4"      # standard calibration corpus
CALIB_SAMPLES = 256               # n calibration sequences
CALIB_SEQLEN = 512                # tokens per calibration sequence

# AWQ quant config: 4-bit, group size 128 (the standard W4A16 AWQ setup).
AWQ_BITS = 4
AWQ_GROUP_SIZE = 128

# Generators eligible for self-AWQ (Context-1 excluded by design — see module doc).
SELF_QUANT_KEYS = [
    "qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
    "gemma4-e2b", "gemma4-e4b", "gemma4-12b",
]


def _source_repo(model_key: str) -> str:
    from core.models import _CATALOGUE
    if model_key not in _CATALOGUE:
        raise SystemExit(f"Unknown model key {model_key!r}. Eligible: {SELF_QUANT_KEYS}")
    return _CATALOGUE[model_key]["bf16"]


def _output_dir(model_key: str, out_root: Path) -> Path:
    from core.models import _CATALOGUE
    awq_path = _CATALOGUE[model_key]["awq4"]      # e.g. "models/awq/qwen3.5-2b-awq"
    return out_root / Path(awq_path).name


def _load_calibration(n: int):
    """Load n short text samples for calibration (deterministic slice)."""
    from datasets import load_dataset
    ds = load_dataset(CALIB_DATASET, "en", split="train", streaming=True)
    texts = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if t:
            texts.append(t)
        if len(texts) >= n:
            break
    return texts


def quantize_one(model_key: str, out_root: Path) -> None:
    # GPTQModel 5.8.0 selects AWQ via QuantizeConfig(quant_method=METHOD.AWQ),
    # NOT a dedicated AWQConfig class (that arrived in a later 5.x release) and
    # NOT QuantizeConfig(format="awq") (format is the checkpoint serialization,
    # not the method -> mismatch error). FORMAT.GEMM is the standard AWQ kernel
    # format that vLLM's awq_marlin loader expects. AWQ uses zero-point
    # quantization, i.e. sym=False (the source sets zero_point = not sym).
    # AWQ is calibration-based, so a list of text samples goes to .quantize().
    from gptqmodel import GPTQModel, QuantizeConfig
    from gptqmodel.quantization.config import METHOD, FORMAT

    src = _source_repo(model_key)
    dst = _output_dir(model_key, out_root)
    if any(dst.glob("*.safetensors")):
        print(f"[quantize] {model_key}: already built at {dst} — skipping.")
        return
    dst.mkdir(parents=True, exist_ok=True)

    # Standard AWQ format is GEMM; if this GPTQModel build names it differently,
    # fall back to method-only (GPTQModel then picks its default AWQ format).
    cfg_kwargs = dict(bits=AWQ_BITS, group_size=AWQ_GROUP_SIZE,
                      quant_method=METHOD.AWQ, sym=False)
    awq_format = getattr(FORMAT, "GEMM", None)
    if awq_format is not None:
        cfg_kwargs["format"] = awq_format
    qcfg = QuantizeConfig(**cfg_kwargs)
    print(f"[quantize] {model_key}: loading {src} …")
    model = GPTQModel.load(src, qcfg)

    print(f"[quantize] {model_key}: calibrating "
          f"({CALIB_DATASET}, n={CALIB_SAMPLES}, seqlen={CALIB_SEQLEN}) "
          f"and quantizing to AWQ-4bit …")
    calib = _load_calibration(CALIB_SAMPLES)
    model.quantize(calib, batch_size=1)
    model.save(str(dst))
    print(f"[quantize] {model_key}: saved AWQ weights -> {dst}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Self-quantize generators to AWQ-4bit (GPTQModel).")
    ap.add_argument("--all", action="store_true", help="quantize all six generators")
    ap.add_argument("--models", nargs="+", default=[],
                    help="specific model keys (e.g. qwen3.5-2b gemma4-12b)")
    ap.add_argument("--out", default="models/awq", help="output root dir")
    ap.add_argument("--probe", action="store_true",
                    help="print GPTQModel METHOD/FORMAT enums and verify the AWQ "
                         "QuantizeConfig builds, without loading or quantizing any model")
    args = ap.parse_args()

    if args.probe:
        from gptqmodel import QuantizeConfig
        from gptqmodel.quantization.config import METHOD, FORMAT
        print("METHODS:", [m for m in METHOD])
        print("FORMATS:", [f for f in FORMAT])
        cfg_kwargs = dict(bits=AWQ_BITS, group_size=AWQ_GROUP_SIZE,
                          quant_method=METHOD.AWQ, sym=False)
        awq_format = getattr(FORMAT, "GEMM", None)
        if awq_format is not None:
            cfg_kwargs["format"] = awq_format
        cfg = QuantizeConfig(**cfg_kwargs)
        print("AWQ QuantizeConfig built OK:", cfg)
        return 0

    if args.all:
        keys = list(SELF_QUANT_KEYS)
    elif args.models:
        keys = args.models
    else:
        ap.error("pass --all or --models <keys…> (or --probe)")

    out_root = Path(args.out)
    for k in keys:
        if k not in SELF_QUANT_KEYS:
            print(f"[quantize] WARNING: {k!r} not in self-quant set "
                  f"({SELF_QUANT_KEYS}); skipping.", file=sys.stderr)
            continue
        quantize_one(k, out_root)
    print("[quantize] all requested models done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
