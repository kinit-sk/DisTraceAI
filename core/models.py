from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

# --- vLLM 0.22 DeepGEMM warmup workaround ----------------------------------
# On Hopper GPUs (H200, sm_90) vLLM 0.22's kernel_warmup calls deep_gemm_warmup
# unconditionally; if the optional `deep_gemm` package isn't installed it raises
# "DeepGEMM backend is not available or outdated" during engine init — even for
# bf16 / AWQ models that have no FP8 layers (vLLM issue #41849). Skipping the
# warmup avoids the crash and costs nothing for our non-FP8 models. Set before
# vLLM is imported anywhere. Respect an existing user override.
os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

# transformers 5.12 + kernels 0.15 incompatibility: transformers' hub_kernels
# integration builds LayerRepository(...) with no version/revision, which the
# installed `kernels` rejects at import ("Either a revision or a version must be
# specified"). We don't use hub kernels (vLLM has its own), so disable the
# mapping before transformers is imported anywhere (the detectors import it).
os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")


# ---- precision axis -------------------------------------------------------
# The pipeline's model precision is a configurable axis replacing the old GGUF
# quantization levels. Values:
#   awq4 - AWQ 4-bit (W4A16). Runs on Ampere (RTX 3090, A100) AND Hopper (H200).
#          The portable default: one artifact runs on every target GPU.
#   bf16 - 16-bit. Runs everywhere with enough VRAM; the quality reference point.
# Both run on the full hardware range (3090 / A100 / H200) with no device
# detection, which is why they are the only two precisions supported.
_PRECISIONS = ("awq4", "bf16")

# Map a precision token to the vLLM `quantization=` argument and torch dtype.
# awq4 -> quantization="awq_marlin" (the fast Marlin kernel, Ampere+Hopper),
#         weights loaded from a pre-quantized AWQ checkpoint on disk.
# bf16 -> no quantization, dtype bfloat16, loaded from the full-precision repo.
_PRECISION_VLLM = {
    "awq4": {"quantization": "awq_marlin", "dtype": "float16"},
    "bf16": {"quantization": None,         "dtype": "bfloat16"},
}


def normalize_precision(precision: str) -> str:
    """Validate/normalize a precision token (tolerates legacy GGUF names by
    mapping them onto the nearest precision: Q4* -> awq4, Q6/Q8 -> bf16)."""
    p = (precision or "").strip().lower()
    if p in _PRECISIONS:
        return p
    legacy = {"q4_k_m": "awq4", "q6_k": "bf16", "q8_0": "bf16"}
    if p in legacy:
        logger.warning("[models] legacy quant %r mapped to precision %r",
                       precision, legacy[p])
        return legacy[p]
    raise ValueError(f"Unknown precision {precision!r}. Valid: {_PRECISIONS}")


def _resolve_model_path(model: str) -> str:
    """Return a string vLLM/HF will load correctly.

    A LOCAL directory (e.g. our AWQ weights under ``models/awq/...``) must be
    handed to HF as a filesystem path. HF's ``validate_repo_id`` rejects a
    slashed string that is not a ``namespace/name`` repo id, so a relative local
    path raises ``HFValidationError``. We:
      - return the absolute path for an existing local dir;
      - if the string is clearly a local catalogue path (under ``models/``) but
        the dir is MISSING, raise a clear, actionable error (the AWQ weights have
        not been built yet) instead of letting HF raise a cryptic repo-id error;
      - otherwise return the string unchanged (a genuine HF repo id).
    """
    p = Path(model)
    if p.exists():
        return str(p.resolve())
    # Heuristic: our local AWQ catalogue paths start with "models/". A real HF
    # repo id is "namespace/name" and never begins with our models dir.
    looks_local = model.startswith("models/") or model.startswith("./") or model.startswith("/")
    if looks_local:
        raise FileNotFoundError(
            f"Model weights not found at '{model}'. For an awq4 precision this "
            f"means the AWQ weights have not been built yet. Build them once with:\n"
            f"    conda activate distrace && ./setup_quantize.sh\n"
            f"or run this step with --*-precision bf16 to load the full-precision "
            f"model from HuggingFace instead."
        )
    return model


# ---- generator catalogue (vLLM) ------------------------------------------
# model key -> {precision: HF repo or local path}. AWQ checkpoints live under
# models/ (built once by quantize.py); bf16 points at the upstream repo and is
# fetched into the same models/ HF cache. Context-1 is the agentic retriever /
# veracity verifier (a 20B gpt-oss MoE) — its low-bit checkpoint is QAT-distilled
# upstream, so we do NOT self-AWQ it; use a 4-bit community/official checkpoint
# or run bf16 on a big GPU.
_MODELS_DIR = Path("models")

_CATALOGUE: dict[str, dict[str, str]] = {
    "qwen3.5-2b": {
        "awq4": "models/awq/qwen3.5-2b-awq",
        "bf16": "Qwen/Qwen3.5-2B",
    },
    "qwen3.5-4b": {
        "awq4": "models/awq/qwen3.5-4b-awq",
        "bf16": "Qwen/Qwen3.5-4B",
    },
    "qwen3.5-9b": {
        "awq4": "models/awq/qwen3.5-9b-awq",
        "bf16": "Qwen/Qwen3.5-9B",
    },
    "gemma4-e2b": {
        "awq4": "models/awq/gemma4-e2b-awq",
        "bf16": "google/gemma-4-E2B-it",
    },
    "gemma4-e4b": {
        "awq4": "models/awq/gemma4-e4b-awq",
        "bf16": "google/gemma-4-E4B-it",
    },
    "gemma4-12b": {
        # Google ships an official QAT W4A16 checkpoint for the 12B
        # (google/gemma-4-12B-it-qat-w4a16-ct) — quantization-aware-trained, so
        # higher quality than self-AWQ. Prefer it over building our own for 12B.
        "awq4": "models/awq/gemma4-12b-awq",
        "bf16": "google/gemma-4-12B-it",
    },
    # Context-1: 20B gpt-oss MoE agentic search/verify model. Served via vLLM
    # (as Chroma do). The official MXFP4 checkpoint is "coming soon" but NOT yet
    # released, and the only community 4-bit (foadmk/context-1-MLX-MXFP4) is MLX
    # (Apple Silicon), which vLLM cannot load. So until Chroma ship MXFP4, run
    # BF16 on a large GPU (~40 GB; trivial on the H200). Do not self-quantize —
    # Context-1's low-bit form is QAT-distilled upstream and uses a nonstandard
    # interleaved expert weight layout. Point awq4 here at the official MXFP4
    # checkpoint once it lands.
    "context-1": {
        "awq4": "chromadb/context-1",   # TODO: official MXFP4 when released
        "bf16": "chromadb/context-1",
    },
}

# Per-model context window (max_model_len). Context-1 needs headroom for the
# multi-turn agentic search history; the rest default to 16k.
_DEFAULT_CTX = {"context-1": 32768}


def resolve_generator(model_key: str, precision: str) -> tuple[str, dict]:
    """Map (model key | 'vendor/key' alias, precision) -> (model_path_or_repo,
    vllm_kwargs) where vllm_kwargs carries quantization+dtype for that precision."""
    precision = normalize_precision(precision)
    key = model_key if model_key in _CATALOGUE else model_key.split("/")[-1]
    if key not in _CATALOGUE:
        raise ValueError(f"Unknown generator {model_key!r}. Available: {sorted(_CATALOGUE)}")
    by_prec = _CATALOGUE[key]
    if precision not in by_prec:
        raise ValueError(f"{key!r} has no {precision!r} variant (have {sorted(by_prec)})")
    return by_prec[precision], dict(_PRECISION_VLLM[precision])


# ---- embedder (vLLM pooling runner) --------------------------------------
def make_embedder(model_name: str, *, max_seq_length: int = 512):
    """Load the embedder through vLLM in pooling mode.

    Returns a thin wrapper exposing ``.encode(texts, ...)`` so the rest of the
    pipeline (corpus.py, gen_* steps via encode_with_backoff) is unchanged.
    vLLM manages its own GPU memory; we cap max_model_len since fact-check claims
    are short and a long window wastes activation memory.

    vLLM 0.22 selects the embedding path via ``runner="pooling"`` (the older
    ``task="embed"`` argument was removed). Override the sequence cap with
    DISTRACE_EMBED_MAXLEN; lower the memory fraction with DISTRACE_EMBED_GPU_UTIL
    (default 0.30 so the embedder phase can coexist with other allocations).
    """
    from vllm import LLM
    try:
        max_seq_length = int(os.environ.get("DISTRACE_EMBED_MAXLEN", max_seq_length))
    except ValueError:
        pass
    try:
        gpu_util = float(os.environ.get("DISTRACE_EMBED_GPU_UTIL", "0.30"))
    except ValueError:
        gpu_util = 0.30
    logger.info("[models] embedder %s via vLLM (runner=pooling, max_len=%s)",
                model_name, max_seq_length)
    engine = LLM(model=_resolve_model_path(model_name), runner="pooling",
                 max_model_len=max_seq_length,
                 gpu_memory_utilization=gpu_util,
                 enforce_eager=True, trust_remote_code=True)
    return _VLLMEmbedder(engine, model_name)


class _VLLMEmbedder:
    """SentenceTransformers-compatible ``.encode`` shim over a vLLM embed engine."""

    def __init__(self, engine, model_name: str) -> None:
        self.engine = engine
        self.model_name = model_name

    def encode(self, texts, batch_size: int = 32, convert_to_numpy: bool = True,
               show_progress_bar: bool = False, **_ignore):
        import numpy as np
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        outputs = self.engine.embed(texts)
        vecs = [o.outputs.embedding for o in outputs]
        return np.asarray(vecs, dtype=np.float32)

    def close(self) -> None:
        _teardown_vllm(getattr(self, "engine", None))
        self.engine = None


def encode_with_backoff(embedder, texts: Sequence[str],
                        initial_batch_size: int = 32, min_batch_size: int = 4,
                        show_progress: bool = False):
    """Encode texts to embeddings. Kept for signature compatibility with the
    ~6 call sites in corpus.py / gen_* . vLLM batches internally and manages its
    own memory, so the old CUDA-OOM batch-halving ladder is no longer needed;
    this now simply delegates to the embedder's encode()."""
    return embedder.encode(list(texts), batch_size=initial_batch_size,
                           convert_to_numpy=True, show_progress_bar=show_progress)


# ---- generator (vLLM) -----------------------------------------------------
def _teardown_vllm(engine) -> None:
    """Deterministically free a vLLM engine's GPU VRAM. vLLM does NOT release
    VRAM when the Python object goes out of scope, so phasing (load model A ->
    run -> free -> load model B on one GPU) requires this explicit teardown."""
    if engine is None:
        return
    import gc
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel, destroy_distributed_environment)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as exc:                       # pragma: no cover - runtime/version
        logger.debug("[gen] vLLM parallel-state teardown raised (ignored): %s", exc)
    try:
        del engine
    except Exception:
        pass
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class VLLMGenerator:
    """Callable vLLM generator implementing the `generate(system, user, **kw)`
    contract used across the pipeline.

    Backed by one resident vLLM engine. Calls are made serially (the offline
    `LLM` engine is driven one .chat() at a time); throughput comes from vLLM's
    paged-attention batching within each call. temperature/max_tokens are
    honoured; `/no_think` is appended unless thinking.
    """

    def __init__(self, model_key: str, precision: str, *,
                 context_size: int | None = None, temperature: float = 0.0,
                 gpu_memory_utilization: float | None = None) -> None:
        from vllm import LLM
        model_path, vkw = resolve_generator(model_key, precision)
        self.model_key, self.precision = model_key, normalize_precision(precision)
        self.temperature = temperature
        key = model_key if model_key in _CATALOGUE else model_key.split("/")[-1]
        context_size = context_size or _DEFAULT_CTX.get(key, 16384)
        self._context_size = context_size
        if gpu_memory_utilization is None:
            try:
                gpu_memory_utilization = float(
                    os.environ.get("DISTRACE_GEN_GPU_UTIL", "0.90"))
            except ValueError:
                gpu_memory_utilization = 0.90

        llm_kwargs = dict(model=_resolve_model_path(model_path), dtype=vkw["dtype"],
                          max_model_len=context_size,
                          gpu_memory_utilization=gpu_memory_utilization,
                          trust_remote_code=True)
        if vkw["quantization"] is not None:
            llm_kwargs["quantization"] = vkw["quantization"]
        logger.info("[gen] loading %s [%s] via vLLM (ctx=%d, util=%.2f, quant=%s)",
                    model_path, self.precision, context_size, gpu_memory_utilization,
                    vkw["quantization"])
        self.llm = LLM(**llm_kwargs)

    def __call__(self, system: str, user: str, *, temperature: float | None = None,
                 max_tokens: int = 256, thinking: bool = False) -> str:
        from vllm import SamplingParams
        if not thinking and "/no_think" not in system:
            system = system + " /no_think"
        sp = SamplingParams(
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens, top_p=0.8, top_k=20, min_p=0.0,
            presence_penalty=1.5, repetition_penalty=1.0)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        outputs = self.llm.chat(messages, sp, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def generate_json(self, system: str, user: str, schema: dict, *,
                      temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Schema-constrained JSON generation via vLLM structured outputs.

        Replaces the llama.cpp `create_chat_completion(response_format=...)` path
        NodeRAG used. vLLM renamed this API across versions — newer releases use
        ``StructuredOutputsParams`` + ``SamplingParams(structured_outputs=...)``,
        older ones ``GuidedDecodingParams`` + ``guided_decoding=...``. We try the
        new names first and fall back, so it works across the rename.
        Returns the raw JSON text (caller json.loads it).
        """
        from vllm import SamplingParams
        if "/no_think" not in system:
            system = system + " /no_think"
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        try:                                   # newer vLLM (structured_outputs)
            from vllm.sampling_params import StructuredOutputsParams
            so = StructuredOutputsParams(json=schema)
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                structured_outputs=so)
        except ImportError:                    # older vLLM (guided_decoding)
            from vllm.sampling_params import GuidedDecodingParams
            gd = GuidedDecodingParams(json=schema)
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                guided_decoding=gd)
        outputs = self.llm.chat(messages, sp, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def close(self) -> None:
        """Release the model and free GPU VRAM deterministically (call between
        pipeline phases so the next model can load on the same GPU)."""
        llm = getattr(self, "llm", None)
        self.llm = None
        _teardown_vllm(llm)

    def __enter__(self) -> "VLLMGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def make_generator(model_key: str, precision: str, **kw):
    """Build a vLLM generator.

    Set DISTRACE_FORCE_PRECISION=bf16 (or awq4) to override the configured
    precision for ALL generators in a run — handy for verifying the pipeline
    end-to-end with bf16 weights pulled from HF before building AWQ weights, or
    for forcing AWQ once they're built, without editing configs or CLI flags.
    """
    forced = os.environ.get("DISTRACE_FORCE_PRECISION")
    if forced:
        precision = forced
    return VLLMGenerator(model_key, precision, **kw)


def close_generator(generate) -> None:
    """Safely close any generator (VLLMGenerator / None). Frees GPU VRAM
    deterministically between pipeline steps (replaces bare `del llm`)."""
    if generate is None:
        return
    close = getattr(generate, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:                        # pragma: no cover - runtime
            logger.debug("[gen] close_generator raised (ignored): %s", exc)
