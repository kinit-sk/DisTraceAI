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

# --- FlashInfer sampler JIT-build workaround -------------------------------
# vLLM's sampler prefers FlashInfer's top-k/top-p kernel, which FlashInfer
# JIT-COMPILES on first use via nvcc/ninja. On a freshly-built env the nvcc
# toolchain (conda cuda-toolkit + conda gcc, targeting sm_90a on the H200) can
# fail that compile ("ninja: build stopped: subcommand failed" →
# CalledProcessError in flashinfer/jit), which aborts engine startup at the
# dummy sampler warmup. We don't need the FlashInfer sampler — it's only a
# speed optimization, irrelevant for our small eval workloads — so force vLLM's
# PyTorch-native top-k/top-p sampler instead. No nvcc, no JIT, no build step.
# Override with DISTRACE_USE_FLASHINFER_SAMPLER=1 if you have a working toolchain
# and want the kernel back.
if os.environ.get("DISTRACE_USE_FLASHINFER_SAMPLER", "0") == "1":
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
else:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

# transformers 5.12 + kernels 0.15 incompatibility: transformers' hub_kernels
# integration builds LayerRepository(...) with no version/revision, which the
# installed `kernels` rejects at import ("Either a revision or a version must be
# specified"). We don't use hub kernels (vLLM has its own), so disable the
# mapping before transformers is imported anywhere (the detectors import it).
os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")

# --- silence vLLM / NodeRAG "Processed prompts" tqdm bars -------------------
# vLLM (and NodeRAG's internal pipeline) print a per-call tqdm progress bar
# ("Processed prompts: ...") that floods the console on top of our own Rich
# progress bars. Per-call use_tqdm=False covers OUR call sites but not NodeRAG's
# internal calls. Since this project uses Rich for ALL of its own progress
# output and never uses tqdm directly, we disable tqdm globally by making
# `disable=True` its default — killing only the library-internal bars while
# leaving our Rich bars untouched. Set DISTRACE_SHOW_TQDM=1 to keep them.
if os.environ.get("DISTRACE_SHOW_TQDM", "0") != "1":
    try:
        import functools as _functools
        import tqdm as _tqdm_mod
        from tqdm import tqdm as _tqdm_cls

        class _SilentTqdm(_tqdm_cls):  # type: ignore[misc]
            def __init__(self, *a, **kw):
                kw.setdefault("disable", True)
                super().__init__(*a, **kw)

        _tqdm_mod.tqdm = _SilentTqdm
        # tqdm.auto re-exports its own reference; patch it too if importable.
        try:
            import tqdm.auto as _tqdm_auto
            _tqdm_auto.tqdm = _SilentTqdm
        except Exception:
            pass
    except Exception:
        pass

# --- HuggingFace Hub download robustness -----------------------------------
# Unauthenticated Hub requests are rate-limited and frequently drop mid-download
# ("Server disconnected without sending a response"), which surfaces as an
# OSError "Can't load the configuration of <repo>" when vLLM resolves a model
# that isn't cached locally yet. Give the Hub HTTP client a longer timeout, and
# enable hf_transfer (faster/more robust large-file downloads) ONLY if it's
# installed — setting the flag without the package would itself error. Set
# HF_TOKEN in your shell to lift the rate limits entirely (recommended).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
try:
    import importlib.util as _ilu
    if _ilu.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except Exception:
    pass


# ---- inference precision --------------------------------------------------
# The pipeline runs all generators in bf16 (16-bit). This runs on every target
# GPU (RTX 3090 / A100 / H200) with no quantization step, no device detection,
# and no precision configuration — the simplest path. Models are loaded straight
# from their full-precision HuggingFace repos.


# ---- generator catalogue (vLLM, bf16) ------------------------------------
# model key -> HuggingFace repo id. All generators run bf16, loaded straight
# from their upstream repos. Context-1 is the agentic retriever / veracity
# verifier (a 20B gpt-oss MoE) — run bf16 on a large GPU (~40 GB; fine on H200).
_CATALOGUE: dict[str, str] = {
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
    "gemma4-12b": "google/gemma-4-12B-it",
    "context-1":  "chromadb/context-1",
}

# Per-model context window (max_model_len). Context-1 needs headroom for the
# multi-turn agentic search history; the rest default to 16k.
_DEFAULT_CTX = {"context-1": 32768}


def resolve_generator(model_key: str) -> str:
    """Map a model key (or 'vendor/key' alias) to its HuggingFace repo id."""
    key = model_key if model_key in _CATALOGUE else model_key.split("/")[-1]
    if key not in _CATALOGUE:
        raise ValueError(f"Unknown generator {model_key!r}. Available: {sorted(_CATALOGUE)}")
    return _CATALOGUE[key]


def _construct_llm_with_retry(LLM, llm_kwargs: dict, *, attempts: int = 3):
    """Build a vLLM ``LLM`` engine, retrying on transient HuggingFace Hub
    download failures.

    When a model isn't cached locally, vLLM resolves it from the Hub during
    construction. Unauthenticated/rate-limited requests can drop mid-download and
    surface as an OSError ("Can't load the configuration of <repo>") caused by an
    httpx RemoteProtocolError. These are transient, so we retry with backoff. Set
    HF_TOKEN to avoid the rate limits in the first place. Non-transient errors
    (bad repo id, OOM, unsupported architecture) are re-raised immediately.
    """
    import time
    model = llm_kwargs.get("model", "<unknown>")
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            return LLM(**llm_kwargs)
        except OSError as exc:
            msg = str(exc)
            transient = ("Can't load the configuration" in msg
                         or "Server disconnected" in msg
                         or "RemoteProtocolError" in msg
                         or "Connection" in msg
                         or "timed out" in msg.lower())
            if not transient or i == attempts:
                if transient:
                    raise OSError(
                        f"Failed to download '{model}' from HuggingFace after "
                        f"{attempts} attempts (transient network error). Set HF_TOKEN "
                        f"to lift rate limits, check connectivity, or pre-download the "
                        f"model, then retry.\nOriginal error: {msg}") from exc
                raise
            wait = 5 * i
            logger.warning("[models] transient Hub error loading %s (attempt %d/%d): "
                           "%s — retrying in %ds", model, i, attempts, msg, wait)
            time.sleep(wait)
            last_exc = exc
    if last_exc:
        raise last_exc


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
    engine = _construct_llm_with_retry(LLM, dict(
        model=model_name, runner="pooling",
        max_model_len=max_seq_length,
        gpu_memory_utilization=gpu_util,
        enforce_eager=True, trust_remote_code=True))
    return _VLLMEmbedder(engine, model_name, max_seq_length=max_seq_length)


class _VLLMEmbedder:
    """SentenceTransformers-compatible ``.encode`` shim over a vLLM embed engine."""

    def __init__(self, engine, model_name: str, max_seq_length: int = 512) -> None:
        self.engine = engine
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self._tok = None

    def _truncate_to_window(self, texts):
        """Truncate inputs to the embedder window so vLLM can't reject long texts.

        The embedder runs with a fixed ``max_model_len`` (512 by default, tuned for
        short fact-check claims). NodeRAG, however, embeds long units — text chunks
        and attributes (the attribute prompt allows up to ~2000 words) — and vLLM
        raises ``VLLMValidationError`` for any input over the window rather than
        truncating. So we truncate here, token-accurately and multilingually.

        Short texts pass through untouched (no decode round-trip); only over-length
        texts are clipped. For larger windows (better NodeRAG retrieval) set
        ``DISTRACE_EMBED_MAXLEN`` before the run.
        """
        hard = max(8, int(getattr(self, "max_seq_length", 512)) - 8)  # margin for pooler specials
        # Fast path: anything under hard//2 CHARACTERS cannot exceed `hard` tokens
        # for our tokenizers (even CJK), so skip tokenization for the common short
        # case (claims / sub-narratives) and keep the main pipeline fast.
        if all(len(t) <= hard // 2 for t in texts):
            return texts
        try:
            if self._tok is None:
                self._tok = self.engine.get_tokenizer()
            ids_batch = self._tok(texts, add_special_tokens=False)["input_ids"]
            out = []
            for t, ids in zip(texts, ids_batch):
                out.append(self._tok.decode(ids[:hard]) if len(ids) > hard else t)
            return out
        except Exception as exc:  # tokenizer unavailable → conservative char clip
            logger.warning("[models] embedder token-truncation unavailable (%s); "
                           "falling back to a character clip", exc)
            climit = hard * 4
            return [t if len(t) <= climit else t[:climit] for t in texts]

    def encode(self, texts, batch_size: int = 32, convert_to_numpy: bool = True,
               show_progress_bar: bool = False, **_ignore):
        import numpy as np
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        texts = self._truncate_to_window(texts)
        outputs = self.engine.embed(texts, use_tqdm=False)
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

    def __init__(self, model_key: str, *,
                 context_size: int | None = None, temperature: float = 0.0,
                 gpu_memory_utilization: float | None = None) -> None:
        from vllm import LLM
        model_repo = resolve_generator(model_key)
        self.model_key = model_key
        self.temperature = temperature
        key = model_key if model_key in _CATALOGUE else model_key.split("/")[-1]
        context_size = context_size or _DEFAULT_CTX.get(key, 16384)
        self._context_size = context_size
        if gpu_memory_utilization is None:
            # vLLM's gpu_memory_utilization is a fraction of TOTAL VRAM and must
            # leave room for anything already resident. In several steps (sub-
            # narratives, narratives, veracity) an embedder is loaded BEFORE the
            # generator and stays resident, taking ~0.30 (DISTRACE_EMBED_GPU_UTIL).
            # So the generator default is 0.60, keeping embedder+generator at ~0.90
            # of the card with headroom. On a single-model step this is a little
            # conservative; raise it with DISTRACE_GEN_GPU_UTIL=0.9 if no embedder
            # shares the GPU, or lower it on tight-VRAM cards.
            try:
                gpu_memory_utilization = float(
                    os.environ.get("DISTRACE_GEN_GPU_UTIL", "0.60"))
            except ValueError:
                gpu_memory_utilization = 0.60

        llm_kwargs = dict(model=model_repo, dtype="bfloat16",
                          max_model_len=context_size,
                          gpu_memory_utilization=gpu_memory_utilization,
                          trust_remote_code=True)
        logger.info("[gen] loading %s [bf16] via vLLM (ctx=%d, util=%.2f)",
                    model_repo, context_size, gpu_memory_utilization)
        self.llm = _construct_llm_with_retry(LLM, llm_kwargs)

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


def make_generator(model_key: str, **kw):
    """Build a bf16 vLLM generator for the given model key."""
    return VLLMGenerator(model_key, **kw)


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
