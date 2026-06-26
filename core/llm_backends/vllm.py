"""vLLM inference backend for DisTraceAI.

Implements the same public surface as the llama_cpp backend:
  make_embedder(model_name, ...)      -> embedder with .encode()
  encode_with_backoff(embedder, ...)  -> np.ndarray
  make_generator(model_key, ...)      -> VLLMGenerator
  close_generator(generate)
  generator_is_concurrent(generate)  -> bool
  parallel_map(fn, items, ...)

Models run in bf16, loaded directly from HuggingFace repos.

Environment variables honoured by this module (all read at call-time so TUI
changes in the same process take effect):

  vLLM workarounds (applied at import-time; see comments for details):
    VLLM_DEEP_GEMM_WARMUP          default "skip"  (Hopper FP8 crash workaround)
    VLLM_USE_DEEP_GEMM             default "0"
    DISTRACE_USE_FLASHINFER_SAMPLER default "0" — set to "1" to re-enable FlashInfer
    DISABLE_KERNEL_MAPPING         default "1"  (transformers 5.12 + kernels 0.15)

  Embedder:
    DISTRACE_EMBED_MAXLEN          Integer: cap embedder max_model_len (default 512)
    DISTRACE_EMBED_GPU_UTIL        Float:   vLLM gpu_memory_utilization (default 0.30)

  Generator:
    DISTRACE_GEN_GPU_UTIL          Float:   vLLM gpu_memory_utilization (default 0.40)
    DISTRACE_SHOW_TQDM             Set "1" to keep vLLM/library tqdm bars

  HuggingFace:
    HF_HUB_DOWNLOAD_TIMEOUT        default "60"
    HF_TOKEN                       (optional) lifts HF rate limits
    HF_HUB_ENABLE_HF_TRANSFER      auto-set if hf_transfer is installed
"""
from __future__ import annotations

import logging
import os
from typing import Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Process-wide env workarounds — applied at import time
# ---------------------------------------------------------------------------

os.environ.setdefault("VLLM_DEEP_GEMM_WARMUP", "skip")
os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

if os.environ.get("DISTRACE_USE_FLASHINFER_SAMPLER", "0") == "1":
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "1")
else:
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

os.environ.setdefault("DISABLE_KERNEL_MAPPING", "1")

# Silence vLLM / NodeRAG tqdm bars (Rich handles all progress output)
if os.environ.get("DISTRACE_SHOW_TQDM", "0") != "1":
    try:
        import tqdm as _tqdm_mod
        from tqdm import tqdm as _tqdm_cls

        class _SilentTqdm(_tqdm_cls):  # type: ignore[misc]
            def __init__(self, *a, **kw):
                kw.setdefault("disable", True)
                super().__init__(*a, **kw)

        _tqdm_mod.tqdm = _SilentTqdm
        try:
            import tqdm.auto as _tqdm_auto
            _tqdm_auto.tqdm = _SilentTqdm
        except Exception:
            pass
    except Exception:
        pass

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
try:
    import importlib.util as _ilu
    if _ilu.find_spec("hf_transfer") is not None:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
except Exception:
    pass


# ---------------------------------------------------------------------------
# Model catalogue  (bf16 HuggingFace repos)
# ---------------------------------------------------------------------------

_CATALOGUE: dict[str, str] = {
    "qwen3.5-2b": "Qwen/Qwen3.5-2B",
    "qwen3.5-4b": "Qwen/Qwen3.5-4B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    "gemma4-e2b": "google/gemma-4-E2B-it",
    "gemma4-e4b": "google/gemma-4-E4B-it",
    "gemma4-12b": "google/gemma-4-12B-it",
    "context-1":  "chromadb/context-1",
}

_DEFAULT_CTX: dict[str, int] = {"context-1": 32768}


def resolve_generator(model_key: str) -> str:
    """Map a model key (or 'vendor/key' alias) to its HuggingFace repo id."""
    key = model_key if model_key in _CATALOGUE else model_key.split("/")[-1]
    if key not in _CATALOGUE:
        raise ValueError(
            f"Unknown generator {model_key!r}. Available: {sorted(_CATALOGUE)}")
    return _CATALOGUE[key]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _construct_llm_with_retry(LLM, llm_kwargs: dict, *, attempts: int = 3):
    """Build a vLLM LLM engine, retrying on transient HuggingFace Hub failures."""
    import time
    model = llm_kwargs.get("model", "<unknown>")
    last_exc = None
    for i in range(1, attempts + 1):
        try:
            return LLM(**llm_kwargs)
        except OSError as exc:
            msg = str(exc)
            transient = (
                "Can't load the configuration" in msg
                or "Server disconnected" in msg
                or "RemoteProtocolError" in msg
                or "Connection" in msg
                or "timed out" in msg.lower()
            )
            if not transient or i == attempts:
                if transient:
                    raise OSError(
                        f"Failed to download '{model}' from HuggingFace after "
                        f"{attempts} attempts. Set HF_TOKEN to lift rate limits.\n"
                        f"Original error: {msg}") from exc
                raise
            wait = 5 * i
            logger.warning("[vllm] transient Hub error loading %s (attempt %d/%d): "
                           "%s — retrying in %ds", model, i, attempts, msg, wait)
            time.sleep(wait)
            last_exc = exc
    if last_exc:
        raise last_exc


def _teardown_vllm(engine) -> None:
    """Deterministically free a vLLM engine's GPU VRAM."""
    if engine is None:
        return
    import gc
    try:
        from vllm.distributed.parallel_state import (
            destroy_model_parallel, destroy_distributed_environment)
        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as exc:
        logger.debug("[vllm] parallel-state teardown raised (ignored): %s", exc)
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


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

def make_embedder(model_name: str, *, max_seq_length: int = 512):
    """Load the embedder through vLLM in pooling mode.

    Returns a _VLLMEmbedder wrapping a vLLM LLM(runner="pooling") engine.
    Override the sequence cap with DISTRACE_EMBED_MAXLEN; override GPU util
    with DISTRACE_EMBED_GPU_UTIL (default 0.30).
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
    logger.info("[vllm] embedder %s via vLLM (runner=pooling, max_len=%s)",
                model_name, max_seq_length)
    engine = _construct_llm_with_retry(LLM, dict(
        model=model_name, runner="pooling",
        max_model_len=max_seq_length,
        gpu_memory_utilization=gpu_util,
        enforce_eager=True, trust_remote_code=True))
    return _VLLMEmbedder(engine, model_name, max_seq_length=max_seq_length)


class _VLLMEmbedder:
    """SentenceTransformers-compatible .encode shim over a vLLM embed engine."""

    def __init__(self, engine, model_name: str, max_seq_length: int = 512) -> None:
        self.engine = engine
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self._tok = None

    def _truncate_to_window(self, texts):
        hard = max(8, int(getattr(self, "max_seq_length", 512)) - 8)
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
        except Exception as exc:
            logger.warning("[vllm] embedder token-truncation unavailable (%s); "
                           "falling back to char clip", exc)
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
    """Encode texts to embeddings (vLLM manages batching internally)."""
    return embedder.encode(list(texts), batch_size=initial_batch_size,
                           convert_to_numpy=True, show_progress_bar=show_progress)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class VLLMGenerator:
    """Callable vLLM generator implementing the generate(system, user, **kw)
    contract used across the pipeline."""

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
            try:
                gpu_memory_utilization = float(
                    os.environ.get("DISTRACE_GEN_GPU_UTIL", "0.40"))
            except ValueError:
                gpu_memory_utilization = 0.40

        llm_kwargs = dict(model=model_repo, dtype="bfloat16",
                          max_model_len=context_size,
                          gpu_memory_utilization=gpu_memory_utilization,
                          trust_remote_code=True)
        logger.info("[vllm] loading %s [bf16] (ctx=%d, util=%.2f)",
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
                    {"role": "user",   "content": user}]
        outputs = self.llm.chat(messages, sp, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def generate_json(self, system: str, user: str, schema: dict, *,
                      temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Schema-constrained JSON generation via vLLM structured outputs."""
        from vllm import SamplingParams
        if "/no_think" not in system:
            system = system + " /no_think"
        messages = [{"role": "system", "content": system},
                    {"role": "user",   "content": user}]
        try:
            from vllm.sampling_params import StructuredOutputsParams
            so = StructuredOutputsParams(json=schema)
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                structured_outputs=so)
        except ImportError:
            from vllm.sampling_params import GuidedDecodingParams
            gd = GuidedDecodingParams(json=schema)
            sp = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                guided_decoding=gd)
        outputs = self.llm.chat(messages, sp, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        self.llm = None
        _teardown_vllm(llm)

    def __enter__(self) -> "VLLMGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# Duck-type alias so code that imports LlamaGenerator still works
LlamaGenerator = VLLMGenerator


# ---------------------------------------------------------------------------
# Public factory / helpers  (mirror llama_cpp backend interface)
# ---------------------------------------------------------------------------

def make_generator(model_key: str, quant: str = "", **kw):
    """Build a bf16 vLLM generator. quant is accepted but ignored (no GGUF)."""
    # Strip llama-cpp-only kwargs that vLLM doesn't understand
    kw.pop("server_url", None)
    kw.pop("workers", None)
    kw.pop("main_gpu", None)
    kw.pop("gpu_only", None)
    return VLLMGenerator(model_key, **kw)


def generator_is_concurrent(generate) -> bool:
    return bool(getattr(generate, "concurrent", False))


def parallel_map(fn, items, max_workers: int = 1, description: str | None = None):
    from rich.progress import track
    items = list(items)
    if max_workers <= 1 or len(items) <= 1:
        return [fn(x) for x in (track(items, description) if description else items)]
    import concurrent.futures as cf
    results: list = [None] * len(items)
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(fn, x): i for i, x in enumerate(items)}
        completed = cf.as_completed(futures)
        if description:
            completed = track(completed, description, total=len(futures))
        for fut in completed:
            results[futures[fut]] = fut.result()
    return results


def close_generator(generate) -> None:
    """Safely close any generator, freeing GPU VRAM."""
    if generate is None:
        return
    close = getattr(generate, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.debug("[gen] close_generator raised (ignored): %s", exc)
