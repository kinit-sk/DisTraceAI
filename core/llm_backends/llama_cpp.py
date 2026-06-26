"""llama-cpp-python inference backend for DisTraceAI.

Implements the same public surface as the vLLM backend:
  make_embedder(model_name, ...)      -> embedder with .encode()
  encode_with_backoff(embedder, ...)  -> np.ndarray
  make_generator(model_key, quant, ...) -> LlamaGenerator | LlamaPool | ServerGenerator
  close_generator(generate)
  generator_is_concurrent(generate)  -> bool
  parallel_map(fn, items, ...)
  VLLMGenerator alias              -> LlamaGenerator (for duck-type compatibility)

All GGUF models are loaded from the models/ directory (auto-downloaded via
llama-cpp-python's Llama.from_pretrained if not cached).

Environment variables honoured by this module (all read at call-time, not
import-time, so TUI changes in the same process are effective):

  DISTRACE_EMBEDDER_DEVICE   Override compute device for SentenceTransformer
                             embedder (default: cuda if available, else cpu).
  DISTRACE_EMBED_FP32        Set to "1" to disable fp16 for the embedder.
  DISTRACE_EMBED_MAXLEN      Integer: cap embedder max_seq_length.
  DISTRACE_ENCODE_BATCH      Integer: initial encode batch size (default 32).
  DISTRACE_NODERAG_WORKERS   Integer: force pool worker count (0 = disable pool).
  HF_HOME                    Override HuggingFace cache root (default models/hub).
"""
from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder_device() -> str:
    """Return the device for the embedder. Override via DISTRACE_EMBEDDER_DEVICE."""
    override = os.environ.get("DISTRACE_EMBEDDER_DEVICE")
    if override:
        return override
    return get_device()


# ---------------------------------------------------------------------------
# Embedder  (SentenceTransformer, fp16 optional)
# ---------------------------------------------------------------------------

def make_embedder(model_name: str, *, max_seq_length: int = 512, fp16: bool = True):
    """Load the SentenceTransformer embedder.

    Two memory safeguards: load weights in fp16 (halves the fp32 footprint) and
    cap the sequence length.  Override per env:
      DISTRACE_EMBED_FP32=1   to disable fp16
      DISTRACE_EMBED_MAXLEN=N to change the cap
    """
    from sentence_transformers import SentenceTransformer
    device = get_embedder_device()
    model = SentenceTransformer(model_name, device=device)

    try:
        max_seq_length = int(os.environ.get("DISTRACE_EMBED_MAXLEN", max_seq_length))
    except ValueError:
        pass
    if max_seq_length and getattr(model, "max_seq_length", None):
        model.max_seq_length = min(model.max_seq_length, max_seq_length)

    use_fp16 = fp16 and os.environ.get("DISTRACE_EMBED_FP32") != "1"
    if use_fp16 and isinstance(device, str) and device.startswith("cuda"):
        try:
            model = model.half()
        except Exception as exc:
            logger.warning("[llama_cpp] could not cast embedder to fp16 (%s)", exc)
    logger.info("[llama_cpp] embedder %s on %s (max_seq=%s, fp16=%s)",
                model_name, device, getattr(model, "max_seq_length", "?"), use_fp16)
    return model


def encode_with_backoff(embedder, texts: Sequence[str],
                        initial_batch_size: int = 32, min_batch_size: int = 4,
                        show_progress: bool = False):
    """Encode with batch-size back-off on CUDA OOM.

    Halves the batch on OOM down to min_batch_size before, as a last resort,
    falling back to CPU.  Override initial batch size with DISTRACE_ENCODE_BATCH.
    """
    try:
        import torch
    except ImportError:
        torch = None
    try:
        bs = int(os.environ.get("DISTRACE_ENCODE_BATCH", initial_batch_size))
    except ValueError:
        bs = initial_batch_size
    bs = max(min_batch_size, bs)
    while bs >= min_batch_size:
        try:
            return embedder.encode(list(texts), batch_size=bs,
                                   convert_to_numpy=True, show_progress_bar=show_progress)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() and "cuda" not in str(exc).lower():
                raise
            try:
                torch.cuda.empty_cache(); torch.cuda.synchronize()
            except Exception:
                pass
            if bs == min_batch_size:
                break
            bs = max(min_batch_size, bs // 2)
            logger.warning("[llama_cpp] CUDA OOM during encode; retry at batch_size=%d", bs)
    logger.warning("[llama_cpp] GPU encode failed at min batch; falling back to CPU")
    try:
        embedder.to("cpu")
        gc.collect()
        try:
            import torch as _t; _t.cuda.empty_cache()
        except Exception:
            pass
    except Exception as move_exc:
        logger.debug("[llama_cpp] embedder.to(cpu) raised (ignored): %s", move_exc)
    return embedder.encode(list(texts), batch_size=min(16, initial_batch_size),
                           convert_to_numpy=True, show_progress_bar=show_progress)


# ---------------------------------------------------------------------------
# GGUF model catalogue
# ---------------------------------------------------------------------------

_CATALOGUE: dict[str, tuple[str, str]] = {
    "gemma4-e2b":     ("unsloth/gemma-4-E2B-it-GGUF",    "gemma-4-E2B-it-{quant}.gguf"),
    "gemma4-e4b":     ("unsloth/gemma-4-E4B-it-GGUF",    "gemma-4-E4B-it-{quant}.gguf"),
    "gemma4-12b":     ("unsloth/gemma-4-12b-it-GGUF",    "gemma-4-12b-it-{quant}.gguf"),
    "gemma4-26b-a4b": ("unsloth/gemma-4-26B-A4B-it-GGUF","gemma-4-26B-A4B-it-{quant}.gguf"),
    "gemma4-31b":     ("unsloth/gemma-4-31B-it-GGUF",    "gemma-4-31B-it-{quant}.gguf"),
    "qwen3.5-2b":     ("lmstudio-community/Qwen3.5-2B-GGUF", "Qwen3.5-2B-{quant}.gguf"),
    "qwen3.5-4b":     ("lmstudio-community/Qwen3.5-4B-GGUF", "Qwen3.5-4B-{quant}.gguf"),
    "qwen3.5-9b":     ("lmstudio-community/Qwen3.5-9B-GGUF", "Qwen3.5-9B-{quant}.gguf"),
    "qwen3.5-27b":    ("lmstudio-community/Qwen3.5-27B-GGUF","Qwen3.5-27B-{quant}.gguf"),
    "context-1":      ("nicolasembleton/context-1-GGUF",  "*{quant}*.gguf"),
}

_DEFAULT_CTX: dict[str, int] = {"context-1": 32768}


def resolve_generator(model_key: str, quant: str) -> tuple[str, str]:
    """Map a model key + quant to (repo_id, filename)."""
    key = model_key if model_key in _CATALOGUE else model_key.split("/")[-1]
    if key not in _CATALOGUE:
        raise ValueError(
            f"Unknown generator {model_key!r}. Available: {sorted(_CATALOGUE)}")
    repo, tmpl = _CATALOGUE[key]
    return repo, tmpl.format(quant=quant)


# ---------------------------------------------------------------------------
# In-process generator
# ---------------------------------------------------------------------------

def _warn_if_cpu_only(llm, n_gpu: int, main_gpu: int) -> None:
    try:
        import llama_cpp
        supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        gpu_built = bool(supports()) if callable(supports) else None
    except Exception:
        gpu_built = None

    if gpu_built is False:
        logger.warning(
            "[gen] llama-cpp-python was built WITHOUT CUDA — running on CPU. "
            'Reinstall: CMAKE_ARGS="-DGGML_CUDA=on" pip install --force-reinstall '
            "--no-cache-dir llama-cpp-python")
    elif n_gpu == 0:
        logger.warning(
            "[gen] generator loaded with n_gpu_layers=0 (CPU only) after VRAM "
            "back-off — expect slow generation.")


class LlamaGenerator:
    """Callable llama.cpp generator implementing the generate(system, user, **kw)
    contract used across the pipeline."""

    def __init__(self, model_key: str, quant: str, *,
                 context_size: int | None = None, temperature: float = 0.0,
                 main_gpu: int = 0, gpu_only: bool = False) -> None:
        from llama_cpp import Llama
        repo, filename = resolve_generator(model_key, quant)
        self.model_key, self.quant = model_key, quant
        self.temperature = temperature
        key = model_key if model_key in _CATALOGUE else model_key.split("/")[-1]
        context_size = context_size or _DEFAULT_CTX.get(key, 16384)
        n_threads = os.cpu_count() or 4
        logger.info("[gen] loading %s (%s, ctx=%d)", filename, repo, context_size)

        cache_dir = Path("models")
        cache_dir.mkdir(exist_ok=True)

        vram_markers = ("out of memory", "cuda", "failed to create llama_context",
                        "ggml_backend", "cublas", "device memory")
        file_markers = ("failed to load model", "failed to load model from file")
        seen_vram = False
        ladder = (-1,) if gpu_only else (-1, 32, 16, 0)
        last_exc: Exception | None = None

        for n_gpu in ladder:
            try:
                self.llm = Llama.from_pretrained(
                    repo_id=repo, filename=filename, n_ctx=context_size,
                    n_gpu_layers=n_gpu, n_threads=n_threads, main_gpu=main_gpu,
                    cache_dir=str(cache_dir), verbose=False)
                if n_gpu != -1:
                    logger.warning("[gen] loaded with n_gpu_layers=%d (VRAM pressure)", n_gpu)
                _warn_if_cpu_only(self.llm, n_gpu, main_gpu)
                return
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                self.llm = None
                gc.collect()
                is_vram = any(k in msg for k in vram_markers)
                is_file = any(k in msg for k in file_markers)
                if is_vram:
                    seen_vram = True
                if gpu_only:
                    raise RuntimeError(
                        f"[gen] gpu_only worker for {filename} did not fit on "
                        f"cuda:{main_gpu} ({exc})") from exc
                if is_vram or (is_file and seen_vram):
                    logger.warning("[gen] n_gpu_layers=%d failed (%s); retrying lower",
                                   n_gpu, exc)
                    continue
                raise RuntimeError(
                    f"[gen] failed to load {filename} from {repo} "
                    f"(n_gpu_layers={n_gpu}): {exc}") from exc
        raise RuntimeError(
            f"[gen] could not load {filename} from {repo} at any layer count "
            f"(last error: {last_exc})")

    def __call__(self, system: str, user: str, *, temperature: float | None = None,
                 max_tokens: int = 256, thinking: bool = False) -> str:
        if not thinking and "/no_think" not in system:
            system = system + " /no_think"
        resp = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens, top_p=0.8, top_k=20, min_p=0.0,
            presence_penalty=1.5, repeat_penalty=1.0)
        return resp["choices"][0]["message"]["content"].strip()

    def generate_json(self, system: str, user: str, schema: dict, *,
                      temperature: float = 0.0, max_tokens: int = 4096) -> str:
        """Schema-constrained JSON generation via llama.cpp response_format."""
        if not thinking and "/no_think" not in system:
            system = system + " /no_think"
        resp = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}],
            response_format={"type": "json_object", "schema": schema},
            temperature=temperature, max_tokens=max_tokens)
        return resp["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        llm = getattr(self, "llm", None)
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:
                    logger.debug("[gen] llm.close() raised (ignored): %s", exc)
            self.llm = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def __enter__(self) -> "LlamaGenerator":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# Duck-type alias so code that imports VLLMGenerator still works when this
# backend is active (e.g. type-hints in gen_* modules).
VLLMGenerator = LlamaGenerator


# ---------------------------------------------------------------------------
# Server-backed generator  (llama-server with continuous batching)
# ---------------------------------------------------------------------------

class ServerGenerator:
    """OpenAI-compatible client for a llama-server started with
    --parallel N --cont-batching."""

    concurrent = True

    def __init__(self, base_url: str, model: str = "local", *,
                 temperature: float = 0.0, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def __call__(self, system: str, user: str, *, temperature: float | None = None,
                 max_tokens: int = 256, thinking: bool = False) -> str:
        import requests
        if not thinking and "/no_think" not in system:
            system = system + " /no_think"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user",   "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(f"{self.base_url}/chat/completions",
                          json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Pool of in-process workers
# ---------------------------------------------------------------------------

class LlamaPool:
    """Pool of independent in-process llama.cpp models for parallel decoding."""

    concurrent = True

    def __init__(self, model_key: str, quant: str, n_workers: int, **kw) -> None:
        import queue
        self.n_workers = max(1, int(n_workers))
        self._free: "queue.Queue" = queue.Queue()
        self.instances: list = []
        for i in range(self.n_workers):
            inst_kw = dict(kw); inst_kw.setdefault("main_gpu", 0)
            logger.info("[gen] LlamaPool: loading worker %d/%d on cuda:0",
                        i + 1, self.n_workers)
            gen = LlamaGenerator(model_key, quant, **inst_kw)
            self.instances.append(gen); self._free.put(gen)
        logger.info("[gen] LlamaPool ready: %d workers on cuda:0", self.n_workers)

    @classmethod
    def from_placement(cls, model_key: str, quant: str, placement: list[int],
                       **kw) -> "LlamaPool":
        import queue
        self = cls.__new__(cls)
        self.n_workers = 0
        self._free = queue.Queue()
        self.instances = []
        for i, gpu in enumerate(placement):
            inst_kw = dict(kw)
            inst_kw["main_gpu"] = gpu
            inst_kw["gpu_only"] = True
            try:
                logger.info("[gen] LlamaPool: loading worker %d/%d on cuda:%d",
                            i + 1, len(placement), gpu)
                gen = LlamaGenerator(model_key, quant, **inst_kw)
            except Exception as exc:
                logger.warning("[gen] LlamaPool: worker %d did not fit (%s); "
                               "stopping at %d GPU workers", i + 1, exc, self.n_workers)
                break
            self.instances.append(gen)
            self._free.put(gen)
            self.n_workers += 1
        if self.n_workers == 0:
            raise RuntimeError("[gen] LlamaPool.from_placement: no workers loaded")
        logger.info("[gen] LlamaPool ready: %d workers (placement %s)",
                    self.n_workers, placement[:self.n_workers])
        return self

    def __call__(self, system: str, user: str, **kw) -> str:
        gen = self._free.get()
        try:
            return gen(system, user, **kw)
        finally:
            self._free.put(gen)

    def close(self) -> None:
        for gen in self.instances:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
        self.instances = []


# ---------------------------------------------------------------------------
# VRAM / worker planning helpers
# ---------------------------------------------------------------------------

def _visible_gpus() -> list[int]:
    try:
        import torch
        return [0] if torch.cuda.is_available() else []
    except Exception:
        return []


def _gguf_size_bytes(model_key: str, quant: str) -> int:
    import fnmatch
    try:
        _, filename = resolve_generator(model_key, quant)
    except Exception:
        return 0
    best = 0
    for p in Path("models").rglob("*.gguf"):
        if fnmatch.fnmatch(p.name, filename) or (quant in p.name and quant in filename):
            best = max(best, p.stat().st_size)
    return best


def _free_vram_per_gpu(gpus: list[int]) -> dict[int, int]:
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        free, _ = torch.cuda.mem_get_info(0)
        return {0: int(free)}
    except Exception:
        return {}


def plan_noderag_workers(model_key: str, quant: str, *,
                         gpus: list[int] | None = None,
                         ctx: int = 16384, ceiling: int = 8) -> list[int]:
    """Decide how many extra NodeRAG worker contexts to spawn (VRAM-aware)."""
    override = os.environ.get("DISTRACE_NODERAG_WORKERS")
    if override is not None:
        try:
            want = max(0, int(override))
        except ValueError:
            want = 0
        return [0] * min(want, ceiling)

    gpus = gpus if gpus is not None else _visible_gpus()
    free = _free_vram_per_gpu(gpus)
    if not free:
        return []

    gguf = _gguf_size_bytes(model_key, quant)
    weights = int(gguf * 1.15) if gguf else 6 * 1024 ** 3
    kv_margin = int((ctx / 16384) * 1.5 * 1024 ** 3)
    overhead = 1 * 1024 ** 3
    per_worker = max(weights + kv_margin + overhead, 1)
    reserve = per_worker
    n = max(0, int((free.get(0, 0) - reserve) // per_worker))
    return [0] * min(n, ceiling)


# ---------------------------------------------------------------------------
# Public factory / helpers
# ---------------------------------------------------------------------------

def make_generator(model_key: str, quant: str, *, server_url: str | None = None,
                   workers: int = 1, **kw):
    """Build a generator:
      • server_url set → ServerGenerator (continuous batching via llama-server);
      • workers > 1   → LlamaPool (N in-process models);
      • otherwise     → single LlamaGenerator.
    """
    if server_url:
        return ServerGenerator(server_url, model=model_key,
                               temperature=kw.get("temperature", 0.0))
    if workers and workers > 1:
        return LlamaPool(model_key, quant, workers, **kw)
    return LlamaGenerator(model_key, quant, **kw)


def launch_llama_server(model_key: str, quant: str, *, n_parallel: int = 4,
                        n_ctx: int = 16384, n_gpu_layers: int = -1, port: int = 8080,
                        main_gpu: int = 0, host: str = "127.0.0.1",
                        binary: str = "llama-server", wait_seconds: int = 600):
    """Start a native llama-server and return (process, base_url)."""
    import shutil, subprocess, time, urllib.request, urllib.error
    repo, filename = resolve_generator(model_key, quant)
    exe = shutil.which(binary)
    if exe is None:
        raise RuntimeError(
            f"`{binary}` not on PATH. Build/install llama.cpp's server binary.")
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(Path("models") / "hub"))
    cmd = [exe, "-hf", f"{repo}:{filename}", "--parallel", str(n_parallel),
           "--cont-batching", "-c", str(n_ctx * n_parallel),
           "-ngl", str(n_gpu_layers), "--main-gpu", str(main_gpu),
           "--host", host, "--port", str(port)]
    logger.info("[server] launching: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    base_url = f"http://{host}:{port}/v1"
    health  = f"http://{host}:{port}/health"
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early (code {proc.returncode})")
        try:
            with urllib.request.urlopen(health, timeout=2) as resp:
                if resp.status == 200:
                    logger.info("[server] ready at %s", base_url)
                    return proc, base_url
        except (urllib.error.URLError, OSError):
            time.sleep(1.0)
    proc.terminate()
    raise RuntimeError(f"llama-server did not become healthy within {wait_seconds}s")


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
    """Safely close any generator, freeing VRAM."""
    if generate is None:
        return
    close = getattr(generate, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:
            logger.debug("[gen] close_generator raised (ignored): %s", exc)
