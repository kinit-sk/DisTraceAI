from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)


# ---- device placement ---------------------------------------------------
def get_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_embedder_device() -> str:
    """Pin the embedder to its own GPU when more than one is available, so the 4B
    embedder never competes with the LLM for VRAM. Override via
    DISTRACE_EMBEDDER_DEVICE."""
    override = os.environ.get("DISTRACE_EMBEDDER_DEVICE")
    if override:
        return override
    import torch
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return f"cuda:{torch.cuda.device_count() - 1}"
    return get_device()


def make_embedder(model_name: str, *, max_seq_length: int = 512, fp16: bool = True):
    """Load the SentenceTransformer embedder.

    Two memory safeguards matter for a 4B embedder on large corpora (e.g.
    embedding the ~435k MultiClaim fact-checks): load the weights in fp16 (halves
    the ~16 GB fp32 footprint) and CAP the sequence length. Qwen3-Embedding has a
    very large native context (32k); fact-check claims are short, so a long cap
    wastes enormous activation memory and triggers CUDA OOM. Override per env:
    DISTRACE_EMBED_FP32=1 to disable fp16, DISTRACE_EMBED_MAXLEN=N to change cap.
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
        except Exception as exc:                       # pragma: no cover - runtime only
            logger.warning("[models] could not cast embedder to fp16 (%s)", exc)
    logger.info("[models] embedder %s on %s (max_seq=%s, fp16=%s)",
                model_name, device, getattr(model, "max_seq_length", "?"), use_fp16)
    return model


def encode_with_backoff(embedder, texts: Sequence[str],
                        initial_batch_size: int = 32, min_batch_size: int = 4,
                        show_progress: bool = False):
    """Encode with batch-size back-off on CUDA OOM (README §7).

    Halves the batch on OOM down to ``min_batch_size`` before, as a last resort,
    falling back to CPU. A 4B embedder cannot encode thousands of chunks at the
    old ``batch_size=256``.
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
            logger.warning("[models] CUDA OOM during encode; retry at batch_size=%d", bs)
    logger.warning("[models] GPU encode failed at min batch; falling back to CPU "
                   "(set DISTRACE_EMBEDDER_DEVICE=cuda:1 on a 2-GPU node to avoid this)")
    return embedder.encode(list(texts), batch_size=min(16, initial_batch_size),
                           convert_to_numpy=True, show_progress_bar=show_progress, device="cpu")


# ---- generator (llama.cpp) ------------------------------------------------
# GGUF catalogue: model key -> (HF repo, filename template). The default
# generator is Gemma E4B (README §7); Context-1 is the agentic retriever/verifier.
_CATALOGUE: dict[str, tuple[str, str]] = {
    "gemma4-e2b":     ("unsloth/gemma-4-E2B-it-GGUF", "gemma-4-e2b-it-{quant}.gguf"),
    "gemma4-e4b":     ("unsloth/gemma-4-E4B-it-GGUF", "gemma-4-e4b-it-{quant}.gguf"),
    "gemma4-12b":     ("unsloth/gemma-4-12b-it-GGUF", "gemma-4-12b-it-{quant}.gguf"),
    "gemma4-26b-a4b": ("unsloth/gemma-4-26B-A4B-it-GGUF", "gemma-4-26B-A4B-it-{quant}.gguf"),
    "gemma4-31b":     ("unsloth/gemma-4-31B-it-GGUF", "gemma-4-31B-it-{quant}.gguf"),
    "qwen3.5-2b":     ("lmstudio-community/Qwen3.5-2B-GGUF", "Qwen3.5-2B-{quant}.gguf"),
    "qwen3.5-4b":     ("lmstudio-community/Qwen3.5-4B-GGUF", "Qwen3.5-4B-{quant}.gguf"),
    "qwen3.5-9b":     ("lmstudio-community/Qwen3.5-9B-GGUF", "Qwen3.5-9B-{quant}.gguf"),
    "qwen3.5-27b":    ("lmstudio-community/Qwen3.5-27B-GGUF", "Qwen3.5-27B-{quant}.gguf"),
    "context-1":      ("nicolasembleton/context-1-GGUF", "*{quant}*.gguf"),
}
# Some GGUF repos vary filename conventions; llama.cpp's from_pretrained accepts a
# glob for `filename`, so the catalogue may use a "*{quant}*.gguf" pattern to stay
# tolerant of exact naming (used for context-1, whose repo naming is not fixed).
# context defaults (n_ctx); Context-1 needs more headroom for multi-turn history
_DEFAULT_CTX = {"context-1": 32768}


def resolve_generator(model_key: str, quant: str) -> tuple[str, str]:
    """Map a model key (or 'vendor/key' alias) + quant to (repo_id, filename)."""
    key = model_key if model_key in _CATALOGUE else model_key.split("/")[-1]
    if key not in _CATALOGUE:
        raise ValueError(f"Unknown generator {model_key!r}. Available: {sorted(_CATALOGUE)}")
    repo, filename_tmpl = _CATALOGUE[key]
    return repo, filename_tmpl.format(quant=quant)


class LlamaGenerator:
    """Callable llama.cpp generator implementing the `generate(system, user, **kw)`
    contract used across the pipeline. Per-call `temperature`/`max_tokens` are
    honoured (SpecFi-C raises temperature for diverse hypotheticals); `/no_think`
    is appended unless `thinking=True`.
    """

    def __init__(self, model_key: str, quant: str, *,
                 context_size: int | None = None, temperature: float = 0.0,
                 main_gpu: int = 0) -> None:
        from llama_cpp import Llama
        repo, filename = resolve_generator(model_key, quant)
        self.model_key, self.quant = model_key, quant
        self.temperature = temperature
        context_size = context_size or _DEFAULT_CTX.get(
            model_key if model_key in _CATALOGUE else model_key.split("/")[-1], 16384)
        n_threads = os.cpu_count() or 4
        logger.info("[gen] loading %s (%s, ctx=%d)", filename, repo, context_size)

        last_exc: Exception | None = None
        cache_dir = Path("models")
        cache_dir.mkdir(exist_ok=True)

        # Only these errors are sensitive to the GPU layer count; stepping down
        # n_gpu_layers can fix them. A genuine "failed to load model from file"
        # (missing/corrupt GGUF) fails identically at every layer count, so on a
        # FIRST attempt it must NOT trigger the ladder. But once a VRAM failure
        # has occurred, a subsequent "failed to load model from file" is usually
        # fallout from the half-built llama.cpp state, so we keep retrying lower.
        vram_markers = ("out of memory", "cuda", "failed to create llama_context",
                        "ggml_backend", "cublas", "device memory")
        file_markers = ("failed to load model", "failed to load model from file")
        seen_vram = False

        for n_gpu in (-1, 32, 16, 0):   # back off GPU offload under VRAM pressure
            try:
                self.llm = Llama.from_pretrained(
                    repo_id=repo, filename=filename, n_ctx=context_size,
                    n_gpu_layers=n_gpu, n_threads=n_threads, main_gpu=main_gpu,
                    cache_dir=str(cache_dir),
                    verbose=False)
                if n_gpu != -1:
                    logger.warning("[gen] loaded with n_gpu_layers=%d (VRAM pressure)", n_gpu)
                return
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                # Free the half-constructed model now so its (buggy) __del__ runs
                # inside this handler and releases VRAM before the next attempt,
                # instead of firing at an arbitrary later GC.
                self.llm = None
                import gc
                gc.collect()
                is_vram = any(k in msg for k in vram_markers)
                is_file = any(k in msg for k in file_markers)
                if is_vram:
                    seen_vram = True
                # Retry lower on a VRAM error, or on a file error that is
                # plausibly fallout from a prior VRAM failure this run.
                if is_vram or (is_file and seen_vram):
                    logger.warning("[gen] n_gpu_layers=%d failed (%s); retrying with "
                                   "fewer layers", n_gpu, exc)
                    continue
                # A first-attempt file error (or any other error) is not
                # layer-count-sensitive — fail fast with context.
                raise RuntimeError(
                    f"[gen] failed to load {filename} from {repo} "
                    f"(n_gpu_layers={n_gpu}): {exc}") from exc
        raise RuntimeError(
            f"[gen] could not load {filename} from {repo} at any GPU layer "
            f"count (last error: {last_exc})")

    def __call__(self, system: str, user: str, *, temperature: float | None = None,
                 max_tokens: int = 256, thinking: bool = False) -> str:
        if not thinking and "/no_think" not in system:
            system = system + " /no_think"
        resp = self.llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens, top_p=0.8, top_k=20, min_p=0.0,
            presence_penalty=1.5, repeat_penalty=1.0)
        return resp["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        """Release the model and free GPU VRAM deterministically.

        `del generator` only drops the Python reference and leaves freeing to a
        later GC (which also triggers llama-cpp-python's buggy __del__). In a
        multi-step run (e.g. gen_dataset) the next step's model would otherwise
        load while this one still occupies VRAM. Call this between steps.
        """
        llm = getattr(self, "llm", None)
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                try:
                    close()
                except Exception as exc:                # pragma: no cover - runtime
                    logger.debug("[gen] llm.close() raised (ignored): %s", exc)
            self.llm = None
        import gc
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


class ServerGenerator:
    """OpenAI-compatible client for a llama.cpp server started with
    `--parallel N --cont-batching`. Concurrent __call__s (e.g. via `parallel_map`)
    are decoded in parallel by the server's continuous batching — the speed-up
    the in-process single-context model cannot provide."""

    concurrent = True   # safe to call from multiple threads at once

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
                         {"role": "user", "content": user}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()

    def close(self) -> None:
        """No-op: the server owns the model; nothing to free in-process."""
        return None


def generator_is_concurrent(generate) -> bool:
    """True if `generate` can be called concurrently for a real speed-up (a
    server-backed generator). In-process llama.cpp is lock-serialised → False."""
    return bool(getattr(generate, "concurrent", False))


def parallel_map(fn, items, max_workers: int = 1, description: str | None = None):
    """Apply `fn` over `items`, up to `max_workers` at a time, results in order.

    Only worth >1 worker when `fn` releases the GIL / does I/O — i.e. a
    server-backed generator. `max_workers<=1` runs sequentially with no thread
    overhead. Progress is shown via `core.progress.track` when `description` set.
    """
    from core.progress import track
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


def launch_llama_server(model_key: str, quant: str, *, n_parallel: int = 4,
                        n_ctx: int = 16384, n_gpu_layers: int = -1, port: int = 8080,
                        main_gpu: int = 0, host: str = "127.0.0.1",
                        binary: str = "llama-server", wait_seconds: int = 600):
    """Best-effort: start a native `llama-server` with continuous batching and
    return (process, base_url). The total KV cache is sized n_ctx*n_parallel (the
    context is a shared budget split across slots). Prefer the native binary —
    the `llama-cpp-python` server serialises requests. Caller must `.terminate()`.

    ``n_ctx`` is the PER-SLOT context window; for parity with the in-process path
    (and so NodeRAG's chat prompts do not overflow) pass the same value as
    ``generator_context_size`` (default 32768).  It defaults to 16384 here only to
    bound the shared KV cache (16384*4 slots) on smaller GPUs; raise it when the
    generator context is larger.
    """
    import shutil, subprocess, time, urllib.request, urllib.error
    repo, filename = resolve_generator(model_key, quant)
    exe = shutil.which(binary)
    if exe is None:
        raise RuntimeError(
            f"`{binary}` not on PATH. Build/install llama.cpp's server (the native "
            f"binary does true continuous batching), or run one yourself and set "
            f"generator_server_url.")
    # Ensure the native server also uses models/ for its HF cache.
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(Path("models") / "hub"))
    cmd = [exe, "-hf", f"{repo}:{filename}", "--parallel", str(n_parallel),
           "--cont-batching", "-c", str(n_ctx * n_parallel), "-ngl", str(n_gpu_layers),
           "--main-gpu", str(main_gpu), "--host", host, "--port", str(port)]
    logger.info("[server] launching: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env)
    base_url = f"http://{host}:{port}/v1"
    health = f"http://{host}:{port}/health"
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


def make_generator(model_key: str, quant: str, *, server_url: str | None = None,
                   workers: int = 1, **kw):
    """Build a generator:
      • `server_url` set → `ServerGenerator` (llama.cpp server, continuous batching);
      • `workers` > 1   → `LlamaPool` of that many in-process models (parallel
        decoding with no external server — VRAM scales with the worker count);
      • otherwise       → a single in-process `LlamaGenerator` (serialised).
    """
    if server_url:
        return ServerGenerator(server_url, model=model_key,
                               temperature=kw.get("temperature", 0.0))
    if workers and workers > 1:
        return LlamaPool(model_key, quant, workers, **kw)
    return LlamaGenerator(model_key, quant, **kw)


def _visible_gpus() -> list[int]:
    override = os.environ.get("DISTRACE_GENERATOR_GPUS")
    if override:
        return [int(x) for x in override.replace(",", " ").split()]
    try:
        import torch
        n = torch.cuda.device_count()
        return list(range(n)) if n else []
    except Exception:
        return []


def _gguf_size_bytes(model_key: str, quant: str) -> int:
    """Best-effort on-disk size of the resolved GGUF (a weights-footprint proxy).

    Searches the local HF cache (models/) for a file matching the resolved
    filename, tolerating the catalogue's '*{quant}*.gguf' glob patterns.
    Returns 0 when nothing matches (caller falls back to a conservative default).
    """
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
    """Return {gpu_index: free_bytes} measured AFTER current allocations.

    Uses torch.cuda.mem_get_info so the figure reflects the embedder + any
    already-resident generator instance. Empty dict when CUDA is unavailable.
    """
    out: dict[int, int] = {}
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        for g in (gpus or range(torch.cuda.device_count())):
            try:
                free, _total = torch.cuda.mem_get_info(g)
                out[int(g)] = int(free)
            except Exception:
                pass
    except Exception:
        return {}
    return out


def plan_noderag_workers(model_key: str, quant: str, *,
                         gpus: list[int] | None = None,
                         ctx: int = 16384,
                         ceiling: int = 8) -> list[int]:
    """Decide how many extra NodeRAG worker contexts to spawn and on which GPUs.

    Strategy (auto, no manual tuning - overridable via DISTRACE_NODERAG_WORKERS):
      1. Estimate one worker's footprint = GGUF size x 1.15 (weights + overhead)
         + a KV-cache margin scaled by ctx. Falls back to a conservative 6 GiB
         when the GGUF size cannot be read.
      2. Measure free VRAM per visible GPU (after the embedder + the first
         generator are already resident).
      3. Pack floor(free / per_worker) workers onto each GPU, returning a flat
         list of GPU indices (one entry per worker to spawn) up to `ceiling`.

    Returns a list of GPU indices, e.g. [0, 0, 1, 1] = four workers, two per GPU.
    An empty list means "no extra workers" (single-context / CPU fallback).
    """
    override = os.environ.get("DISTRACE_NODERAG_WORKERS")
    gpus = gpus if gpus is not None else _visible_gpus()

    free = _free_vram_per_gpu(gpus)
    if not free:
        return []   # no CUDA visibility - single context only

    gguf = _gguf_size_bytes(model_key, quant)
    weights = int(gguf * 1.15) if gguf else 6 * 1024**3
    kv_margin = int((ctx / 16384) * 1.5 * 1024**3)   # ~1.5 GiB at 16k ctx
    per_worker = max(weights + kv_margin, 1)

    if override is not None:
        try:
            want = max(0, int(override))
        except ValueError:
            want = 0
        usable = [g for g in (gpus or sorted(free)) if free.get(g, 0) > per_worker]
        if not usable:
            return []
        return [usable[i % len(usable)] for i in range(min(want, ceiling))]

    # Auto: floor(free / per_worker) per GPU (the floor leaves a safety reserve),
    # then INTERLEAVE across GPUs so the ceiling is distributed evenly rather
    # than packing the first GPU to capacity and starving the rest.
    capacity = {g: int(free.get(g, 0) // per_worker) for g in (gpus or sorted(free))}
    plan: list[int] = []
    order = [g for g in (gpus or sorted(free)) if capacity.get(g, 0) > 0]
    round_no = 0
    while order and len(plan) < ceiling:
        progressed = False
        for g in order:
            if round_no < capacity[g]:
                plan.append(g)
                progressed = True
                if len(plan) >= ceiling:
                    break
        round_no += 1
        if not progressed:
            break
    return plan[:ceiling]


class LlamaPool:
    """A pool of independent in-process llama.cpp models for parallel decoding
    when an external server is not permitted (e.g. locked-down HPC).

    Each worker owns its own `LlamaGenerator` (its own weights + KV cache), handed
    out through a queue so concurrent callers always use distinct instances and
    the GPU decodes them in parallel. Drop-in for the single generator, and
    `concurrent=True` so `parallel_map` fans work out across the workers. Workers
    are spread round-robin across the visible GPUs.

    VRAM scales with `n_workers` — each instance is a full model copy. Size it to
    fit (small quant helps); loading uses the same GPU-layer back-off as the
    single generator, per instance.
    """

    concurrent = True

    def __init__(self, model_key: str, quant: str, n_workers: int, *,
                 gpus: list[int] | None = None, **kw) -> None:
        import queue
        self.n_workers = max(1, int(n_workers))
        gpus = gpus if gpus is not None else _visible_gpus()
        self._free: "queue.Queue" = queue.Queue()
        self.instances: list = []
        for i in range(self.n_workers):
            inst_kw = dict(kw)
            if gpus:
                inst_kw["main_gpu"] = gpus[i % len(gpus)]
            logger.info("[gen] LlamaPool: loading worker %d/%d%s", i + 1, self.n_workers,
                        f" on cuda:{inst_kw['main_gpu']}" if gpus else "")
            gen = LlamaGenerator(model_key, quant, **inst_kw)
            self.instances.append(gen)
            self._free.put(gen)
        logger.info("[gen] LlamaPool ready: %d workers across GPUs %s", self.n_workers,
                    gpus or "default")

    @classmethod
    def from_placement(cls, model_key: str, quant: str, placement: list[int],
                       **kw) -> "LlamaPool":
        """Build a pool from an explicit per-worker GPU placement list.

        ``placement`` is a list of GPU indices, one entry per worker to spawn
        (e.g. [0, 0, 1] = two workers on GPU 0, one on GPU 1). Loading is
        INCREMENTAL and OOM-TOLERANT: if a worker fails to load (e.g. the VRAM
        estimate was slightly optimistic, or another process grabbed memory),
        loading stops and the pool keeps whatever workers succeeded. This makes
        the auto-sizing estimate only need to be roughly right.
        """
        import queue
        self = cls.__new__(cls)
        self.n_workers = 0
        self._free = queue.Queue()
        self.instances = []
        for i, gpu in enumerate(placement):
            inst_kw = dict(kw)
            inst_kw["main_gpu"] = gpu
            try:
                logger.info("[gen] LlamaPool: loading worker %d/%d on cuda:%d",
                            i + 1, len(placement), gpu)
                gen = LlamaGenerator(model_key, quant, **inst_kw)
            except Exception as exc:
                logger.warning("[gen] LlamaPool: worker %d failed to load (%s); "
                               "stopping at %d workers", i + 1, exc, self.n_workers)
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
        gen = self._free.get()          # blocks until a worker is free
        try:
            return gen(system, user, **kw)
        finally:
            self._free.put(gen)

    def close(self) -> None:
        """Close every worker, freeing all pooled models' VRAM."""
        for gen in self.instances:
            close = getattr(gen, "close", None)
            if callable(close):
                close()
        self.instances = []


def close_generator(generate) -> None:
    """Safely close any generator (LlamaGenerator/LlamaPool/ServerGenerator/None).

    Replaces bare `del llm`, which only drops the reference and defers VRAM
    release (and the buggy llama-cpp __del__) to an arbitrary later GC. Use this
    between pipeline steps so each model's VRAM is freed before the next loads.
    """
    if generate is None:
        return
    close = getattr(generate, "close", None)
    if callable(close):
        try:
            close()
        except Exception as exc:                        # pragma: no cover - runtime
            logger.debug("[gen] close_generator raised (ignored): %s", exc)


# ---- CW detector (HF sequence-classification) -----------------------------
def make_cw_detector(model_path: str):
    """Load the fine-tuned check-worthiness classifier (mdb-multicw / xlm-multicw)."""
    from core.claims.cw_detector import CheckWorthinessDetector
    return CheckWorthinessDetector(model_path)
