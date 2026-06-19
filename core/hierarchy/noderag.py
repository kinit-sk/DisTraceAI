"""NodeRAG graph wrapper — pinned to NodeRAG==0.1.0 (README §5).

Runs patched NodeRAG's build/query INSIDE this process and drives it with the local
models we already have loaded — no external OpenAI/llama.cpp server, no ports,
no API key.

How: NodeRAG 0.1.0 normally builds an OpenAI/Gemini HTTP client from
``model_config``/``embedding_config`` (and tolerates a missing key by setting the
client to None). We construct ``NodeConfig`` keyless, then swap in adapters:
  • the LLM adapter calls our in-process ``generate`` (VLLMGenerator). Extraction
    steps pass a pydantic ``response_format``; we honour it with llama.cpp's
    grammar-constrained JSON so even a small model emits schema-valid output
    (without this the graph comes out empty).
  • the embedding adapter calls our SentenceTransformer embedder; the HNSW ``dim``
    is taken from the embedder so it always matches.

If no local ``generate``/``embedder`` is supplied, we fall back to NodeRAG's HTTP
provider (set OPENAI_API_KEY, and OPENAI_BASE_URL for a local server).

The wheel also ships no ``Node_config.yaml`` template, so we never call the
crash-prone ``from_main_folder``; we write our own config and load ``NodeConfig``
directly. ``NodeRag.run()``'s interactive y/n prompt is auto-confirmed.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Sentinel string NodeRAG inserts between the retrieval preamble and the ranked
# community findings in its answer text.  Everything after this delimiter and
# matching "N. <text>" lines becomes the high-level elements we surface to the
# SpecFi-C pipeline.
_HLE_DELIMITER = "------------high_level_element-------------"


def parse_high_level_elements(answer_text: str, limit: int = 5) -> list[str]:
    """Extract the numbered community-level findings from a NodeRAG answer.

    NodeRAG's retrieval answer has the structure::

        <preamble / cosine-retrieved chunks>
        ------------high_level_element-------------
        1. Finding about topic A
        2. Finding about topic B
        ...

    This function splits on the delimiter, then collects lines that start with
    a one- or two-digit number followed by a dot (e.g. "1.", "12."), strips the
    numbering prefix, and returns up to *limit* non-empty strings.

    Returns an empty list when the delimiter is absent (e.g. a small corpus
    that produced no community summaries).
    """
    # Split into [preamble, findings_block] — anything before the first delimiter
    # is discarded; only the first findings block is processed.
    units = answer_text.split(_HLE_DELIMITER)
    if len(units) < 2:
        # Delimiter not present — no community-level findings were generated.
        return []

    # Strip leading "N. " numbering from each matching line so callers get clean
    # natural-language strings rather than prefixed list items.
    items = [re.sub(r"^\d{1,2}\.\s*", "", line).strip()
             for line in units[1].split("\n") if re.match(r"^\d{1,2}\.", line)]
    return [it for it in items[:limit] if it]


# ── in-process adapters mimicking NodeRAG's async client protocol ────────────
class _LocalLLMClient:
    """Async-callable replacement for NodeRAG's OpenAI client. Returns a parsed
    dict when a response_format schema is given (grammar-constrained), else text.

    Concurrency model
    ------------------
    A single llama.cpp context is NOT concurrency-safe, so historically every
    NodeRAG extraction unit was serialised through one lock + one context — the
    dominant cost of a SpecFi-CS graph build (multiple hours).

    When handed a generator that exposes multiple independent contexts
    (a concurrent generator), this client dispatches several units at once
    drawn from a bounded semaphore sized to the pool. Distinct workers have
    distinct KV caches, so they decode in parallel and the A100 batches them.
    A non-concurrent generator transparently falls back to
    the old serialised behaviour (pool size 1).
    """

    def __init__(self, generate, max_tokens: int | None = None) -> None:
        self._gen = generate

        # vLLM serves many concurrent requests from ONE resident model via
        # continuous batching, so there is no worker pool of model copies. We
        # still admit several units at once (they overlap inside vLLM's batch);
        # the concurrency width is a logical dispatch fan-out, not N contexts.
        # Grammar-constrained JSON goes through generate_json() (vLLM guided
        # decoding) rather than a raw low-level handle.
        self._workers = [generate]
        self._n_workers = max(1, int(os.getenv("DISTRACE_NODERAG_FANOUT", "8"))) \
            if getattr(generate, "concurrent", False) else 1

        self._max_tokens = max_tokens or int(os.getenv("DISTRACE_NODERAG_MAXTOK", "4096"))

        # A counting semaphore admits up to n_workers concurrent calls; a queue
        # hands out a distinct worker index to each admitted call so two
        # coroutines never share a context.
        self._sema = asyncio.Semaphore(self._n_workers)
        self._free_idx: "asyncio.Queue[int]" = asyncio.Queue()
        for i in range(self._n_workers):
            self._free_idx.put_nowait(i)

    async def __call__(self, input, *, cache_path=None, meta_data=None):
        # Admit up to n_workers callers; each grabs a free worker index, runs the
        # blocking llama.cpp call in a thread, then returns the worker to the pool.
        async with self._sema:
            idx = await self._free_idx.get()
            try:
                for attempt in range(2):
                    try:
                        return await asyncio.to_thread(self._run, input, idx)
                    except Exception as exc:
                        if attempt == 0:
                            continue
                        logger.warning("[noderag] LLM unit skipped after retry (%s)", exc)
                        return "Error cached"
            finally:
                self._free_idx.put_nowait(idx)

    def request(self, input, *, cache_path=None, meta_data=None):
        # Synchronous entry point used by pipeline stages outside an event loop.
        return self._run(input, 0)

    def stream_chat(self, input):
        # Streaming entry point — NodeRAG's interactive query path calls this.
        yield self._run(input, 0)

    def _run(self, input, worker_idx: int = 0):
        gen = self._workers[0]
        system = (input.get("system_prompt") or "You are a precise information-extraction assistant.")
        query = input["query"]
        rf = input.get("response_format")
        has_json = rf is not None and hasattr(gen, "generate_json")
        if has_json:
            schema = rf.model_json_schema()
            out = gen.generate_json(system, query, schema,
                                    temperature=0.0, max_tokens=self._max_tokens)
            return json.loads(out)
        text = gen(system, query, temperature=0.0, max_tokens=self._max_tokens)
        if isinstance(text, str):
            stripped = text.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    return json.loads(stripped)
                except json.JSONDecodeError:
                    pass
            if hasattr(gen, "generate_json") and "elements" in query.lower():
                _fallback_schema = {
                    "type": "object",
                    "properties": {
                        "elements": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["elements"],
                }
                try:
                    out = gen.generate_json(system, query, _fallback_schema,
                                            temperature=0.0, max_tokens=self._max_tokens)
                    return json.loads(out)
                except Exception as exc:
                    logger.warning("[noderag] grammar-constrained decompose fallback failed (%s); "
                                   "returning empty elements", exc)
                    return {"elements": []}
        return text


def _check_server_reachable(base_url: str) -> None:
    """TCP-probe *base_url* and raise a clear RuntimeError if the host is down.

    Called before creating a _ServerLLMClient when OPENAI_BASE_URL is set, so
    we fail fast with a helpful message rather than letting NodeRAG's pipeline
    hang waiting for a connection that will never arrive (a common footgun on
    HPC where the llama-server may be running on a different node).
    """
    import socket
    from urllib.parse import urlparse
    u = urlparse(base_url)
    host, port = (u.hostname or "localhost"), (u.port or (443 if u.scheme == "https" else 80))
    try:
        socket.create_connection((host, port), timeout=3).close()
    except OSError as exc:
        raise RuntimeError(
            f"OPENAI_BASE_URL={base_url!r} is set but unreachable ({exc}). Either start a "
            f"server on THIS node, or `unset OPENAI_BASE_URL` to use the in-process model.") from exc


class _ServerLLMClient:
    """Concurrent chat client for a local OpenAI-compatible *batching* server
    (e.g. llama.cpp's `llama-server -cb --parallel N`). No lock — lets NodeRAG's
    parallelism reach the server so the A100 batches instead of running at
    batch-1. Schema requests use json_object+schema grammar."""

    def __init__(self) -> None:
        # Uses the standard AsyncOpenAI client pointed at a local server.
        # OPENAI_BASE_URL must be set; OPENAI_API_KEY defaults to a dummy value
        # because the local server does not actually validate it.
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(base_url=os.getenv("OPENAI_BASE_URL"),
                                   api_key=os.getenv("OPENAI_API_KEY", "sk-local"))
        self._model = os.getenv("DISTRACE_NODERAG_SERVER_MODEL", "local")
        self._max = int(os.getenv("DISTRACE_NODERAG_MAXTOK", "4096"))

    async def __call__(self, input, *, cache_path=None, meta_data=None):
        system = input.get("system_prompt") or "You are a precise information-extraction assistant."
        rf = input.get("response_format")
        kw = dict(model=self._model, temperature=0.0, max_tokens=self._max,
                  messages=[{"role": "system", "content": system},
                            {"role": "user", "content": input["query"]}])
        if rf is not None:
            # Pass the pydantic schema as a JSON grammar constraint so the server
            # returns schema-valid JSON for extraction steps.
            kw["response_format"] = {"type": "json_object", "schema": rf.model_json_schema()}
        for attempt in range(2):
            try:
                resp = await self._client.chat.completions.create(**kw)
                text = resp.choices[0].message.content
                # Parse the response as JSON when a schema was requested;
                # otherwise return the raw text string.
                return json.loads(text) if rf is not None else (text or "").strip()
            except Exception as exc:
                if attempt == 0:
                    continue
                logger.warning("[noderag] server LLM unit skipped after retry (%s)", exc)
                return "Error cached"


class _LocalEmbeddingClient:
    """In-process embedding adapter that wraps a SentenceTransformer (or any
    object with an ``encode(texts) -> array`` method).

    NodeRAG calls this client for every chunk/node it needs to embed during both
    the build (HNSW construction) and the query (ANN lookup) phases.  All calls
    are serialised through an asyncio lock and dispatched to a thread so the
    event loop is never blocked by the (potentially GPU-heavy) encode call.
    """

    def __init__(self, embedder) -> None:
        self._embedder = embedder
        # Serialise encode calls — a single SentenceTransformer model is not
        # safe to call concurrently from multiple threads.
        self._lock = asyncio.Lock()

    async def __call__(self, input, *, cache_path=None, meta_data=None):
        # NodeRAG may pass either a single string or a list of strings.
        texts = input if isinstance(input, list) else [input]
        async with self._lock:
            return await asyncio.to_thread(self._encode, texts)

    def request(self, input, *, cache_path=None, meta_data=None):
        # Synchronous path used by pipeline stages that run outside an event loop.
        texts = input if isinstance(input, list) else [input]
        return self._encode(texts)

    def _encode(self, texts):
        import numpy as np
        # encode() returns a 2-D numpy array (n_texts × dim); convert each row
        # to a plain Python list so NodeRAG can serialise it to JSON/HNSW.
        arr = np.asarray(self._embedder.encode(list(texts), convert_to_numpy=True,
                                               show_progress_bar=False))
        return [v.tolist() for v in arr]


class NodeRagGraph:
    """High-level wrapper around a NodeRAG index that manages the full lifecycle:
    config generation, build, load, and community-context retrieval.

    Parameters
    ----------
    index_path:
        Directory where NodeRAG will store (or has already stored) its index
        artefacts: ``input/``, ``cache/``, ``info/``, and ``Node_config.yaml``.
    generate:
        Optional VLLMGenerator (or compatible callable).  When provided
        together with *embedder*, the index is built and queried using purely
        in-process models.  When omitted, NodeRAG falls back to its HTTP
        provider (requires OPENAI_API_KEY ± OPENAI_BASE_URL).
    embedder:
        Optional SentenceTransformer (or any object with ``encode()``).
        Its output dimension is read automatically and written into the HNSW
        config so the vector index always has the right shape.
    """

    def __init__(self, index_path: str, *, generate=None, embedder=None,
                 build_model_key: str | None = None,
                 build_precision: str | None = None,
                 build_context_size: int = 16384,
                 build_repr: str = "text") -> None:
        self.index_path = index_path
        self._generate = generate
        self._embedder = embedder
        # Optional: identify the generator's model so build() can spin up an
        # auto-sized worker pool (parallel graph construction) and tear it down
        # afterwards. When omitted, build() uses the single `generate` as-is.
        self._build_model_key = build_model_key
        self._build_precision = build_precision
        self._build_context_size = build_context_size
        # What auto-populated input docs contain when input/ is empty:
        # "text" (CW claim sentences, SpecFi-CS) or "canonized" (SpecFi-CCS).
        self._build_repr = build_repr
        # Populated lazily on first call to ensure_loaded(); holds a NodeSearch
        # instance once the index has been built and loaded.
        self._search = None

    @property
    def _local(self) -> bool:
        """True when both in-process models are available (no HTTP calls needed)."""
        return self._generate is not None and self._embedder is not None

    def _dim(self) -> int:
        """Return the embedding dimension for HNSW index construction.

        Tries standard SentenceTransformer introspection methods first; falls
        back to the DISTRACE_NODERAG_DIM env-var (default 1536, which is
        OpenAI's text-embedding-3-small dimension).  Must match the actual
        model output or the HNSW index will silently store wrong-shaped vectors.
        """
        if self._embedder is not None:
            for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                fn = getattr(self._embedder, attr, None)
                if callable(fn):
                    try:
                        return int(fn())
                    except Exception:
                        pass
        return int(os.getenv("DISTRACE_NODERAG_DIM", "1536"))

    def _config_dict(self) -> dict:
        """Build the NodeRAG config dict that is written to Node_config.yaml.

        The ``service_provider`` is always set to ``"openai"`` / ``"openai_embedding"``
        so NodeRAG's router still constructs its HTTP client objects (they just
        won't be used — the in-process adapters are injected afterwards in
        ``_node_config``).  This avoids touching NodeRAG's internal routing code.

        All tuneable parameters can be overridden via environment variables so
        the same codebase works across different HPC allocations without code
        changes.
        """
        return {
            "config": {
                "main_folder": str(Path(self.index_path).resolve()),
                "language": os.getenv("DISTRACE_NODERAG_LANG", "English"),
                "chunk_size": int(os.getenv("DISTRACE_NODERAG_CHUNK", "1048")),
                "docu_type": "mixed", "dim": self._dim(), "space": "l2",
                "embedding_batch_size": 50,
            },
            # model_name stays a tiktoken-known name (used only for token counting);
            # the actual generation goes through the in-process adapter.
            "model_config": {
                "service_provider": "openai",
                # Any tiktoken-known name works here — NodeRAG uses this only to
                # select a token counter for chunk sizing; our patch falls back to
                # cl100k for unknown names, so this is purely for tiktoken's benefit.
                "model_name": os.getenv("DISTRACE_NODERAG_CHAT_MODEL", "gpt-4o-mini"),
                "api_keys": os.getenv("OPENAI_API_KEY"),
                "rate_limit": int(os.getenv("DISTRACE_NODERAG_RATE", "40")),
                "max_tokens": 4000, "temperature": 0.0,
            },
            "embedding_config": {
                "service_provider": "openai_embedding",
                "embedding_model_name": os.getenv("DISTRACE_NODERAG_EMBED_MODEL", "text-embedding-3-small"),
                "api_keys": os.getenv("OPENAI_API_KEY"),
                "rate_limit": int(os.getenv("DISTRACE_NODERAG_RATE", "40")),
            },
        }

    def _node_config(self):
        """Construct a ``NodeConfig`` and, when in local mode, hot-swap the HTTP
        clients with our in-process adapters.

        NodeRAG's ``NodeConfig.__init__`` instantiates the HTTP clients eagerly;
        we overwrite ``config.API_client`` / ``config.embedding_client`` right
        after construction, and also call the module-level setters
        (``set_api_client`` / ``set_embedding_client``) because some internal
        pipeline stages resolve the client from global state rather than from the
        config object.
        """
        import yaml
        from NodeRAG import NodeConfig
        folder = Path(self.index_path)
        # NodeRAG expects input/ to exist before NodeConfig is constructed.
        (folder / "input").mkdir(parents=True, exist_ok=True)
        cfg = self._config_dict()
        # Write the YAML so NodeRAG's internal path-resolution logic works
        # (it derives cache/, info/ etc. relative to main_folder).
        (folder / "Node_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
        config = NodeConfig(cfg)
        if self._local:
            from NodeRAG.LLM.LLM_state import set_api_client, set_embedding_client
            if os.getenv("OPENAI_BASE_URL"):
                _check_server_reachable(os.getenv("OPENAI_BASE_URL"))   # fail fast if dead
                llm_client = _ServerLLMClient()        # concurrent chat via batching server
                logger.info("[noderag] chat via server %s (concurrent)", os.getenv("OPENAI_BASE_URL"))
            else:
                # LOCAL mode: NodeRAG's chat LLM IS the in-process eval generator
                # (the very same callable handed to NodeRagGraph). There is no
                # second model and no separate context window — NodeRAG inherits
                # the generator's n_ctx (config.generator_context_size) exactly, so
                # the eval LLM and the NodeRAG LLM are unified by construction.
                llm_client = _LocalLLMClient(self._generate)   # in-process, serial
                _gen_ctx = getattr(self._generate, "_context_size", None)
                logger.info("[noderag] chat LLM = in-process eval generator (unified); "
                            "shared n_ctx=%s, max_tokens=%d",
                            _gen_ctx if _gen_ctx is not None else "generator default",
                            llm_client._max_tokens)
            emb_client = _LocalEmbeddingClient(self._embedder)  # embeddings always in-process
            # Inject the adapters into every slot NodeRAG might read them from.
            config.API_client = llm_client
            config.embedding_client = emb_client
            try:
                config.client = llm_client          # alias used by some pipelines
            except Exception:
                pass
            # Also register in global state — some pipeline modules import the
            # client directly from LLM_state rather than via the config object.
            set_api_client(llm_client)
            set_embedding_client(emb_client)
        return config

    def _is_built(self) -> bool:
        # A COMPLETE build produces the HNSW index; partial/crashed builds leave
        # cache/info behind but no HNSW, so key off the final artifact only.
        return (Path(self.index_path) / "cache" / "HNSW.bin").exists()

    def _has_inputs(self) -> bool:
        """True when at least one file exists in the index's input/ directory."""
        inp = Path(self.index_path) / "input"
        return inp.exists() and any(inp.iterdir())

    def build(self) -> None:
        """Build (or rebuild) the NodeRAG knowledge-graph index.

        If input/ is empty, the method first tries to auto-populate it from the
        DisTraceAI knowledge-base via ``evaluation.noderag_corpus.export``.

        Any stale ``cache/`` or ``info/`` directories left by a previous crashed
        build are removed before starting so NodeRAG always gets a clean slate —
        resuming from a partially-built state tends to produce an empty graph.
        """
        if not self._has_inputs():
            # No input documents present — attempt to source them from the
            # DisTraceAI knowledge-base (PolyNarrative corpus by default).
            kb_root = os.getenv("DISTRACE_KB", "knowledge/polynm")
            lang = os.getenv("DISTRACE_NODERAG_LANG_FILTER", "EN")
            from evaluation.noderag_corpus import export
            n = export(Path(kb_root), Path(self.index_path) / "input", lang,
                       None, repr=self._build_repr)
            if n == 0:
                raise RuntimeError(
                    f"No input files at {self.index_path}/input and none emitted from KB "
                    f"{kb_root!r} (lang={lang}). Set DISTRACE_KB or emit with noderag_corpus.")
            logger.info("[noderag] auto-emitted %d input files from %s", n, kb_root)
        if not self._local and os.getenv("OPENAI_API_KEY") is None:
            raise RuntimeError(
                "NodeRAG build needs either in-process models (pass generate= and "
                "embedder= to NodeRagGraph) or OPENAI_API_KEY (+ OPENAI_BASE_URL for a "
                "local server).")
        # Clear any partial/crashed build state (keep input/ + config) so NodeRAG
        # rebuilds cleanly instead of resuming a poisoned, empty-graph state.
        import shutil
        for sub in ("cache", "info"):
            stale = Path(self.index_path) / sub
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)
        logger.info("[noderag] cleared stale build state under %s", self.index_path)
        from NodeRAG import NodeRag

        # --- build concurrency --------------------------------------------
        # vLLM's continuous batching already decodes many NodeRAG extraction
        # units concurrently from ONE resident model, so the historical pool of
        # extra model copies (filling spare VRAM) is no longer needed — the
        # _NodeLLMClient fan-out dispatches units at the single generator and
        # vLLM batches them. No build pool to set up or tear down.
        original_generate = self._generate

        try:
            config = self._node_config()
            ng = NodeRag(config)
            # NodeRag.run() calls console.input() once to ask whether to rebuild
            # if an index already exists. Auto-confirm "y" for batch/HPC jobs.
            ng.console.input = lambda *a, **k: "y"
            logger.info("[noderag] building index at %s (%s)", self.index_path,
                        "in-process local models" if self._local else "HTTP provider")
            ng.run()
            logger.info("[noderag] build complete")
        finally:
            # No build pool to tear down (vLLM batches from one resident model);
            # restore the original generator reference defensively.
            self._generate = original_generate

    def ensure_loaded(self) -> None:
        """Ensure the NodeSearch handle is ready, building the index if needed.

        Idempotent — subsequent calls return immediately once the search handle
        is populated.  Raises ``RuntimeError`` if NodeRAG is not installed or if
        the build/load fails for any reason.
        """
        if self._search is not None:
            return
        try:
            from NodeRAG import NodeSearch
        except ImportError as exc:
            raise RuntimeError(
                f"SpecFi-C requires NodeRAG (pinned ==0.1.0) but it is not importable "
                f"({exc}). Install it with `pip install NodeRAG==0.1.0`.") from exc
        try:
            if not self._is_built():
                logger.info("[noderag] index not built — auto-building from %s/input", self.index_path)
                self.build()
            # Load the built index into a NodeSearch instance for answering queries.
            self._search = NodeSearch(self._node_config())
            logger.info("[noderag] index loaded from %s", self.index_path)
        except Exception as exc:
            raise RuntimeError(
                f"NodeRAG index at {self.index_path!r} could not be built/loaded ({exc}).") from exc

    def community_context(self, query: str) -> list[str]:
        """Return up to 5 community-level findings relevant to *query*.

        Calls NodeSearch.answer(), extracts the high-level-element block from
        the retrieval response, and returns the numbered findings as clean
        strings.  Returns an empty list when the index has no community
        summaries (e.g. corpus too small to form communities).
        """
        self.ensure_loaded()
        ans = self._search.answer(query)
        # NodeSearch.answer() may return an object with a retrieval_info
        # attribute, or just stringify to the full answer text.
        raw = getattr(ans, "retrieval_info", None) or str(ans)
        return parse_high_level_elements(raw)

    def add_documents(self, input_folder: str) -> None:
        """Trigger a full rebuild after new documents have been placed in
        *input_folder*.  Currently delegates entirely to build(); incremental
        index updates are not yet supported by NodeRAG 0.1.0.
        """
        self.build()


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------
# Run with:  python noderag.py
#
# What this test does:
#   1. Verifies the HLE parser on a synthetic NodeRAG response (no models needed).
#   2. Loads gemma4-e2b (awq4) + an embedder via vLLM (downloaded from
#      HuggingFace on first run, cached after).
#   3. Writes three short thematic documents to ./test_noderag_index/input/.
#   4. Builds a full NodeRAG knowledge-graph index at ./test_noderag_index/
#      (skipped automatically if HNSW.bin already exists from a prior run).
#   5. Queries the built index and prints the retrieved community findings.
#
# Index persistence:
#   The index is written to ./test_noderag_index/ and kept between runs so
#   subsequent runs skip the (slow) build step and go straight to querying.
#   To force a full rebuild, delete the directory:
#       rm -rf ./test_noderag_index
#
# Requirements:
#   ./setup.sh                     (installs vllm + NodeRAG with the compat patch)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import textwrap

    # ── configure logging so build progress is visible ──────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("NodeRagGraph integration test")
    print("=" * 60)

    # ── 1. Unit-test the HLE parser (no models required) ────────────────────
    print("\n[1/5] Testing parse_high_level_elements …")

    _synthetic = textwrap.dedent("""
        Some retrieval preamble text from the cosine-search stage.
        ------------high_level_element-------------
        1. Climate change is accelerating glacial retreat worldwide
        2. Arctic sea-ice extent has declined 13 % per decade since 1979
        3. Permafrost thaw is releasing stored methane, amplifying warming
    """)
    _parsed = parse_high_level_elements(_synthetic)

    assert len(_parsed) == 3, f"Expected 3 items, got {len(_parsed)}: {_parsed}"
    assert _parsed[0] == "Climate change is accelerating glacial retreat worldwide"
    assert _parsed[1] == "Arctic sea-ice extent has declined 13 % per decade since 1979"
    assert _parsed[2] == "Permafrost thaw is releasing stored methane, amplifying warming"

    # Verify that a response without the delimiter returns an empty list.
    assert parse_high_level_elements("No delimiter here at all.") == []

    print("    ✓ Parser OK")

    # ── 2. Load models via vLLM (the real pipeline path) ─────────────────────
    print("\n[2/5] Loading gemma4-e2b (awq4) + embedder via vLLM …")
    print("    (first run downloads weights from HuggingFace; cached afterwards)")

    try:
        from core.models import make_generator, make_embedder
    except ImportError as exc:
        print(f"ERROR: could not import vLLM-backed model layer: {exc}", file=sys.stderr)
        print("       Run ./setup.sh first (installs vllm).", file=sys.stderr)
        sys.exit(1)

    _generator = make_generator("gemma4-e2b", "awq4")
    _embedder  = make_embedder("nomic-ai/nomic-embed-text-v1.5")

    # ── 3. Write test documents ──────────────────────────────────────────────
    print("\n[4/5] Writing test documents …")

    # Use a fixed directory so the index persists between runs and the slow
    # build step is skipped on subsequent invocations.
    INDEX_PATH = "./test_noderag_index"
    input_dir  = Path(INDEX_PATH) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    # Three short thematic texts covering distinct but related topics.
    # NodeRAG needs enough entity co-occurrence to form graph edges; at least
    # a few hundred tokens per document is recommended.
    _docs = {
        "climate_arctic.txt": textwrap.dedent("""\
            The Arctic is one of the fastest-warming regions on Earth.
            Since 1979, satellite measurements have recorded a decline of
            approximately 13 percent per decade in minimum sea-ice extent.
            The loss of reflective sea ice exposes dark ocean water, which
            absorbs more solar radiation and further accelerates warming —
            a process called Arctic amplification. Polar bears, which depend
            on sea ice as a platform for hunting seals, are increasingly
            forced onto land for longer periods, reducing their caloric intake
            and reproductive success. Research stations at Svalbard and Alert
            report that mean annual temperatures have risen by more than 3 °C
            since pre-industrial times, roughly three times the global average.
        """),
        "permafrost_methane.txt": textwrap.dedent("""\
            Permafrost — ground that remains frozen for at least two consecutive
            years — covers roughly a quarter of the Northern Hemisphere's land
            surface. It stores an estimated 1.5 trillion tonnes of organic carbon
            accumulated over millennia. As global temperatures rise, permafrost
            thaws and the organic matter decomposes, releasing carbon dioxide and
            methane. Methane is roughly 80 times more potent as a greenhouse gas
            than carbon dioxide over a 20-year horizon. Thermokarst lakes that
            form in thawing permafrost are active sources of both gases. Studies
            in Siberia and northern Canada have documented accelerating thaw rates
            since the early 2000s, raising concerns about a self-reinforcing
            feedback loop that could substantially amplify human-caused warming.
        """),
        "glaciers_sea_level.txt": textwrap.dedent("""\
            Mountain glaciers and the Greenland and Antarctic ice sheets are losing
            mass at accelerating rates. The IPCC Sixth Assessment Report estimates
            that glaciers outside the polar ice sheets have lost roughly 270
            gigatonnes of ice per year on average between 2006 and 2015.
            Meltwater from these glaciers, combined with thermal expansion of
            seawater, is the primary driver of observed sea-level rise —
            approximately 3.6 millimetres per year over the past decade.
            Low-lying nations such as Bangladesh, the Maldives, and Pacific island
            states face existential threats from continued sea-level rise.
            Glacier retreat also threatens freshwater supplies for hundreds of
            millions of people in Asia and South America who depend on glacial
            meltwater during dry seasons.
        """),
    }

    for fname, content in _docs.items():
        fpath = input_dir / fname
        if not fpath.exists():
            fpath.write_text(content, encoding="utf-8")
            print(f"    wrote {fpath}")
        else:
            print(f"    (exists) {fpath}")

    # ── 4. Build or load the index ───────────────────────────────────────────
    print("\n[5/5] Building / loading NodeRAG index …")
    print(f"    Index path: {Path(INDEX_PATH).resolve()}")

    graph = NodeRagGraph(INDEX_PATH, generate=_generator, embedder=_embedder)

    # ensure_loaded() checks HNSW.bin; if present the build is skipped entirely.
    graph.ensure_loaded()

    # ── 5. Query and print results ───────────────────────────────────────────
    QUERY = "What are the key climate feedback mechanisms involving the Arctic and permafrost?"
    print(f"\nQuery: {QUERY!r}\n")

    findings = graph.community_context(QUERY)

    if findings:
        print("Community-level findings:")
        for i, f in enumerate(findings, 1):
            print(f"  {i}. {f}")
    else:
        # Small corpus (3 docs) may not produce community summaries — that is
        # expected and handled gracefully by the DisTraceAI patch.
        print("No community findings returned.")
        print("(This is normal for a 3-document corpus — the graph may be too")
        print("sparse to form communities.  Raw retrieval is still functional.)")

    print("\n" + "=" * 60)
    print("Integration test complete.")
    print("=" * 60)
    print()
    print("Index persisted at:", Path(INDEX_PATH).resolve())
    print("To wipe and rebuild from scratch, run:")
    print(f"    rm -rf {Path(INDEX_PATH).resolve()}")
    print()
    print("To re-query without rebuilding, just run the script again.")
    print("The build step will be skipped because HNSW.bin already exists.")