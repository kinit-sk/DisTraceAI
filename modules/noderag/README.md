# NodeRAG local-model patch (llama.cpp)

NodeRAG ships only commercial model integrations (OpenAI / Gemini). The model
keys in its docs are outdated, and it has no first-class way to drive a local
model. This patch adds a **llama.cpp** integration so NodeRAG can build and query
indexes with local GGUF models (e.g. Gemma 4 26B-A4B), with no API keys and no
external server — the model runs in-process via `llama-cpp-python`.

Pinned upstream commit: `f77dd6adb34cf4dda1d88b30b2bf0b17d14480a9`

## What the patch changes (5 files, MIT-licensed upstream)

- **`NodeRAG/LLM/LLM.py`**
  - Makes the `openai` / `google` exception imports optional, so the module
    loads with neither commercial SDK installed.
  - Adds `Llama_cpp` — a chat-completion class mirroring the `OPENAI` class API
    (`predict`, `predict_async`, `stream_chat`, schema-constrained JSON via
    `response_format`). Blocking llama.cpp calls run off the event loop
    (`asyncio.to_thread`) and are serialised with a lock, since a llama.cpp
    context is not concurrency-safe.
  - Adds `Llama_cpp_Embedding` — embeddings via `llama-cpp-python`
    (`embedding=True`), accepting a string or list and returning one vector per
    input, mirroring `OpenAI_Embedding`.
- **`NodeRAG/LLM/LLM_route.py`** — routes `service_provider: llama_cpp` and
  `llama_cpp_embedding` to the new classes.
- **`NodeRAG/build/pipeline/attribute_generation.py`** — guards the node-importance step (K-core / average-degree / betweenness) against **empty or tiny graphs**. A very small corpus produces a near-empty graph; upstream then divides by zero (`avarege_degree`) and calls `log(0)`. The guards return no important nodes instead of crashing, so the build completes.
- **`NodeRAG/utils/token_utils.py`** — `get_token_counter` previously *raised*
  for any non-GPT/Gemini model name; it now falls back to a cl100k (gpt-4o)
  tokenizer for local models. Token counts here only drive chunk sizing, so the
  approximation is harmless.

## Install

```bash
conda activate distrace            # the env DisTraceAI imports NodeRAG from
./install_noderag_local.sh         # clones, applies the patch, installs editable
```

The script clones NodeRAG at the pinned commit, applies `noderag_llamacpp.patch`
(idempotently), builds `llama-cpp-python` (CUDA if `nvcc` is present, else CPU),
and `pip install -e .` into the active conda env.

## Configure

Set the model sections in your NodeRAG config to the local providers — see
`Node_config.llama_cpp.example.yaml`. Key points:

- `model_name` / `embedding_model_name` may be a **local `.gguf` path** or a
  **HuggingFace repo id**; for a repo, `gguf_file` selects the file (glob OK).
- Set `config.dim` to your **embedding model's** output dimension (NOT OpenAI's
  1536) — e.g. 768 for nomic-embed, 2560 for Qwen3-Embedding-4B.
- Keep `rate_limit` low (llama.cpp generation is serialised in-process).

## Relationship to DisTraceAI's in-process adapter

DisTraceAI's `core/retrieval/noderag.py` already drives NodeRAG with the models
DisTraceAI has loaded, by swapping adapter clients into a keyless `NodeConfig` at
runtime — useful when NodeRAG is embedded in the pipeline. This source patch is
the complementary, permanent option: it makes NodeRAG natively local-capable via
its own `service_provider: llama_cpp` config, so NodeRAG's own CLI/build/query
work standalone. Use whichever fits; they don't conflict.

- **`NodeRAG/build/pipeline/summary_generation.py`** — guards the community **Summary** stage. A degenerate graph (or a local model that emits no valid community summaries) leaves no `community_summary.jsonl`; upstream then crashes opening it. The patch treats a missing summary file as “no high-level elements” so the build completes and the index is usable.

## Corpus size matters

NodeRAG builds a knowledge graph from its input documents, so it needs enough text to be useful. Feeding it only a handful of short sub-narrative labels (e.g. PolyNarrative EN-CC has ~19 train sub-narratives) yields a near-empty graph: with the patch the build no longer crashes, but SpecFi-C's community conditioning will be sparse or empty (it then behaves like plain HyDE + cosine). For a meaningful SpecFi-C graph, broaden the corpus (more languages/domains) or feed richer text than the bare labels.
