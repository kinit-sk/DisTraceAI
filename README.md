# DisTraceAI - EUDisinfoAtlas dataset
This repository contains the materials that allow reproducing the work introduced 
in the paper `A Hierarchical Framework for the Detection of Coordinated Disinformation
Campaigns`.

DisTraceAI is an automated, end-to-end pipeline that transforms raw multilingual 
news articles into structured intelligence about coordinated disinformation campaigns. 
It operates at four hierarchical levels: **Articles → Sub-Narratives → Narratives → 
Campaigns**, with each level building on the one below and persisting independently 
to a JSON-based knowledge base.

# Setup
## 0. Prerequisites
- A machine with an NVIDIA GPU (H200 / A100 / RTX 3090). The H200 (Driver 580 /
  CUDA 13.0) is the primary environment. All generators run bf16, so the GPU must
  have enough VRAM for the chosen model (the 12B and Context-1 want a large GPU).
- conda installed and initialised (`conda init bash`, then restart your shell).
- The datasets placed under `data/` (MultiCW, FakeCTI, PolyNarrative, MultiClaim,
  MassiveSumm) — not shipped in the repo.

## 1. Create the environment (ONCE)
From a shell where `conda` works (e.g. base), run:

    bash setup.sh

It does the whole install end-to-end: creates and activates the `distrace` env,
installs vLLM + the pipeline requirements + NodeRAG (patched local clone), removes
the `kernels` package, and finishes with a verification pass that reinstalls
anything missing and prints an OK/MISSING line per critical package. It does NOT
abort partway on a single failed install — every stage runs and gaps are reported
at the end (this is deliberate: a network blip on one wheel must not leave the
rest uninstalled). If the final report shows NodeRAG still MISSING, re-run just
that step in the active env:

    conda activate distrace
    bash modules/noderag/install_noderag_local.sh

If you ever accidentally installed into system Python, run (with NO env active):

    bash cleanup_system_python.sh

`setup.sh` also installs a conda CUDA toolkit (13.3) and, if the system driver's
`libcuda.so` is present under `/usr/lib/x86_64-linux-gnu`, points the loader and
linker at it (needed for FlashInfer). Those exports apply only to the shell that
ran setup — the SLURM step scripts re-export them, but for interactive runs add
this to your shell rc:

    export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

## 2. Set HF_TOKEN (recommended)
Models download from HuggingFace on first use. Unauthenticated downloads are
rate-limited and can drop mid-transfer (you'll see "Server disconnected" /
"Can't load the configuration of <repo>"). Export a token to avoid this:

    export HF_TOKEN=hf_...         # also add to your shell rc

The loader retries transient Hub disconnects automatically, but a token removes
the rate limit that causes them. If a download was interrupted, a stale partial
cache can wedge it — clearing it forces a clean re-download:

    rm -rf ~/.cache/huggingface/hub

## 3. Run the pipeline (TUI)
    conda activate distrace
    python main.py                # interactive menu

Menu: 1 Claim detection · 2 Canonization · 3 Veracity · 4 Sub-narratives ·
5 Narratives · 6 Campaigns. Steps 1–5 have Evaluation / Generate; step 6 has
Verify / Deep verify / Evaluation / Generate Dataset.

All generators load bf16 weights straight from HuggingFace on first use (cached
afterwards).

## 4. Run a single step headless (SLURM or shell)
Each step has a script that activates the env and sets the needed env vars:

    sbatch gen_claim_canonization.sh      # on SLURM
    bash   gen_claim_canonization.sh      # or directly

Scripts: gen_/eval_ claim_detection, claim_canonization, claim_extract,
sub_narratives, narratives, campaigns, claim_veracity; gen_dataset;
eval_narratives_benchmark.

## Troubleshooting
- **SpecFi-CS/CCS crash in the NodeRAG "Attribute pipeline" with
  `TypeError: sequence item 0: expected str instance, dict found`** — the
  in-process LLM adapter must return a plain string for free-text steps
  (attribute/summary generation) and a dict only when NodeRAG passes a
  `response_format`. If a model's attribute text starts with `{`/`[` it must NOT
  be auto-parsed to a dict, or NodeRAG stores it as `raw_context` and hashing it
  via `genid(["".join([...])])` fails. This is handled in
  `core/hierarchy/noderag.py:_run` (structured output only when requested).
- **vLLM startup crashes in `flashinfer/jit` with "Ninja build failed" / nvcc
  `CalledProcessError`** — vLLM's sampler tries to JIT-compile a FlashInfer CUDA
  kernel via nvcc/ninja, and a fresh env's toolchain can fail that build. The
  pipeline disables it by default (`VLLM_USE_FLASHINFER_SAMPLER=0` → PyTorch-native
  sampling, set in main.py and the step scripts), so you shouldn't hit this. If you
  do (e.g. running vLLM directly), export `VLLM_USE_FLASHINFER_SAMPLER=0` and
  `rm -rf ~/.cache/flashinfer`. There's no accuracy cost — it's only a sampler
  speed optimization, irrelevant for these small eval workloads.
- **"No module named 'NodeRAG'" / "No module named 'rank_bm25'" / other missing
  deps** — usually means `pip install -r requirements.txt` aborted partway during
  setup, so packages after the failing line never installed, and the NodeRAG step
  never ran. Re-run setup in the active env, or just the two install steps:
  `pip install -r requirements.txt` then
  `bash modules/noderag/install_noderag_local.sh`. The NodeRAG script now verifies
  `import NodeRAG` at the end and fails loudly if it didn't land. NodeRAG is NOT a
  PyPI package here — it's a patched local clone, so don't `pip install NodeRAG`.
- **NodeRAG install fails: "No matching distribution found for hnswlib-noderag==0.8.2"**
  — NodeRAG pins an unpublished package, but its code falls back to plain
  `hnswlib`. The install script handles this (strips the bogus pin, installs
  `hnswlib`/`pandas`/`scipy`/`backoff`, then `pip install -e . --no-deps`). To
  unblock an already-cloned tree by hand:
  `python -m pip install hnswlib pandas scipy backoff` then
  `cd modules/noderag/NodeRAG_local && python -m pip install -e . --no-deps`.
  If `import NodeRAG` then names another missing module, `pip install` it (the
  package under-declares a few deps) and retry.
- **"Free memory on device ... is less than desired GPU memory utilization"** —
  the embedder and generator share one GPU (sub-narratives, narratives, veracity
  load the embedder first and keep it resident). vLLM's `gpu_memory_utilization`
  is a fraction of TOTAL VRAM, so the two must sum to ≤ 1.0. Defaults are
  embedder 0.30 + generator 0.60 = 0.90 (10% buffer). If you still hit this,
  lower one: `DISTRACE_GEN_GPU_UTIL=0.5` or `DISTRACE_EMBED_GPU_UTIL=0.2`. On a
  single-model step with no embedder resident you can raise the generator back up
  with `DISTRACE_GEN_GPU_UTIL=0.9`. (Context length is not the cause here — this
  fails at weight reservation, before the KV cache.)
- **"Can't load the configuration of <repo>" / "Server disconnected"** — transient
  HF download failure. Set `HF_TOKEN`, retry; if it persists, `rm -rf
  ~/.cache/huggingface/hub` and retry.
- **FlashInfer / libcuda load errors after a driver or toolkit change** — clear
  the JIT cache: `rm -rf ~/.cache/flashinfer`, and make sure
  `LD_LIBRARY_PATH` includes the dir holding `libcuda.so` (usually
  `/usr/lib/x86_64-linux-gnu`).
- **gemma4-12b fails to load** — the 12B is an encoder-free multimodal model
  whose native vLLM support is newer than 0.22.1; it may fall back to the generic
  Transformers path and crash under torch.compile. The five smaller models
  (qwen3.5-2b/4b/9b, gemma4-e2b/e4b) use supported architectures.

## Environment workarounds (set automatically)
These are exported by main.py and the step scripts; you don't need to set them
manually, but they explain the moving parts of the current dependency stack:
- `VLLM_DEEP_GEMM_WARMUP=skip` — vLLM 0.22 Hopper FP8-warmup crash (issue #41849).
- `DISABLE_KERNEL_MAPPING=1` — transformers 5.12 + kernels 0.15 import skew
  (belt-and-braces; setup.sh also uninstalls the `kernels` package, the real fix).
- `LD_LIBRARY_PATH` (libcuda dir) — FlashInfer needs the driver stub at JIT time.
- `HF_HUB_DOWNLOAD_TIMEOUT`, `HF_HUB_ENABLE_HF_TRANSFER` — more robust Hub
  downloads (the latter only if `hf_transfer` is installed).
- `HF_TOKEN` — set this yourself to lift HF rate limits (recommended).
