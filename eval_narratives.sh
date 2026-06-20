#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "EvalNarratives"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.out

module load GCC/13.2.0
module load CUDA/12.4.0

eval "$(conda shell.bash hook)"
conda activate distrace
export VLLM_DEEP_GEMM_WARMUP=skip   # vLLM #41849: skip FP8 warmup (no deep_gemm; non-FP8 models)
export DISABLE_KERNEL_MAPPING=1     # transformers 5.12 + kernels 0.15 import-time skew

# Narrative retrieval benchmark.
# Metrics: Acc@1/3/5 + MAP, overall and per-language.
#
# Methods table:
#   dense      — pure cosine; pick representation with --nar-dense-repr
#                {subnar|article|canonized}. Three separate experiments.
#   bm25-rag   — BM25+dense RRF hybrid, no LLM. Strongest non-LLM baseline.
#   specfi-cs  — reproduced original SpecFi-CS (NodeRAG over article texts,
#                n=10 hypotheticals). Requires patched NodeRAG.
#   cspecfi    — our continuous variant (no NodeRAG, conditioned on sub-
#                narrative's own claims, n=10 hypotheticals).
#   context-1  — agentic multi-turn search harness.
#
# Embedder: Qwen3-Embedding-4B (matching SpecFi paper for reproducibility).
# Hypotheticals: 10 (matching paper's generate_hypotheticals n=10).

DISTRACE=$HOME/distrace

# --- three dense representations -------------------------------------------
for REPR in subnar article canonized; do
  python $DISTRACE/main.py \
    --eval narratives \
    --nar-detector both \
    --nar-extractor dense \
    --nar-dense-repr $REPR \
    --nar-embedder Qwen/Qwen3-Embedding-4B \
    --nar-eval-split test
done

# --- BM25-RAG (no LLM) ---------------------------------------------------
python $DISTRACE/main.py \
  --eval narratives \
  --nar-detector both \
  --nar-extractor bm25-rag \
  --nar-embedder Qwen/Qwen3-Embedding-4B \
  --nar-eval-split test

# --- reproduced SpecFi-CS baseline ----------------------------------------
python $DISTRACE/main.py \
  --eval narratives \
  --nar-detector both \
  --nar-extractor specfi-cs \
  --nar-embedder Qwen/Qwen3-Embedding-4B \
  --nar-generator qwen3.5-2b \
  --nar-precision awq4 \
  --nar-specfi-hypotheticals 10 \
  --nar-eval-split test

# --- our cSpecFi (no NodeRAG, claim-conditioned) ---------------------------
python $DISTRACE/main.py \
  --eval narratives \
  --nar-detector both \
  --nar-extractor cspecfi \
  --nar-embedder Qwen/Qwen3-Embedding-4B \
  --nar-generator qwen3.5-2b \
  --nar-precision awq4 \
  --nar-specfi-hypotheticals 10 \
  --nar-eval-split test

# --- Context-1 agentic harness --------------------------------------------
python $DISTRACE/main.py \
  --eval narratives \
  --nar-detector both \
  --nar-extractor context-1 \
  --nar-embedder Qwen/Qwen3-Embedding-4B \
  --nar-generator qwen3.5-2b \
  --nar-precision awq4 \
  --nar-context1-context-size 32768 \
  --nar-context1-max-turns 8 \
  --nar-context1-token-budget 8192 \
  --nar-eval-split test
