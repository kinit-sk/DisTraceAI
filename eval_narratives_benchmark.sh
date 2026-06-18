#!/bin/bash
#SBATCH --account=p1605-25-3
#SBATCH -J "NarBenchmark"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH -o stdout.%J.out
#SBATCH -e stderr.%J.out

# ---------------------------------------------------------------------------
# Narrative extraction Evaluation — FULL benchmark of every retrieval method.
#
# Uses the `all` option for --nar-extractor, which runs every method in one
# process and prints a side-by-side summary table at the end (best value per
# metric highlighted, plus per-detector MAP spread statistics). The console
# output is saved as a structured HTML report at:
#
#     results/narratives/<dataset>/<detector>__all.html
#
# where <dataset> reflects the selected domain (polynarrative, or
# polynarrative-CC / polynarrative-URW for a single-domain subset).
#
# Methods covered by `all`:
#   dense (subnar repr), bm25-rag, specfi-cs, specfi-ccs, cspecfi, context-1
# Metrics: Acc@1 / Acc@3 / Acc@5 + MAP, overall and per-language.
#
# NOTE on GPU use: SpecFi-CS / SpecFi-CCS build a NodeRAG graph. The build
# auto-sizes a pool of GPU worker contexts to fill spare VRAM (parallel build,
# torn down afterwards). Pool workers are GPU-only — if VRAM is tight the pool
# simply uses fewer workers rather than silently falling back to CPU. To cap
# the pool manually (e.g. when sharing a node) set DISTRACE_NODERAG_WORKERS=N.
# ---------------------------------------------------------------------------

module load GCC/13.2.0
module load CUDA/12.4.0

eval "$(conda shell.bash hook)"
conda activate distrace

DISTRACE=$HOME/distrace

# --- Tunables (override on the command line: VAR=value sbatch ...) ----------
DETECTOR=${DETECTOR:-both}                       # both | models/xlm-multicw | models/mdb-multicw
EMBEDDER=${EMBEDDER:-Qwen/Qwen3-Embedding-4B}    # SpecFi-paper default
GENERATOR=${GENERATOR:-qwen3.5-2b}               # HyDE / NodeRAG generator
QUANT=${QUANT:-Q4_K_M}
HYPOTHETICALS=${HYPOTHETICALS:-10}               # matches the paper's n=10
SPLIT=${SPLIT:-test}                             # dev | test
DOMAIN=${DOMAIN:-all}                            # all | CC | URW

echo "[benchmark] detector=$DETECTOR domain=$DOMAIN split=$SPLIT"
echo "[benchmark] embedder=$EMBEDDER generator=$GENERATOR ($QUANT) hypotheticals=$HYPOTHETICALS"

python "$DISTRACE/main.py" \
  --eval narratives \
  --nar-extractor all \
  --nar-detector "$DETECTOR" \
  --nar-embedder "$EMBEDDER" \
  --nar-generator "$GENERATOR" \
  --nar-quantization "$QUANT" \
  --nar-specfi-hypotheticals "$HYPOTHETICALS" \
  --nar-context1-context-size 32768 \
  --nar-context1-max-turns 8 \
  --nar-context1-token-budget 8192 \
  --nar-eval-split "$SPLIT" \
  --nar-eval-domain "$DOMAIN"

echo "[benchmark] done — see results/narratives/ for the HTML report."
