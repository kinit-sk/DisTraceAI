#!/usr/bin/env bash
# DisTraceAI launcher — reads the active backend from ./config.json and
# `conda activate`s the matching env, then drops the user into a ready shell
# (or runs the command they passed as arguments).
#
# Usage:
#     source ./activate_distrace.sh                 # activate, leave you in shell
#     ./activate_distrace.sh python main.py         # activate + run command
#     ./activate_distrace.sh --force vllm           # override backend selection
#
# Why source? `conda activate` only persists in the calling shell when the
# script is sourced. Running it without `source` is fine if you only need to
# launch a one-off command (last invocation form above) — the activation is
# scoped to the subshell where the command runs.
set -o pipefail

VLLM_ENV="distrace-vllm"
LLAMA_ENV="distrace-llama"
CFG_PATH="$(dirname "${BASH_SOURCE[0]:-$0}")/config.json"

# ---- read llm_backend from config.json (fallback to env var, then vllm) ----
OVERRIDE=""
if [[ "${1:-}" == "--force" && -n "${2:-}" ]]; then
  OVERRIDE="$2"
  shift 2
fi

_read_backend() {
  if [[ -n "$OVERRIDE" ]]; then
    echo "$OVERRIDE"; return
  fi
  if [[ -f "$CFG_PATH" ]] && command -v python >/dev/null 2>&1; then
    python - "$CFG_PATH" <<'PYREAD' 2>/dev/null
import json, sys
try:
    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        print(json.load(fh).get("llm_backend", ""))
except Exception:
    pass
PYREAD
    return
  fi
  echo "${DISTRACE_LLM_BACKEND:-}"
}

BACKEND="$(_read_backend)"
case "$BACKEND" in
  vllm)              ENV_NAME="$VLLM_ENV"  ;;
  llama-cpp|llama)   ENV_NAME="$LLAMA_ENV" ;;
  "")
    echo "[activate_distrace] could not determine backend from $CFG_PATH; defaulting to vllm." >&2
    BACKEND="vllm"; ENV_NAME="$VLLM_ENV"
    ;;
  *)
    echo "[activate_distrace] unknown llm_backend '$BACKEND' in $CFG_PATH." >&2
    echo "                    Expected 'vllm' or 'llama-cpp'." >&2
    return 1 2>/dev/null || exit 1
    ;;
esac

# ---- activate ----
if ! command -v conda >/dev/null 2>&1; then
  echo "[activate_distrace] conda not on PATH. Run 'conda init bash' first." >&2
  return 1 2>/dev/null || exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
if [[ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]]; then
  echo "[activate_distrace] failed to activate '$ENV_NAME'." >&2
  echo "                    Run 'bash setup.sh' to (re)create it." >&2
  return 1 2>/dev/null || exit 1
fi
echo "[activate_distrace] backend=$BACKEND  env=$ENV_NAME  ($(python -c 'import sys; print(sys.executable)'))"

# Mirror the choice into the process env so `core.models` doesn't have to
# re-read config.json for every subprocess Python imports.
export DISTRACE_LLM_BACKEND="$BACKEND"

# ---- if extra args were passed, exec them; otherwise leave the user in shell.
if [[ $# -gt 0 ]]; then
  exec "$@"
fi
