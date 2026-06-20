#!/usr/bin/env bash
# install_noderag_local.sh
# Clone the pinned NodeRAG, apply the DisTraceAI COMPATIBILITY patch (local-model
# token counting + small-graph guards — NOT a model provider), and install it
# editable into the active conda env.
#
# NodeRAG's model routing is left STOCK (SpecFi-faithful): the pipeline injects
# its own in-process vLLM clients at runtime, so NodeRAG needs no provider of its
# own and llama-cpp is not involved anywhere.
#
# Usage:
#   conda activate distrace
#   ./install_noderag_local.sh [SRC_DIR]
set -euo pipefail

PIN="f77dd6adb34cf4dda1d88b30b2bf0b17d14480a9"
REPO="https://github.com/Terry-Xu-666/NodeRAG.git"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$HERE/noderag_compat.patch"
SRC_DIR="${1:-$HERE/NodeRAG_local}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: no active conda env. Run 'conda activate distrace' first." >&2
  exit 1
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "distrace" ]]; then
  echo "ERROR: the 'distrace' env is not active (got '${CONDA_DEFAULT_ENV:-none}')." >&2
  echo "       Run 'conda activate distrace' first so NodeRAG installs into it." >&2
  exit 1
fi
echo ">> Target conda env: $CONDA_PREFIX"
echo ">> NodeRAG source dir: $SRC_DIR"

# 1) clone + reset to a pristine pin so the patch always applies cleanly
if [[ ! -d "$SRC_DIR/.git" ]]; then
  echo ">> Cloning NodeRAG from $REPO …"
  git clone "$REPO" "$SRC_DIR"
fi
cd "$SRC_DIR"
git fetch --all --tags >/dev/null 2>&1 || true
git reset --hard "$PIN"
git clean -fd >/dev/null 2>&1 || true
echo ">> Reset NodeRAG to pristine @ $PIN"

# 2) apply the compatibility patch (clean tree -> always a fresh forward apply).
#    If it's already applied (re-run), skip rather than abort.
if git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH"
  echo ">> Applied DisTraceAI compat patch (token_utils local-model fallback +"
  echo "   small-graph guards in attribute/summary/relationship generation)."
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
  echo ">> Compat patch already applied — skipping."
else
  echo "ERROR: the compat patch does not apply to this NodeRAG tree." >&2
  echo "       The pinned commit may have changed. Inspect $PATCH vs $SRC_DIR." >&2
  exit 1
fi

# 3) install NodeRAG's dependencies, then the package itself.
#
# NodeRAG's pyproject/requirements hard-pin `hnswlib-noderag==0.8.2`, an
# unpublished fork ("No matching distribution found"), and its utils/HNSW.py does
# a BARE `import hnswlib_noderag` with no fallback. NodeRAG's exact version pins
# also fight vLLM's stack. So we:
#   (a) install its real deps from requirements.txt, UNPINNED, so pip can't
#       downgrade what vLLM/distrace already installed (collision-free);
#   (b) install plain hnswlib + a shim module named `hnswlib_noderag` that
#       re-exports it, satisfying the bare import without the unpublished fork;
#   (c) install the package editable with --no-deps so the broken pin in
#       pyproject can't re-trigger the failed resolution.
# `python -m pip` (not bare `pip`) guarantees we target the SAME interpreter the
# pipeline runs with — a bare `pip` on PATH can belong to a different env and is
# a common cause of "installed but not importable".
echo ">> Installing NodeRAG dependencies (read robustly, UNPINNED to avoid"
echo "   colliding with the vLLM stack; bogus hnswlib-noderag pin dropped) …"
# NodeRAG's requirements.txt is often a non-UTF-8/binary-looking file (grep
# reports "binary file matches"), AND it hard-pins exact versions that fight
# vLLM's dependency stack. So we read it robustly in Python (any encoding),
# DROP the version pins (keep only package names → pip won't downgrade what
# vLLM/distrace already installed), and drop the unpublished hnswlib-noderag
# pin (NodeRAG's code falls back to plain hnswlib). This installs NodeRAG's
# real deps — rich, networkx, tiktoken, aiohttp, … — in one shot, unpinned.
python - <<'PYREQS'
import re, sys, subprocess, os
names = []
path = "requirements.txt"
if os.path.exists(path):
    raw = open(path, "rb").read().replace(b"\x00", b"")      # strip UTF-16 nulls
    text = raw.decode("latin-1", "ignore")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue
        # take the bare package name (drop version pins / markers / extras)
        m = re.match(r"[A-Za-z0-9_.\-]+", line)
        if not m:
            continue
        pkg = m.group(0)
        low = pkg.lower().replace("_", "-")
        if low in ("hnswlib-noderag",):                       # unpublished; use plain hnswlib
            continue
        names.append(pkg)
# de-dup, keep order
seen, uniq = set(), []
for n in names:
    if n.lower() not in seen:
        seen.add(n.lower()); uniq.append(n)
if uniq:
    print("[noderag] installing (unpinned):", ", ".join(uniq))
    subprocess.call([sys.executable, "-m", "pip", "install", *uniq])
else:
    print("[noderag] could not read requirements.txt — relying on explicit deps below")
PYREQS
# plain hnswlib + ruamel.yaml (NodeRAG needs ruamel.yaml but doesn't list it in
# requirements.txt), then a shim so NodeRAG's bare `import hnswlib_noderag` works.
python -m pip install hnswlib "ruamel.yaml>=0.18.10"

# NodeRAG's utils/HNSW.py does a BARE `import hnswlib_noderag` (no fallback), and
# hnswlib-noderag is an unpublished fork — so installing plain hnswlib alone does
# NOT satisfy that import name. Write a tiny shim module named `hnswlib_noderag`
# into the env's site-packages that re-exports plain hnswlib (API-compatible for
# the standard HNSW Index calls NodeRAG makes). This resolves the import without
# patching NodeRAG's source or needing the unpublished package.
python - <<'PYSHIM'
import os, sysconfig
try:
    import hnswlib  # noqa: F401  (must exist for the shim to re-export)
except Exception as e:
    print("[noderag] WARNING: plain hnswlib not importable, shim may be incomplete:", e)
target = sysconfig.get_paths()["purelib"]
shim = os.path.join(target, "hnswlib_noderag.py")
with open(shim, "w") as f:
    f.write(
        "# Auto-generated by install_noderag_local.sh.\n"
        "# NodeRAG hard-imports `hnswlib_noderag`, an unpublished fork of hnswlib.\n"
        "# Plain hnswlib is API-compatible for NodeRAG's HNSW index usage, so we\n"
        "# re-export it under this name.\n"
        "import hnswlib as _h\n"
        "globals().update({k: getattr(_h, k) for k in dir(_h) if not k.startswith('__')})\n"
    )
print("[noderag] wrote hnswlib_noderag shim ->", shim)
PYSHIM

echo ">> python -m pip install -e . --no-deps (package only; deps handled above) …"
python -m pip install -e . --no-deps

# 3b) verification LOOP: NodeRAG under-declares some imports, and `import NodeRAG`
#     fails on the FIRST missing one. Import it, and if it fails with a missing
#     module, pip-install that module (UNPINNED) and retry — up to a few rounds —
#     so we don't play one-at-a-time whack-a-mole by hand.
echo ">> Resolving any remaining NodeRAG runtime imports …"
python - <<'PYHEAL'
import importlib, subprocess, sys, re
MAXROUNDS = 8
# import-name -> pip-name when they differ
ALIAS = {"yaml": "pyyaml", "sklearn": "scikit-learn", "PIL": "pillow",
         "bs4": "beautifulsoup4", "google": "google-api-core",
         "ruamel": "ruamel.yaml"}
# names we must NOT try to pip-install: unpublished forks / shimmed modules.
# `hnswlib_noderag` is provided by a shim above; if it's still "missing" here the
# shim didn't land — pip-installing it would only spam errors (it's not on PyPI).
SKIP = {"hnswlib_noderag"}
for _ in range(MAXROUNDS):
    try:
        importlib.invalidate_caches()
        importlib.import_module("NodeRAG")
        print("[noderag] import NodeRAG OK")
        break
    except ModuleNotFoundError as e:
        mod = (e.name or "").split(".")[0]
        if not mod:
            print("[noderag] import failed with no module name:", e, file=sys.stderr); break
        if mod in SKIP:
            print(f"[noderag] '{mod}' is not on PyPI and should be provided by a shim — "
                  "not pip-installing. Check the shim step above.", file=sys.stderr)
            break
        pip_name = ALIAS.get(mod, mod)
        print(f"[noderag] missing '{mod}' → pip install {pip_name}")
        subprocess.call([sys.executable, "-m", "pip", "install", pip_name])
    except Exception as e:
        # a non-import error means the package itself loads; stop trying to heal.
        print("[noderag] NodeRAG import reached a non-missing-module error "
              f"(treating as importable): {type(e).__name__}: {e}")
        break
else:
    print("[noderag] WARNING: still resolving imports after %d rounds" % MAXROUNDS,
          file=sys.stderr)
PYHEAL

# 4) VERIFY the package is importable in THIS env — the whole point. If this
#    fails, the install silently didn't take (dependency resolution error, wrong
#    env, etc.) and the pipeline's SpecFi-C steps would crash later with a
#    confusing "No module named 'NodeRAG'".
cd /  # leave the source dir so we test the installed package, not the cwd
if python -c "import NodeRAG; from NodeRAG import NodeSearch" 2>/dev/null; then
  echo ""
  echo ">> Done. NodeRAG installed and importable (stock routing + compat patch)."
  python -c "import NodeRAG, os; print('   NodeRAG at:', os.path.dirname(NodeRAG.__file__))"
  echo ">> The pipeline drives NodeRAG via in-process vLLM clients at runtime."
else
  echo "" >&2
  echo "ERROR: NodeRAG installed but 'import NodeRAG' fails in this env." >&2
  echo "       The real ImportError is printed below. If it's a missing module" >&2
  echo "       (NodeRAG under-declares some deps and we install it --no-deps)," >&2
  echo "       'python -m pip install <that-module>' and re-run this script." >&2
  echo "       Otherwise confirm you are in the 'distrace' env. Diagnostics:" >&2
  echo "         which python; python -c 'import sys; print(sys.executable)'" >&2
  python -c "import NodeRAG" || true   # surface the real ImportError
  exit 1
fi
