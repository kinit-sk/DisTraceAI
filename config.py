"""Single-source configuration.

One settings object, edited by either the TUI or the CLI. Each field carries its
own label / description / choice-list metadata (the single source the TUI reads),
plus get/set/cycle/lock helpers so the editor logic stays testable. CLI flags
override the saved file for the run and are surfaced as locked (read-only) in the
TUI.

Environment-variable settings
------------------------------
Several pipeline behaviours are controlled by OS environment variables.  Rather
than requiring users to manage them in their shell, they are exposed as first-class
Config fields (prefix ``env_``).  On Config.load() the live environment
is read as the default; on save() the values are written to the JSON and applied
back to os.environ so that any subsequently loaded backend module sees them.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import ClassVar

CONFIG_PATH = Path(__file__).parent / "config.json"


def _f(default, label, desc, choices=None):
    return field(default=default, metadata={"label": label, "desc": desc, "choices": choices})


@dataclass
class Config:

    # ------------------------------------------------------------------ #
    # LLM Backend selection
    # ------------------------------------------------------------------ #
    llm_backend: str = _f(
        "vllm",
        "LLM inference backend",
        "Backend used for every generator and embedder call. "
        "'vllm' → highest throughput on H100/H200/A100 GPUs (BF16 only, "
        "CUDA 13.x). 'llama-cpp' → broader hardware support including "
        "older toolchains (uses Q8_0 GGUF quants, CUDA 12.x). The two "
        "live in separate conda envs (distrace-vllm / distrace-llama); "
        "switching here without reactivating the right env will fail at "
        "first import. Use ./activate_distrace.sh to keep both in sync.",
        choices=["vllm", "llama-cpp"],
    )

    # ------------------------------------------------------------------ #
    # Claim detection (step 1)
    # ------------------------------------------------------------------ #
    detector: str = _f(
        "models/mdb-multicw",
        "Check-worthiness classifier",
        "Fine-tuned check-worthiness classifier used in step 1. "
        "'models/mdb-multicw' (mDeBERTa) is the stronger F1 on MultiCW; "
        "'models/xlm-multicw' (XLM-R) is smaller / faster and broadly "
        "comparable. The same choice should flow through downstream steps "
        "(canon/subnar/nar/camp _detector) for a consistent pipeline.",
        choices=["models/xlm-multicw", "models/mdb-multicw"],
    )

    canon_detector: str = _f(
        "models/mdb-multicw",
        "Canonization source detector",
        "Which detector's KB output is canonized. Must match a detector "
        "that already produced check-worthy claims. 'both' runs canonization "
        "on the outputs of both detectors back-to-back (useful for "
        "ablation; doubles canonization cost).",
        choices=["models/xlm-multicw", "models/mdb-multicw", "both"],
    )

    canon_generator: str = _f(
        "gemma4-e4b",
        "Canonization generator",
        "LLM used to decontextualize claims and translate them into "
        "English. LARGER model → cleaner decontextualization and better "
        "translation fidelity for low-resource languages, slower. SMALLER "
        "→ faster, may leave residual context or mistranslate idioms.",
        choices=["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
                 "gemma4-e2b", "gemma4-e4b", "gemma4-12b"],
    )

    # NOTE: canon_precision / subnar_precision / nar_precision / camp_precision
    # were intentionally removed. Per plan §4, the generator precision is fixed
    # by the active backend: BF16 under vLLM, Q8_0 under llama-cpp. The
    # backend layer (core/llm_backends/llama_cpp.py) hard-codes Q8_0 as its
    # default quant; vLLM is BF16-only and silently ignores any quant arg.

    subnar_detector: str = _f(
        "models/mdb-multicw",
        "Sub-narrative source detector",
        "Which detector's canonized output feeds sub-narrative extraction. "
        "Must match a detector that already produced canonized claims. "
        "'both' processes both detectors' outputs in one session.",
        choices=["models/xlm-multicw", "models/mdb-multicw", "both"],
    )

    subnar_embedder: str = _f(
        "Qwen/Qwen3-Embedding-0.6B",
        "Sub-narrative embedder",
        "SentenceTransformer model that embeds canonized claims for the "
        "similarity-clustering step. LARGER model (Qwen3-Embedding-4B, "
        "multilingual-e5-large-instruct) → richer multilingual semantics, "
        "tighter clusters, more VRAM. SMALLER (0.6B) → faster / lighter "
        "but loses some cross-lingual nuance.",
        choices=["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B",
                 "intfloat/multilingual-e5-large-instruct"],
    )

    subnar_generator: str = _f(
        "gemma4-e4b",
        "Sub-narrative generator",
        "LLM that synthesizes a single sub-narrative central claim from "
        "each cluster of canonized claims. LARGER → more faithful and "
        "concise summaries, slower. SMALLER → faster but more likely to "
        "lose nuance in long clusters.",
        choices=["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
                 "gemma4-e2b", "gemma4-e4b", "gemma4-12b"],
    )

    subnar_min_similarity: float = _f(
        0.45,
        "Min claim similarity",
        "Cosine similarity threshold for assigning a canonized claim to the "
        "current sub-narrative. HIGHER → tighter clusters, more sub-narratives, "
        "each more specific; risk of fragmentation. LOWER → larger, looser "
        "clusters with broader themes; risk of merging unrelated claims. "
        "Range 0.0-1.0.",
    )

    subnar_min_claims: int = _f(
        2,
        "Min claims per sub-narrative",
        "A cluster must contain at least this many claims to be promoted to a "
        "sub-narrative; otherwise its claims are discarded. HIGHER → fewer, "
        "more strongly evidenced sub-narratives; rare topics get dropped. "
        "LOWER (down to 2) → captures niche topics but allows weakly evidenced "
        "sub-narratives.",
    )

    subnar_hypotheticals: int = _f(
        3,
        "HyDE hypotheticals",
        "Number of hypothetical sub-narrative descriptions generated per "
        "central claim during EVALUATION retrieval (HyDE style — each one is "
        "used as a query). HIGHER → richer retrieval recall and more stable "
        "voting at the cost of slower eval and more LLM calls. LOWER → faster "
        "eval but more variance per claim. Affects evaluation only, not "
        "Generate.",
    )

    # ------------------------------------------------------------------ #
    # Narrative extraction (step 5)
    # ------------------------------------------------------------------ #
    nar_detector: str = _f(
        "models/mdb-multicw",
        "Narrative source detector",
        "Which detector's sub-narrative chain feeds narrative extraction / "
        "retrieval. Must match a detector that already produced sub-"
        "narratives. 'both' processes both detectors' sub-narrative chains "
        "in one session.",
        choices=["models/xlm-multicw", "models/mdb-multicw", "both"],
    )

    nar_extractor: str = _f(
        "cspecfi",
        "Narrative retrieval method",
        "Retrieval method used by both narrative Eval and Generate. "
        "dense → simple embedding cosine over the corpus item selected by "
        "nar_dense_repr (fastest baseline). "
        "bm25-rag → BM25 + dense RRF hybrid, no LLM, strong non-LLM "
        "baseline. "
        "specfi-cs → static SpecFi over article texts via NodeRAG (paper "
        "reproduction). "
        "specfi-ccs → SpecFi over per-article canonized claims. "
        "cspecfi → continuous SpecFi variant conditioned on sub-narrative "
        "claims (no NodeRAG; recommended). "
        "context-1 → agentic multi-turn search harness (slowest, highest "
        "accuracy on diverse phrasings). "
        "all → EVAL ONLY: benchmark every method side-by-side.",
        choices=["dense", "bm25-rag", "specfi-cs", "specfi-ccs", "cspecfi",
                 "context-1", "all"],
    )

    nar_dense_repr: str = _f(
        "subnar",
        "Dense representation",
        "Read only when nar_extractor=dense. Which text represents each "
        "item in the corpus: 'article' (raw article text — broad but noisy), "
        "'canonized' (set of canonized claims — middle ground), "
        "'subnar' (sub-narrative central claim — concise, recommended for "
        "narrative-level retrieval).",
        choices=["article", "canonized", "subnar"],
    )

    nar_embedder: str = _f(
        "Qwen/Qwen3-Embedding-4B",
        "Narrative embedder",
        "SentenceTransformer model that embeds narrative queries and corpus "
        "items. LARGER model → better retrieval accuracy across phrasings "
        "and languages, more VRAM and slower indexing. SMALLER → faster / "
        "lighter, drops some retrieval quality.",
        choices=["Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-0.6B",
                 "intfloat/multilingual-e5-large-instruct"],
    )

    nar_generator: str = _f(
        "gemma4-e4b",
        "Narrative generator",
        "LLM that (a) synthesizes narrative central claims during Generate "
        "and (b) generates HyDE hypotheticals for specfi-cs / specfi-ccs / "
        "cspecfi / context-1. LARGER → more concise / faithful central "
        "claims and better HyDE coverage, slower. SMALLER → faster but "
        "more likely to produce generic phrasings.",
        choices=["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
                 "gemma4-e2b", "gemma4-e4b", "gemma4-12b"],
    )

    nar_assign_threshold: float = _f(
        0.55,
        "Narrative assign threshold",
        "Cosine score above which a sub-narrative is merged into the best-"
        "matching existing narrative. HIGHER → stricter merging, more "
        "narratives, each more specific. LOWER → narratives accrete more "
        "members and become broader (risk of merging unrelated angles). "
        "Range 0.0-1.0.",
    )

    nar_min_new_size: int = _f(
        2,
        "Min new-narrative size",
        "Minimum number of unassigned sub-narratives that must cluster "
        "together before a new narrative is created. HIGHER → only well-"
        "evidenced narratives are promoted; rare topics get dropped. LOWER "
        "(down to 2) → preserves niche topics but allows weakly evidenced "
        "narratives.",
    )

    nar_new_threshold: float = _f(
        0.55,
        "New-narrative similarity",
        "Cosine threshold the agglomerative grouper uses while clustering the "
        "unassigned-sub-narrative pool into new narratives. HIGHER → very "
        "tight clusters (few new narratives form on diverse corpora). LOWER "
        "→ more narratives form (risk of merging loosely related angles). "
        "Range 0.0-1.0.",
    )

    nar_clustering_linkage: str = _f(
        "average",
        "Agglomerative linkage",
        "Linkage criterion the agglomerative grouper uses to decide pool-"
        "cluster boundaries. 'average' → robust default, clusters by mean "
        "pairwise distance. 'single' → permissive (any close pair merges, "
        "chains form quickly). 'complete' → strict (every pair must be "
        "close, only very tight clusters form).",
        choices=["average", "single", "complete"],
    )

    nar_recluster_cadence: int = _f(
        0,
        "Re-cluster cadence",
        "Run a periodic re-clustering sweep over the unassigned-pool every N "
        "processed sub-narratives during Generate. 0 disables periodic sweeps "
        "(the grouper still re-clusters whenever a sub-narrative misses an "
        "existing narrative). HIGHER N → less expensive but stragglers can "
        "linger longer in the pool. LOWER N → faster cluster discovery at "
        "the cost of extra recompute.",
    )

    nar_specfi_hypotheticals: int = _f(
        10,
        "SpecFi hypotheticals",
        "Number of hypothetical texts generated per query for specfi-cs / "
        "specfi-ccs / cspecfi retrieval. HIGHER → broader query coverage, "
        "higher recall on diverse phrasings, at the cost of extra LLM calls "
        "per query. LOWER → faster but more sensitive to the exact wording of "
        "the seed claim.",
    )

    nar_context1_context_size: int = _f(
        32768,
        "Context-1 model context size",
        "llama.cpp context window when loading the Context-1 retrieval model. "
        "HIGHER → more evidence and longer turns fit per query at the cost of "
        "VRAM and per-token speed. LOWER → cheaper but the harness may "
        "truncate evidence on long retrieval chains. Context-1 is trained on "
        "128K; 32768 is a practical minimum.",
    )

    nar_context1_max_turns: int = _f(
        8,
        "Context-1 max turns",
        "Hard cap on agentic search turns per query when nar_extractor="
        "context-1. HIGHER → more chances to refine the query / explore "
        "different evidence; slower. LOWER → faster but the harness may "
        "stop before finding the best evidence.",
    )

    nar_context1_token_budget: int = _f(
        8192,
        "Context-1 evidence token budget",
        "Cap on the total tokens of retrieved cluster evidence the agentic "
        "harness accumulates before terminating. NOT the model context size "
        "(see nar_context1_context_size). HIGHER → richer evidence pools, "
        "better retrieval at cost of memory + decoding time. LOWER → "
        "faster but may truncate useful evidence.",
    )

    nar_eval_split: str = _f(
        "test",
        "Narrative eval query split",
        "PolyNarrative split that supplies the held-out query sub-narratives "
        "for the narrative-retrieval eval. The retrieval CORPUS is always "
        "built from train. 'test' → official held-out evaluation. 'dev' → "
        "use the smaller dev split for fast iteration; results are not "
        "directly comparable to published numbers.",
        choices=["dev", "test"],
    )

    nar_eval_domain: str = _f(
        "all",
        "Narrative eval domain",
        "Restrict the narrative-retrieval eval to a single PolyNarrative "
        "domain. 'all' → both domains (most directly comparable to the "
        "paper). 'CC' → Climate Change only. 'URW' → Ukraine-Russia War "
        "only. Filters apply to both the query split AND the train corpus, "
        "so accuracy figures are per-domain.",
        choices=["all", "CC", "URW"],
    )

    # ------------------------------------------------------------------ #
    # Claim veracity estimation (step 3)
    # ------------------------------------------------------------------ #
    # NOTE: ``ver_sources`` was removed. The evidence cascade is hard-coded
    # in ``core.eval.eval_claim_veracity`` and ``core.gen.gen_veracity`` to
    # multiclaim → wikipedia → web (in that order); the order is part of the
    # method, not a user choice.

    ver_generator: str = _f(
        "gemma4-e2b",
        "Veracity verdict generator",
        "LLM that synthesizes the True / False / Disputed verdict from "
        "gathered evidence. LARGER model → better long-evidence reasoning "
        "and higher per-verdict accuracy, slower. SMALLER → faster but "
        "more likely to commit to Disputed under conflicting evidence.",
        choices=["gemma4-e2b", "gemma4-e4b", "gemma4-12b",
                 "qwen3.5-2b", "qwen3.5-4b"],
    )

    ver_paraphrase_generator: str = _f(
        "gemma4-12b",
        "Paraphrase generator",
        "LLM that generates English-only meaning-preserving paraphrases of "
        "MultiClaim entries to build the veracity-eval test queries. "
        "LARGER → more diverse, more faithful paraphrases (the eval is "
        "harder but more realistic). SMALLER → faster paraphrase-cache "
        "build but paraphrases may drift in meaning. Output is cached to "
        "knowledge/veracity/ so changing this only costs LLM time on the "
        "first run.",
        choices=["gemma4-12b", "gemma4-e4b", "qwen3.5-9b"],
    )

    ver_max_turns: int = _f(
        6,
        "Veracity max turns",
        "Cap on agentic search turns per claim verification. HIGHER → more "
        "thorough evidence gathering and a better chance the harness reaches "
        "decisive evidence; slower. LOWER → faster but the harness may "
        "commit to a verdict before locating strong evidence.",
    )

    ver_token_budget: int = _f(
        4096,
        "Veracity evidence token budget",
        "Cap on evidence tokens the agentic harness accumulates per claim "
        "(NOT the model context size). HIGHER → more evidence reasoned over "
        "per verdict, slower decoding. LOWER → faster but earlier evidence "
        "wins by default.",
    )

    ver_confidence_threshold: float = _f(
        0.65,
        "Veracity cascade confidence threshold",
        "Confidence floor for accepting a True/False verdict from one stage "
        "of the multiclaim → wikipedia → web cascade. If the verdict is "
        "Disputed OR below this confidence, the next stage is tried. "
        "HIGHER → more queries reach Wikipedia / Web (broader evidence, "
        "slower, more online dependence). LOWER → the cascade accepts "
        "shallow verdicts early (faster, but harder claims may be answered "
        "from weak MultiClaim evidence). Range 0.0-1.0.",
    )

    ver_n_paraphrases: int = _f(
        3,
        "Paraphrases per claim",
        "Number of paraphrase variants generated per MultiClaim entry to "
        "build the evaluation test queries. HIGHER → more robust per-claim "
        "estimates and richer test set; more LLM time at paraphrase-cache "
        "build. LOWER → faster but more variance per claim.",
    )

    ver_n_samples: int = _f(
        100,
        "MultiClaim sample size",
        "Target size of the balanced True/False evaluation test set. The "
        "pipeline first filters MultiClaim to True/False rows (Disputed "
        "dropped), then samples ver_n_samples // 2 from each class. HIGHER "
        "→ tighter per-class accuracy estimate; longer eval. LOWER → faster "
        "but per-class results have wider confidence intervals. Changing "
        "this value invalidates the paraphrase cache. 0 → use everything "
        "available.",
    )

    # NOTE: ``ver_multiclaim_text_col`` / ``ver_multiclaim_label_col`` were
    # removed. The published MultiClaim CSV uses ``claim`` for the claim text
    # and ``ratings`` for the verdict label (Python-repr list of votes). Those
    # are now hard-coded constants in ``core.eval.eval_claim_veracity`` and
    # ``core.gen.gen_veracity``; the schema does not change between runs.

    # ------------------------------------------------------------------ #
    # Campaigns extraction (step 6)
    # ------------------------------------------------------------------ #
    camp_detector: str = _f(
        "models/mdb-multicw",
        "Campaign source detector",
        "Which detector's narrative hierarchy feeds campaign extraction. "
        "Must match a detector that already produced narratives. 'both' "
        "processes both detectors' narrative chains in one session.",
        choices=["models/xlm-multicw", "models/mdb-multicw", "both"],
    )

    camp_extractor: str = _f(
        "dense",
        "Campaign retrieval method",
        "Retrieval method used by Campaigns Generate. Same trade-offs as "
        "nar_extractor: dense → cheapest, bm25-rag → strong non-LLM "
        "baseline, specfi-* → NodeRAG-based variants, cspecfi → continuous "
        "SpecFi (no NodeRAG), context-1 → agentic multi-turn (slowest, "
        "best on diverse phrasings).",
        choices=["dense", "bm25-rag", "specfi-cs", "specfi-ccs", "cspecfi", "context-1"],
    )

    camp_dense_repr: str = _f(
        "subnar",
        "Campaign dense representation",
        "Read only when camp_extractor=dense. Text used to represent each "
        "narrative in the corpus: 'article' (raw article text — noisy at "
        "this level), 'canonized' (set of canonized claims), 'subnar' "
        "(sub-narrative central claim — recommended for narrative → "
        "campaign retrieval).",
        choices=["article", "canonized", "subnar"],
    )

    camp_embedder: str = _f(
        "Qwen/Qwen3-Embedding-4B",
        "Campaign embedder",
        "SentenceTransformer model that embeds narrative central claims "
        "and campaign queries. LARGER → better retrieval across phrasings "
        "and languages, more VRAM. SMALLER → faster / lighter, drops some "
        "retrieval quality.",
        choices=["Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-0.6B",
                 "intfloat/multilingual-e5-large-instruct"],
    )

    camp_generator: str = _f(
        "gemma4-e4b",
        "Campaign generator",
        "LLM that synthesizes campaign central claims from member "
        "narratives, and (for specfi-* / context-1) generates HyDE "
        "hypotheticals. LARGER → more concise / faithful summaries, "
        "slower. SMALLER → faster but more generic phrasings.",
        choices=["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
                 "gemma4-e2b", "gemma4-e4b", "gemma4-12b"],
    )

    camp_assign_threshold: float = _f(
        0.50,
        "Campaign assign threshold",
        "Cosine score above which a narrative is merged into the best-"
        "matching existing campaign. HIGHER → stricter merging, more "
        "campaigns each on a tighter theme. LOWER → campaigns accrete more "
        "narratives and become broader (risk of merging loosely related "
        "narratives). Range 0.0-1.0.",
    )

    camp_min_new_size: int = _f(
        2,
        "Min new-campaign size",
        "Minimum number of unassigned narratives that must cluster together "
        "before a new campaign is created. HIGHER → only multi-narrative "
        "campaigns survive; single-narrative angles are dropped. LOWER "
        "(down to 2) → preserves emerging campaigns at the cost of more "
        "noise.",
    )

    camp_new_threshold: float = _f(
        0.50,
        "New-campaign similarity",
        "Cosine threshold the agglomerative grouper uses when clustering "
        "unassigned narratives into new campaigns. HIGHER → tight clusters, "
        "few campaigns form on diverse corpora. LOWER → more campaigns "
        "form but each may pool loosely related narratives. Range 0.0-1.0.",
    )

    camp_clustering_linkage: str = _f(
        "average",
        "Campaign linkage",
        "Linkage criterion the agglomerative grouper uses to decide pool-"
        "cluster boundaries at the campaign level. 'average' → robust "
        "default. 'single' → permissive (chains form quickly). 'complete' "
        "→ strict (only very tight clusters become campaigns).",
        choices=["average", "single", "complete"],
    )

    camp_recluster_cadence: int = _f(
        0,
        "Campaign recluster cadence",
        "Run a periodic re-clustering sweep over the unassigned-narrative "
        "pool every N processed narratives. 0 disables periodic sweeps "
        "(the grouper still re-clusters whenever a narrative misses every "
        "campaign). HIGHER N → less recompute, slower cluster discovery. "
        "LOWER N → faster discovery but extra work.",
    )

    camp_specfi_hypotheticals: int = _f(
        10,
        "Campaign SpecFi hypotheticals",
        "Number of hypothetical texts generated per query for specfi-cs / "
        "specfi-ccs / cspecfi campaign retrieval. HIGHER → richer recall "
        "across diverse narrative phrasings at the cost of more LLM calls. "
        "LOWER → faster but more sensitive to the exact wording of the "
        "seed narrative.",
    )

    camp_context1_max_turns: int = _f(
        8,
        "Campaign Context-1 max turns",
        "Cap on agentic search turns per query when camp_extractor="
        "context-1. HIGHER → more refinement of the search query, slower. "
        "LOWER → faster but may stop before finding decisive evidence.",
    )

    camp_context1_token_budget: int = _f(
        8192,
        "Campaign Context-1 token budget",
        "Cap on retrieved evidence tokens accumulated per query during "
        "context-1 campaign retrieval. HIGHER → richer evidence per "
        "campaign decision; slower decoding. LOWER → faster but the "
        "harness may truncate useful evidence.",
    )

    camp_coordination_threshold: float = _f(
        0.40,
        "Coordination threshold",
        "Coordination score above which a campaign is classified as a "
        "coordinated campaign (Information / Disinformation Campaign) "
        "rather than Organic Trend. HIGHER → stricter, only strongly "
        "synchronized campaigns are flagged as coordinated. LOWER → more "
        "campaigns are flagged as coordinated, including weakly "
        "synchronized ones. Range 0.0-1.0.",
    )

    camp_veracity_threshold: float = _f(
        0.45,
        "Veracity threshold",
        "Veracity score below which a coordinated campaign is classified "
        "as Disinformation Campaign rather than Information Campaign. "
        "HIGHER → more coordinated campaigns are tagged Disinformation "
        "(stricter information standard). LOWER → only campaigns with "
        "strongly negative veracity are tagged Disinformation. Campaigns "
        "with no veracity verdict default to Information Campaign. "
        "Range 0.0-1.0.",
    )

    # ------------------------------------------------------------------ #
    # Workflow parameters (Campaigns › Generate Dataset)
    # ------------------------------------------------------------------ #

    camp_sample_size: int = _f(
        100,
        "Target dataset sample size",
        "Cap on EUvsDisinfo articles processed during Generate Dataset. "
        "0 → process all available articles. HIGHER → broader coverage at "
        "the cost of compute (whole pipeline scales linearly in article "
        "count). LOWER → faster end-to-end runs but smaller, less stable "
        "extracted hierarchy.",
    )

    camp_re_extract: str = _f(
        "off",
        "Re-Extract campaign candidates",
        "Controls whether Generate Dataset re-runs the extraction pipeline "
        "when results/EUDisinfoAtlas/{campaigns,narratives,sub-narratives}"
        ".csv already exist. 'off' → reuse the existing CSVs (fast — only "
        "re-runs the optional veracity chain). 'on' → force a full "
        "re-extraction from scratch.",
        choices=["on", "off"],
    )

    camp_apply_coordination: str = _f(
        "on",
        "Apply coordination detection",
        "Whether to compute the four coordination signals (temporal burst, "
        "co-amplification, content reuse, cross-lingual co-occurrence) and "
        "combine them into a coordination score for each campaign. 'on' → "
        "campaigns are classified Organic / Information / Disinformation "
        "(see camp_coordination_threshold + camp_veracity_threshold). "
        "'off' → no coordination scores computed; campaigns are persisted "
        "without classification (faster, useful while iterating on "
        "extraction tuning).",
        choices=["on", "off"],
    )

    camp_veracity_mode: str = _f(
        "off",
        "Veracity estimation",
        "Whether Generate Dataset chains veracity verification after "
        "extraction. 'off' → no verification (fastest). 'verify_hierarchy' "
        "→ verify central claims at each level (sub-narrative → narrative "
        "→ campaign), propagating verdicts upward. 'deep_verify' → also "
        "verify every supporting claim under each central claim before "
        "propagation (slowest, most accurate veracity at the campaign "
        "level).",
        choices=["off", "verify_hierarchy", "deep_verify"],
    )

    camp_n1_weight: float = _f(
        0.30,
        "N1 burst weight",
        "Weight of the temporal-burst / time-synchrony signal in the "
        "coordination score (campaigns with publications clustered in "
        "time score higher on N1). HIGHER → coordination is driven more "
        "by synchronized timing. LOWER → timing matters less. Weights "
        "N1+N2+N3+N4 should sum to ~1.0.",
    )

    camp_n2_weight: float = _f(
        0.25,
        "N2 co-amplification weight",
        "Weight of the co-amplification signal in the coordination score "
        "(campaigns whose articles appear across the same outlets / "
        "publishers score higher on N2). HIGHER → coordination is driven "
        "more by shared distribution networks. LOWER → outlet overlap "
        "matters less.",
    )

    camp_n3_weight: float = _f(
        0.25,
        "N3 content-reuse weight",
        "Weight of the near-identical-content-reuse signal in the "
        "coordination score (campaigns whose articles share large textual "
        "fragments score higher on N3). HIGHER → coordination is driven "
        "more by text-level repetition. LOWER → exact reuse matters less.",
    )

    camp_n4_weight: float = _f(
        0.20,
        "N4 cross-lingual weight",
        "Weight of the cross-lingual co-occurrence signal in the "
        "coordination score (campaigns whose narratives surface "
        "simultaneously in multiple languages score higher on N4). "
        "HIGHER → coordination is driven more by multilingual reach. "
        "LOWER → multilingual coverage matters less.",
    )

    # ================================================================== #
    # Environment-variable settings
    # These are exposed in the TUI Settings menu and written to os.environ
    # on save so they take effect for the current session without a restart.
    # ================================================================== #

    # -- vLLM backend -------------------------------------------------- #
    env_vllm_deep_gemm_warmup: str = _f(
        "skip",
        "[vLLM] VLLM_DEEP_GEMM_WARMUP",
        "Set to 'skip' to bypass vLLM's DeepGEMM warmup on Hopper GPUs (H200, "
        "sm_90). Avoids a crash when the optional deep_gemm package is absent. "
        "Safe to set 'skip' for non-FP8 models. Corresponds to env var "
        "VLLM_DEEP_GEMM_WARMUP.",
        choices=["skip", ""],
    )

    env_vllm_use_deep_gemm: str = _f(
        "0",
        "[vLLM] VLLM_USE_DEEP_GEMM",
        "Enable (1) or disable (0) vLLM's DeepGEMM FP8 kernel. Leave at 0 for "
        "bf16 / non-FP8 models. Corresponds to env var VLLM_USE_DEEP_GEMM.",
        choices=["0", "1"],
    )

    env_use_flashinfer_sampler: str = _f(
        "0",
        "[vLLM] DISTRACE_USE_FLASHINFER_SAMPLER",
        "Enable (1) FlashInfer top-k/top-p sampling kernel for vLLM. Requires a "
        "working nvcc/ninja toolchain; leave at 0 if JIT compilation fails during "
        "engine startup. Corresponds to env var DISTRACE_USE_FLASHINFER_SAMPLER.",
        choices=["0", "1"],
    )

    env_disable_kernel_mapping: str = _f(
        "1",
        "[vLLM] DISABLE_KERNEL_MAPPING",
        "Disable transformers hub_kernels integration (transformers 5.12 + "
        "kernels 0.15 incompatibility). Keep at 1 unless you have resolved the "
        "LayerRepository version/revision issue. Corresponds to env var "
        "DISABLE_KERNEL_MAPPING.",
        choices=["1", "0"],
    )

    env_distrace_show_tqdm: str = _f(
        "0",
        "[vLLM] DISTRACE_SHOW_TQDM",
        "Show (1) or silence (0) vLLM / NodeRAG tqdm progress bars. DisTraceAI "
        "uses Rich for all progress output; tqdm bars from library internals are "
        "suppressed by default to avoid rendering conflicts. Corresponds to env "
        "var DISTRACE_SHOW_TQDM.",
        choices=["0", "1"],
    )

    env_distrace_gen_gpu_util: str = _f(
        "0.40",
        "[vLLM] DISTRACE_GEN_GPU_UTIL",
        "Fraction of total GPU VRAM allocated to the vLLM generator engine "
        "(default 0.40). Increase to 0.90 on single-model steps with no "
        "co-resident embedder; decrease on tight-VRAM cards. Corresponds to "
        "env var DISTRACE_GEN_GPU_UTIL.",
    )

    env_distrace_embed_gpu_util: str = _f(
        "0.30",
        "[vLLM] DISTRACE_EMBED_GPU_UTIL",
        "Fraction of total GPU VRAM allocated to the vLLM embedder engine "
        "(default 0.30). The generator default (0.40) leaves room so both can "
        "coexist at ~0.70 of the card. Corresponds to env var "
        "DISTRACE_EMBED_GPU_UTIL.",
    )

    # -- llama-cpp backend --------------------------------------------- #
    env_distrace_embedder_device: str = _f(
        "",
        "[llama-cpp] DISTRACE_EMBEDDER_DEVICE",
        "Override the compute device for the SentenceTransformer embedder used "
        "by the llama-cpp backend ('cuda', 'cpu', 'cuda:1', ...). Leave empty "
        "for auto-detection. Corresponds to env var DISTRACE_EMBEDDER_DEVICE.",
    )

    env_distrace_embed_fp32: str = _f(
        "0",
        "[llama-cpp] DISTRACE_EMBED_FP32",
        "Set to '1' to load the llama-cpp embedder in fp32 instead of fp16. "
        "Doubles VRAM usage; only needed when fp16 is unsupported or unstable on "
        "your card. Corresponds to env var DISTRACE_EMBED_FP32.",
        choices=["0", "1"],
    )

    env_distrace_encode_batch: str = _f(
        "32",
        "[llama-cpp] DISTRACE_ENCODE_BATCH",
        "Initial encode batch size for the llama-cpp SentenceTransformer embedder. "
        "Halved automatically on CUDA OOM. Corresponds to env var "
        "DISTRACE_ENCODE_BATCH.",
    )

    env_distrace_noderag_workers: str = _f(
        "",
        "[llama-cpp] DISTRACE_NODERAG_WORKERS",
        "Force exactly N llama-cpp pool worker contexts for NodeRAG parallel "
        "decoding (0 = single-context / no pool; empty = auto from VRAM). "
        "Corresponds to env var DISTRACE_NODERAG_WORKERS.",
    )

    # -- Shared / embedder --------------------------------------------- #
    env_distrace_embed_maxlen: str = _f(
        "512",
        "[Embedder] DISTRACE_EMBED_MAXLEN",
        "Cap the embedder's max sequence length (tokens). Fact-check claims are "
        "short; a long cap wastes activation memory and can trigger OOM on the "
        "4B embedder. Raise for NodeRAG (which embeds long attribute prompts). "
        "Corresponds to env var DISTRACE_EMBED_MAXLEN.",
    )

    env_distrace_cw_cpu: str = _f(
        "0",
        "[Detector] DISTRACE_CW_CPU",
        "Force the check-worthiness detector to run on CPU ('1') instead of GPU. "
        "Useful when the detector and a large generator compete for the same "
        "VRAM. Corresponds to env var DISTRACE_CW_CPU.",
        choices=["0", "1"],
    )

    # -- HuggingFace --------------------------------------------------- #
    env_hf_hub_download_timeout: str = _f(
        "60",
        "[HF] HF_HUB_DOWNLOAD_TIMEOUT",
        "HTTP timeout in seconds for HuggingFace Hub model downloads (default 60). "
        "Increase on slow or rate-limited connections. Corresponds to env var "
        "HF_HUB_DOWNLOAD_TIMEOUT.",
    )

    env_hf_token: str = _f(
        "",
        "[HF] HF_TOKEN",
        "HuggingFace access token. When set, lifts anonymous download rate limits "
        "and grants access to gated models. Leave empty for unauthenticated access. "
        "Corresponds to env var HF_TOKEN.",
    )

    # -- NodeRAG -------------------------------------------------------- #
    env_distrace_noderag_maxtok: str = _f(
        "4096",
        "[NodeRAG] DISTRACE_NODERAG_MAXTOK",
        "Maximum tokens per LLM call inside NodeRAG graph construction. Increase "
        "for richer community summaries at the cost of speed/VRAM. Corresponds "
        "to env var DISTRACE_NODERAG_MAXTOK.",
    )

    env_distrace_noderag_dim: str = _f(
        "1536",
        "[NodeRAG] DISTRACE_NODERAG_DIM",
        "Embedding dimensionality expected by NodeRAG (must match the embedder "
        "model's output dim). Default 1536 matches the original SpecFi setup. "
        "Corresponds to env var DISTRACE_NODERAG_DIM.",
    )

    env_distrace_noderag_chunk: str = _f(
        "1048",
        "[NodeRAG] DISTRACE_NODERAG_CHUNK",
        "Chunk size (characters) for NodeRAG document splitting. Larger chunks "
        "preserve more context; smaller chunks improve retrieval granularity. "
        "Corresponds to env var DISTRACE_NODERAG_CHUNK.",
    )

    env_distrace_noderag_lang: str = _f(
        "English",
        "[NodeRAG] DISTRACE_NODERAG_LANG",
        "Natural language name passed to NodeRAG for prompt construction "
        "(e.g. 'English'). Corresponds to env var DISTRACE_NODERAG_LANG.",
    )

    env_distrace_noderag_rate: str = _f(
        "40",
        "[NodeRAG] DISTRACE_NODERAG_RATE",
        "NodeRAG API rate-limit (requests per minute). Applies when NodeRAG is "
        "configured to use an external LLM/embedding API rather than the local "
        "vLLM/llama-cpp backend. Corresponds to env var DISTRACE_NODERAG_RATE.",
    )

    env_distrace_noderag_lang_filter: str = _f(
        "EN",
        "[NodeRAG] DISTRACE_NODERAG_LANG_FILTER",
        "ISO 639-1 language code used to filter the PolyNarrative KB articles "
        "fed to NodeRAG (e.g. 'EN' for English-only). Corresponds to env var "
        "DISTRACE_NODERAG_LANG_FILTER.",
    )

    env_distrace_nar_no_llm: str = _f(
        "0",
        "[Narratives] DISTRACE_NAR_NO_LLM",
        "Set to '1' to skip LLM-based narrative central-claim synthesis and run "
        "the dense/BM25 assigner without a generator. Useful for quick smoke "
        "tests. Corresponds to env var DISTRACE_NAR_NO_LLM.",
        choices=["0", "1"],
    )

    env_distrace_nar_noderag_index_root: str = _f(
        "",
        "[Narratives] DISTRACE_NAR_NODERAG_INDEX_ROOT",
        "Override the root directory for NodeRAG index files used by the "
        "narrative extraction step. Leave empty to use the default "
        "knowledge/noderag path. Corresponds to env var "
        "DISTRACE_NAR_NODERAG_INDEX_ROOT.",
    )

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        self._locked: set = set()
        # Apply env-var fields to os.environ so the active session reflects them
        self._apply_env_fields()

    # ---- env-field → os.environ sync --------------------------------- #

    # Mapping: config field name -> os.environ key
    _ENV_FIELD_MAP: ClassVar[dict[str, str]] = {
        "env_vllm_deep_gemm_warmup":            "VLLM_DEEP_GEMM_WARMUP",
        "env_vllm_use_deep_gemm":               "VLLM_USE_DEEP_GEMM",
        "env_use_flashinfer_sampler":           "DISTRACE_USE_FLASHINFER_SAMPLER",
        "env_disable_kernel_mapping":           "DISABLE_KERNEL_MAPPING",
        "env_distrace_show_tqdm":               "DISTRACE_SHOW_TQDM",
        "env_distrace_gen_gpu_util":            "DISTRACE_GEN_GPU_UTIL",
        "env_distrace_embed_gpu_util":          "DISTRACE_EMBED_GPU_UTIL",
        "env_distrace_embedder_device":         "DISTRACE_EMBEDDER_DEVICE",
        "env_distrace_embed_fp32":              "DISTRACE_EMBED_FP32",
        "env_distrace_encode_batch":            "DISTRACE_ENCODE_BATCH",
        "env_distrace_noderag_workers":         "DISTRACE_NODERAG_WORKERS",
        "env_distrace_embed_maxlen":            "DISTRACE_EMBED_MAXLEN",
        "env_distrace_cw_cpu":                  "DISTRACE_CW_CPU",
        "env_hf_hub_download_timeout":          "HF_HUB_DOWNLOAD_TIMEOUT",
        "env_hf_token":                         "HF_TOKEN",
        "env_distrace_noderag_maxtok":          "DISTRACE_NODERAG_MAXTOK",
        "env_distrace_noderag_dim":             "DISTRACE_NODERAG_DIM",
        "env_distrace_noderag_chunk":           "DISTRACE_NODERAG_CHUNK",
        "env_distrace_noderag_lang":            "DISTRACE_NODERAG_LANG",
        "env_distrace_noderag_rate":            "DISTRACE_NODERAG_RATE",
        "env_distrace_noderag_lang_filter":     "DISTRACE_NODERAG_LANG_FILTER",
        "env_distrace_nar_no_llm":              "DISTRACE_NAR_NO_LLM",
        "env_distrace_nar_noderag_index_root":  "DISTRACE_NAR_NODERAG_INDEX_ROOT",
    }

    def _apply_env_fields(self) -> None:
        """Write env-field values to os.environ. Empty strings unset the var."""
        for fname, evar in self._ENV_FIELD_MAP.items():
            val = str(getattr(self, fname, ""))
            if val:
                os.environ[evar] = val
            else:
                os.environ.pop(evar, None)
        # Also propagate DISTRACE_LLM_BACKEND so core.models picks it up
        os.environ["DISTRACE_LLM_BACKEND"] = str(self.llm_backend)

    # ---- persistence / CLI ------------------------------------------- #
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        cfg = cls.__new__(cls)
        # Set dataclass field defaults manually before __post_init__
        for f in fields(cls):
            object.__setattr__(cfg, f.name, f.default)
        # Read live env-vars as defaults for env fields before JSON overlay
        for fname, evar in cls._ENV_FIELD_MAP.items():
            live = os.environ.get(evar)
            if live is not None:
                object.__setattr__(cfg, fname, live)
        # Overlay from saved JSON
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                for f in fields(cls):
                    if f.name in data:
                        object.__setattr__(cfg, f.name, data[f.name])
            except Exception as exc:
                # Surface, don't swallow — if config.json is malformed the
                # user must know rather than silently get defaults.
                import logging
                logging.getLogger(__name__).warning(
                    "[config] could not load %s: %s — using defaults", path, exc)
        object.__setattr__(cfg, "_locked", set())
        cfg._apply_env_fields()
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        path.write_text(json.dumps(d, indent=2), encoding="utf-8")
        # Apply env fields to current process immediately on save
        self._apply_env_fields()

    def apply_cli(self, args: argparse.Namespace) -> list[str]:
        overridden = []
        for f in fields(type(self)):
            val = getattr(args, f.name, None)
            if val is not None:
                self.set(f.name, val)
                overridden.append(f.name)
        self._locked = set(overridden)
        self._apply_env_fields()
        return overridden

    @staticmethod
    def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
        for f in fields(Config):
            argname = "--" + f.name.replace("_", "-")
            if f.metadata.get("choices"):
                parser.add_argument(argname, dest=f.name,
                                    choices=f.metadata["choices"], default=None)
            else:
                parser.add_argument(argname, dest=f.name, default=None)

    # ---- TUI introspection / mutation helpers ------------------------- #
    def field_names(self) -> list[str]:
        return [f.name for f in fields(self)]

    def _meta(self, name: str) -> dict:
        return {f.name: f.metadata for f in fields(self)}[name]

    def label(self, name: str) -> str:
        return self._meta(name).get("label") or name

    def desc(self, name: str) -> str:
        return self._meta(name).get("desc", "")

    def choices(self, name: str):
        return self._meta(name).get("choices")

    def is_locked(self, name: str) -> bool:
        return name in getattr(self, "_locked", set())

    def get(self, name: str):
        return getattr(self, name)

    def set(self, name: str, raw) -> None:
        if self.is_locked(name):
            return
        cur = getattr(self, name)
        if isinstance(cur, bool):
            val = raw if isinstance(raw, bool) \
                  else str(raw).strip().lower() in ("1", "true", "on", "yes")
        elif isinstance(cur, int) and not isinstance(cur, bool):
            val = int(raw)
        elif isinstance(cur, float):
            val = float(raw)
        else:
            val = str(raw)
        setattr(self, name, val)

    def cycle(self, name: str, direction: int) -> None:
        """Toggle a bool, or advance a choice list (wraps). No-op for free fields."""
        if self.is_locked(name):
            return
        cur = getattr(self, name)
        if isinstance(cur, bool):
            setattr(self, name, not cur)
            return
        ch = self.choices(name)
        if ch:
            i = ch.index(cur) if cur in ch else 0
            setattr(self, name, ch[(i + direction) % len(ch)])

    def reset(self) -> None:
        for f in fields(type(self)):
            if f.name in getattr(self, "_locked", set()):
                continue
            setattr(self, f.name, f.default)
