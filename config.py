"""Single-source configuration.

One settings object, edited by either the TUI or the CLI. Each field carries its
own label / description / choice-list metadata (the single source the TUI reads),
plus get/set/cycle/lock helpers so the editor logic stays testable. CLI flags
override the saved file for the run and are surfaced as locked (read-only) in the
TUI.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path

CONFIG_PATH = Path("config.json")


def _f(default, label, desc, choices=None):
    return field(default=default, metadata={"label": label, "desc": desc, "choices": choices})


@dataclass
class Config:
    detector: str = _f(
        "models/xlm-multicw",
        "Check-worthiness classifier",
        "Fine-tuned check-worthiness classifier (mDeBERTa or XLM-R), under models/.",
        choices=["models/xlm-multicw", "models/mdb-multicw"],
    )

    canon_detector: str = _f(
        "models/xlm-multicw",
        "Canonization source detector",
        "Which claim detector's output to canonize (must match a prior claim-detection run).",
        choices=["models/xlm-multicw", "models/mdb-multicw", "both"],
    )

    canon_generator: str = _f(
        "qwen3.5-2b",
        "Canonization generator",
        "LLM used to decontextualize and translate check-worthy claims to English.",
        choices=["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
                 "gemma4-e2b", "gemma4-e4b", "gemma4-12b"],
    )

    canon_precision: str = _f(
        "awq4",
        "Canonization precision",
        "Model precision for the canonization generator (awq4=4-bit, bf16=16-bit).",
        choices=["awq4", "bf16"],
    )

    subnar_detector: str = _f(
        "models/xlm-multicw",
        "Sub-narrative source detector",
        "Which claim detector's canonized output to use for sub-narrative extraction.",
        choices=["models/xlm-multicw", "models/mdb-multicw", "both"],
    )

    subnar_embedder: str = _f(
        "Qwen/Qwen3-Embedding-0.6B",
        "Sub-narrative embedder",
        "SentenceTransformer model used to embed canonized claims for similarity clustering.",
        choices=["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B",
                 "intfloat/multilingual-e5-large-instruct"],
    )

    subnar_generator: str = _f(
        "qwen3.5-2b",
        "Sub-narrative generator",
        "LLM used to synthesize the central claim for each sub-narrative cluster.",
        choices=["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
                 "gemma4-e2b", "gemma4-e4b", "gemma4-12b"],
    )

    subnar_precision: str = _f(
        "awq4",
        "Sub-narrative precision",
        "Model precision for the sub-narrative generator (awq4=4-bit, bf16=16-bit).",
        choices=["awq4", "bf16"],
    )

    subnar_min_similarity: float = _f(
        0.45,
        "Min claim similarity",
        "Cosine similarity threshold: canonized claims at or above this value are "
        "assigned to the current sub-narrative cluster.",
    )

    subnar_min_claims: int = _f(
        2,
        "Min claims per sub-narrative",
        "Minimum number of claims required to form a sub-narrative. Remaining "
        "claims are discarded when the pool falls below this threshold.",
    )

    subnar_hypotheticals: int = _f(
        3,
        "HyDE hypotheticals",
        "Number of hypothetical sub-narrative descriptions generated per central "
        "claim during evaluation retrieval (HyDE style).",
    )

    # ------------------------------------------------------------------ #
    # Narrative extraction (step 5)
    # ------------------------------------------------------------------ #
    nar_detector: str = _f(
        "models/xlm-multicw",
        "Narrative source detector",
        "Which detector's sub-narratives feed narrative extraction / retrieval.",
        choices=["models/xlm-multicw", "models/mdb-multicw", "both"],
    )

    nar_extractor: str = _f(
        "dense",
        "Narrative retrieval method",
        "Retrieval backend for the narrative eval and Generate step: "
        "dense (embedding cosine, repr selected by nar_dense_repr), "
        "bm25-rag (BM25+dense RRF hybrid, no LLM, strongest non-LLM baseline), "
        "specfi-cs (reproduced original static SpecFi-CS, NodeRAG over article texts), "
        "specfi-ccs (SpecFi-CCS, NodeRAG over per-article canonized claims), "
        "cspecfi (our continuous variant, no NodeRAG, conditioned on sub-narrative claims), "
        "context-1 (agentic multi-turn search harness), "
        "all (Evaluation only: benchmark every method and print a summary table).",
        choices=["dense", "bm25-rag", "specfi-cs", "specfi-ccs", "cspecfi",
                 "context-1", "all"],
    )

    nar_dense_repr: str = _f(
        "subnar",
        "Dense representation",
        "Only read when nar_extractor=dense. Which text represents an item: "
        "article (raw article text), canonized (set of canonized claims), or "
        "subnar (sub-narrative central claim).",
        choices=["article", "canonized", "subnar"],
    )

    nar_embedder: str = _f(
        "Qwen/Qwen3-Embedding-4B",
        "Narrative embedder",
        "SentenceTransformer model used to embed queries and corpus items. "
        "4B is the default for reproducibility with the SpecFi paper.",
        choices=["Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-0.6B",
                 "intfloat/multilingual-e5-large-instruct"],
    )

    nar_generator: str = _f(
        "qwen3.5-2b",
        "Narrative generator",
        "LLM used to synthesize narrative central claims (Generate) and HyDE "
        "hypotheticals (specfi-cs / cspecfi / context-1).",
        choices=["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
                 "gemma4-e2b", "gemma4-e4b", "gemma4-12b"],
    )

    nar_precision: str = _f(
        "awq4",
        "Narrative precision",
        "Model precision for the narrative generator (awq4=4-bit, bf16=16-bit).",
        choices=["awq4", "bf16"],
    )

    nar_assign_threshold: float = _f(
        0.55,
        "Narrative assign threshold",
        "Cosine score above which a sub-narrative merges into the top-ranked "
        "existing narrative (Generate).",
    )

    nar_min_new_size: int = _f(
        3,
        "Min new-narrative size",
        "Minimum number of mutually similar unassigned sub-narratives required to "
        "seed a brand-new narrative (Generate).",
    )

    nar_new_threshold: float = _f(
        0.75,
        "New-narrative similarity",
        "Cosine threshold used while clustering the unassigned pool into new "
        "narratives (Generate).",
    )

    nar_recluster_cadence: int = _f(
        0,
        "Recluster / rebuild cadence",
        "Run the periodic sweep (and, for specfi, rebuild the NodeRAG graph) every "
        "N processed articles. 0 disables periodic sweeps (build once up front).",
    )

    nar_specfi_hypotheticals: int = _f(
        10,
        "SpecFi hypotheticals",
        "Number of hypothetical texts generated per query for specfi-cs and "
        "cspecfi. Default 10 matches the paper's generate_hypotheticals(n=10).",
    )

    nar_context1_context_size: int = _f(
        32768,
        "Context-1 model context size",
        "llama.cpp context window size when loading the Context-1 model. "
        "Context-1 is trained on 128K context; 32768 is a practical minimum.",
    )

    nar_context1_max_turns: int = _f(
        8,
        "Context-1 max turns",
        "Hard cap on agentic search turns per query (nar_extractor=context-1).",
    )

    nar_context1_token_budget: int = _f(
        8192,
        "Context-1 evidence token budget",
        "Maximum tokens of retrieved cluster evidence the agentic harness "
        "accumulates before terminating. This is NOT the model context size "
        "(see nar_context1_context_size for that). 8192 ≈ half the default ctx.",
    )

    nar_eval_split: str = _f(
        "test",
        "Narrative eval query split",
        "Which held-out PolyNarrative split supplies query sub-narratives; the "
        "corpus is always built from train.",
        choices=["dev", "test"],
    )

    nar_eval_domain: str = _f(
        "all",
        "Narrative eval domain",
        "Restrict the narrative eval to one PolyNarrative domain: CC (climate "
        "change), URW (Ukraine-Russia war), or all (both). Filters both the "
        "query split and the train corpus to the chosen domain.",
        choices=["all", "CC", "URW"],
    )

    # ------------------------------------------------------------------ #
    # Claim veracity estimation (step 3)
    # ------------------------------------------------------------------ #
    ver_sources: str = _f(
        "multiclaim,wikipedia,web",
        "Veracity evidence sources",
        "Comma-separated list of evidence sources for the agentic harness. "
        "Omit any to disable: multiclaim (local CSV), wikipedia (online API), "
        "web (online search). Sources degrade gracefully when offline.",
    )

    ver_generator: str = _f(
        "gemma4-e2b",
        "Veracity verdict generator",
        "LLM used to synthesize the True/False/Disputed verdict from gathered "
        "evidence snippets.",
        choices=["gemma4-e2b", "gemma4-e4b", "gemma4-12b",
                 "qwen3.5-2b", "qwen3.5-4b"],
    )

    ver_paraphrase_generator: str = _f(
        "gemma4-12b",
        "Paraphrase generator",
        "LLM used to generate paraphrased test queries from MultiClaim for the "
        "veracity evaluation benchmark. Cached to knowledge/veracity/ after "
        "first run.",
        choices=["gemma4-12b", "gemma4-e4b", "qwen3.5-9b"],
    )

    ver_precision: str = _f(
        "awq4",
        "Veracity precision",
        "Model precision for the verdict + paraphrase generators (awq4=4-bit, bf16=16-bit).",
        choices=["awq4", "bf16"],
    )

    ver_max_turns: int = _f(
        6,
        "Veracity max turns",
        "Maximum agentic search turns per claim verification.",
    )

    ver_token_budget: int = _f(
        4096,
        "Veracity evidence token budget",
        "Maximum evidence tokens the agentic harness accumulates per claim. "
        "Not the model context size.",
    )

    ver_n_paraphrases: int = _f(
        3,
        "Paraphrases per claim",
        "How many paraphrase variants to generate per MultiClaim entry for the "
        "veracity evaluation benchmark.",
    )

    ver_multiclaim_text_col: str = _f(
        "claim",
        "MultiClaim text column",
        "Column name in the MultiClaim CSV that contains the claim text.",
    )

    ver_multiclaim_label_col: str = _f(
        "ratings",
        "MultiClaim label column",
        "Column name in the MultiClaim CSV that contains the verdict label "
        "(filtered to True/False/Disputed). When the configured column is not "
        "found, the loader tries label, verdict, ratings, rating in order. "
        "The published fact_checks.csv uses a ratings column with Python-repr "
        "lists (e.g. ['true']) which the loader parses automatically.",
    )

    # ------------------------------------------------------------------ #
    # Campaigns extraction (step 6)
    # ------------------------------------------------------------------ #
    camp_detector: str = _f(
        "models/xlm-multicw",
        "Campaign source detector",
        "Which detector's narrative hierarchy feeds campaign extraction.",
        choices=["models/xlm-multicw", "models/mdb-multicw", "both"],
    )

    camp_extractor: str = _f(
        "dense",
        "Campaign retrieval method",
        "Retrieval backend for campaign extraction: same choices as nar_extractor.",
        choices=["dense", "bm25-rag", "specfi-cs", "specfi-ccs", "cspecfi", "context-1"],
    )

    camp_dense_repr: str = _f(
        "subnar",
        "Campaign dense representation",
        "Text representation for dense retrieval; repurposed as narrative central "
        "claim here (subnar is the correct choice for narrative→campaign).",
        choices=["article", "canonized", "subnar"],
    )

    camp_embedder: str = _f(
        "Qwen/Qwen3-Embedding-4B",
        "Campaign embedder",
        "Embedding model for campaign retrieval.",
        choices=["Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-0.6B",
                 "intfloat/multilingual-e5-large-instruct"],
    )

    camp_generator: str = _f(
        "qwen3.5-2b",
        "Campaign generator",
        "LLM for synthesizing campaign central claims.",
        choices=["qwen3.5-2b", "qwen3.5-4b", "qwen3.5-9b",
                 "gemma4-e2b", "gemma4-e4b", "gemma4-12b"],
    )

    camp_precision: str = _f(
        "awq4",
        "Campaign precision",
        "Model precision for the campaign generator (awq4=4-bit, bf16=16-bit).",
        choices=["awq4", "bf16"],
    )

    camp_assign_threshold: float = _f(
        0.50,
        "Campaign assign threshold",
        "Cosine score above which a narrative merges into the top-ranked "
        "existing campaign.",
    )

    camp_min_new_size: int = _f(
        2,
        "Min new-campaign size",
        "Minimum number of mutually similar unassigned narratives to seed a "
        "new campaign cluster.",
    )

    camp_new_threshold: float = _f(
        0.70,
        "New-campaign similarity",
        "Cosine threshold for clustering unassigned narratives into new campaigns.",
    )

    camp_recluster_cadence: int = _f(
        0,
        "Campaign recluster cadence",
        "Run periodic sweep every N processed narratives. 0 disables.",
    )

    camp_specfi_hypotheticals: int = _f(
        10,
        "Campaign SpecFi hypotheticals",
        "Hypothetical texts per query for specfi-cs / cspecfi campaign retrieval.",
    )

    camp_context1_max_turns: int = _f(
        8,
        "Campaign Context-1 max turns",
        "Max agentic turns per query for context-1 campaign retrieval.",
    )

    camp_context1_token_budget: int = _f(
        8192,
        "Campaign Context-1 token budget",
        "Evidence token budget for context-1 campaign retrieval.",
    )

    camp_coordination_threshold: float = _f(
        0.40,
        "Coordination threshold",
        "Coordination score above which a campaign is classified as coordinated "
        "(Information or Disinformation Campaign) rather than Organic Trend.",
    )

    camp_veracity_threshold: float = _f(
        0.45,
        "Veracity threshold",
        "Veracity score below which a coordinated campaign is classified as "
        "Disinformation Campaign rather than Information Campaign. "
        "Campaigns with no veracity verdict default to Information Campaign.",
    )

    camp_n1_weight: float = _f(
        0.30,
        "N1 burst weight",
        "Weight of the burst/time-synchrony signal in the coordination score.",
    )

    camp_n2_weight: float = _f(
        0.25,
        "N2 co-amplification weight",
        "Weight of the co-amplification (shared outlets) signal.",
    )

    camp_n3_weight: float = _f(
        0.25,
        "N3 content-reuse weight",
        "Weight of the near-identical content reuse signal.",
    )

    camp_n4_weight: float = _f(
        0.20,
        "N4 cross-lingual weight",
        "Weight of the cross-lingual co-occurrence signal.",
    )

    # ------------------------------------------------------------------ #
    def __post_init__(self):
        self._locked: set = set()

    # ---- persistence / CLI ----
    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "Config":
        cfg = cls()
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            for f in fields(cls):
                if f.name in data:
                    setattr(cfg, f.name, data[f.name])
        return cfg

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    def apply_cli(self, args: argparse.Namespace) -> list[str]:
        overridden = []
        for f in fields(type(self)):
            val = getattr(args, f.name, None)
            if val is not None:
                self.set(f.name, val)
                overridden.append(f.name)
        self._locked = set(overridden)
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

    # ---- TUI introspection / mutation helpers ----
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
            return   # locked by a CLI argument for this run; ignore edits
        cur = getattr(self, name)
        if isinstance(cur, bool):
            val = raw if isinstance(raw, bool) else str(raw).strip().lower() in ("1", "true", "on", "yes")
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
            return   # locked by a CLI argument for this run; ignore edits
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
