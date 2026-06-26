"""Agentic search harness over narrative-cluster documents (Context-1 style).

This is the cluster-document adaptation of the Chroma Context-1 agent harness.
The original reference harness (core/search/harness.py in the veracity pipeline)
operates on ``Chunk`` objects from a fact-check corpus; here the unit of
retrieval is a *narrative cluster* whose "document" is the concatenation of its
member central claims, exactly the unit ``Context1Backend`` ranks. We therefore
drop the ``Chunk`` type entirely and speak ``(cluster_id, document_str)`` tuples
throughout — the protocol ``core/hierarchy/backends/context1.py`` already
expects via ``_ClusterSearchTools``.

Differences from the reference harness, all deliberate:
  * Tools are injected (the ``tools`` object: ``search(query, seen, k)``,
    ``grep(pattern, seen)``, ``get(cluster_id)``) rather than a corpus with a
    ``Chunk`` API — so the same loop works over any id/text source.
  * The LLM is a DisTraceAI generator: a callable ``generate(system, user, *,
    temperature, max_tokens)`` (VLLMGenerator),
    NOT an object with a ``.query()`` method.
  * Token accounting is approximate (≈ 4 chars/token over the document text),
    since cluster documents have no precomputed ``token_count``.

The observe→reason→act loop, parallel JSON tool calls, soft/hard budget hints,
``<think>`` stripping, and seen-id deduplication mirror the reference.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_BUDGET_LINE = "\n[Token usage: {used}/{total}]"
_SOFT_HINT = ("\n[CONTEXT BUDGET WARNING: {pct:.0f}% full. "
              "Consider pruning irrelevant items or issuing your final answer.]")
_HARD_BLOCK = "\n[CONTEXT BUDGET CRITICAL: Only prune_chunks or done allowed.]"
_SEEN_LINE = "\n[Items already in context: {n}]"


def _approx_tokens(text: str) -> int:
    """Approximate token count from character length (~4 chars/token)."""
    return max(1, len(text) // 4)


@dataclass
class _State:
    """Mutable agent state across turns. ``active`` maps cluster_id -> document."""
    active: dict[str, str] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)
    costs: dict[str, int] = field(default_factory=dict)
    tokens_used: int = 0
    turn_count: int = 0

    def add(self, cluster_id: str, document: str, cost: int | None = None) -> None:
        if cluster_id in self.active:
            return
        self.active[cluster_id] = document
        self.seen.add(cluster_id)
        # Charge the agent's *context* budget, not the full gathered document.
        # The harness only ever surfaces a short summary of each hit to the LLM
        # (see _do_search); the full document is retained purely as OUTPUT for
        # the llm_backends to re-score. Charging the full doc here made a single
        # top_k search of large cluster docs blow the budget and trip the hard
        # cutoff on turn 1 — collapsing the multi-turn loop to a single pass.
        c = _approx_tokens(document) if cost is None else int(cost)
        self.costs[cluster_id] = c
        self.tokens_used += c

    def prune(self, cluster_ids: list[str]) -> int:
        freed = 0
        for cid in cluster_ids:
            if cid in self.active:
                freed += self.costs.pop(cid, _approx_tokens(self.active[cid]))
                del self.active[cid]
        self.tokens_used = max(0, self.tokens_used - freed)
        return freed

    def gathered(self) -> list[tuple[str, str]]:
        return list(self.active.items())


def _summary(text: str, n: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + "…"


class AgenticSearchHarness:
    """Multi-turn agentic search over cluster documents.

    Parameters
    ----------
    tools :
        Adapter exposing ``search(query, seen, k) -> [(cid, doc)]``,
        ``grep(pattern, seen) -> [(cid, doc)]``, and ``get(cid) -> str | None``
        (see ``_ClusterSearchTools`` in the Context-1 llm_backends).
    generate :
        DisTraceAI generator callable ``generate(system, user, *, temperature,
        max_tokens) -> str``.
    system_prompt :
        The discovery system prompt describing the available tools.
    token_budget, top_k, max_turns, soft_threshold, hard_cutoff :
        Loop / budget controls (same semantics as the reference harness).
    """

    def __init__(self, tools, generate, system_prompt, *,
                 token_budget: int = 8192, top_k: int = 5, max_turns: int = 8,
                 soft_threshold: float = 0.80, hard_cutoff: float = 0.95,
                 min_searches: int = 2) -> None:
        self.tools = tools
        self.generate = generate
        self.system_prompt = system_prompt
        self.token_budget = token_budget
        self.top_k = top_k
        self.max_turns = max_turns
        self.soft_limit = int(token_budget * soft_threshold)
        self.hard_limit = int(token_budget * hard_cutoff)
        # The agent must actually retrieve before it's allowed to finish: a
        # premature `done` (or an unparseable first reply that the parser maps to
        # a synthetic `done`) would otherwise collapse the whole multi-turn loop
        # into a single inference. Until this many search_corpus calls have run,
        # `done` is ignored and a parse failure falls back to a decomposed search
        # on the query rather than terminating.
        self.min_searches = max(1, min_searches)

    # ---- tool execution --------------------------------------------------
    def _do_search(self, query: str, state: _State) -> str:
        results = self.tools.search(query, state.seen, self.top_k)
        if not results:
            return f"[search_corpus] No new results for: {query!r}"
        lines = [f"[search_corpus] {len(results)} results for {query!r}:"]
        for cid, doc in results:
            summ = _summary(doc)
            state.add(cid, doc, _approx_tokens(summ))
            lines.append(f"  [{cid}] {summ}")
        return "\n".join(lines)

    def _do_grep(self, pattern: str, state: _State) -> str:
        results = self.tools.grep(pattern, state.seen)
        if not results:
            return f"[grep_corpus] No new matches for pattern: {pattern!r}"
        lines = [f"[grep_corpus] {len(results)} matches for {pattern!r}:"]
        for cid, doc in results:
            summ = _summary(doc)
            state.add(cid, doc, _approx_tokens(summ))
            lines.append(f"  [{cid}] {summ}")
        return "\n".join(lines)

    def _do_read(self, cluster_id: str, state: _State) -> str:
        doc = self.tools.get(cluster_id)
        if doc is None:
            return f"[read_document] Not found: {cluster_id}"
        state.add(cluster_id, doc)
        return f"[read_document] [{cluster_id}]\n{doc}"

    def _do_prune(self, cluster_ids: list[str], state: _State) -> str:
        freed = state.prune(cluster_ids)
        return (f"[prune_chunks] Pruned {len(cluster_ids)} item(s), freed ~{freed} "
                f"tokens. Budget used: {state.tokens_used}/{self.token_budget}.")

    # ---- LLM call + parsing ---------------------------------------------
    def _build_user(self, observation: str, state: _State) -> str:
        body = observation + _BUDGET_LINE.format(used=state.tokens_used,
                                                  total=self.token_budget)
        if state.tokens_used >= self.hard_limit:
            body += _HARD_BLOCK
        elif state.tokens_used >= self.soft_limit:
            pct = state.tokens_used / max(1, self.token_budget) * 100
            body += _SOFT_HINT.format(pct=pct)
        body += _SEEN_LINE.format(n=len(state.active))
        return body

    def _call_llm(self, user_msg: str) -> str:
        try:
            return (self.generate(self.system_prompt, user_msg,
                                  temperature=0.0, max_tokens=512) or "").strip()
        except TypeError:
            # Generator without temperature/max_tokens kwargs.
            return (self.generate(self.system_prompt, user_msg) or "").strip()
        except Exception as exc:                       # pragma: no cover - runtime
            logger.warning("[harness] LLM call failed: %s", exc)
            return '{"tool": "done", "reasoning": "LLM error"}'

    @staticmethod
    def _parse_calls(raw: str) -> list[dict]:
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        for attempt in (raw, raw.split("\n")[0] if raw else ""):
            try:
                parsed = json.loads(attempt)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [c for c in parsed if isinstance(c, dict)]
        for pattern in (r"\[.*?\]", r"\{.*?\}"):
            m = re.search(pattern, raw, re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(parsed, dict):
                    return [parsed]
                if isinstance(parsed, list):
                    return [c for c in parsed if isinstance(c, dict)]
        logger.debug("[harness] could not parse LLM output: %s", raw[:200])
        return [{"tool": "done", "reasoning": "parse_error"}]

    # ---- main loop -------------------------------------------------------
    def search(self, query: str) -> list[tuple[str, str]]:
        """Run the observe→reason→act loop; return gathered [(cluster_id, doc)]."""
        state = _State()
        searches_done = 0
        observation = (
            f"Claim to find matching narrative clusters for:\n{query}\n\n"
            "Decompose this claim into sub-topics and start searching. "
            "Use parallel tool calls where possible."
        )
        for turn in range(self.max_turns):
            state.turn_count = turn + 1
            if (state.tokens_used >= self.hard_limit and turn > 0
                    and searches_done >= self.min_searches):
                logger.info("[harness] hard token cutoff at turn %d", turn)
                break
            raw = self._call_llm(self._build_user(observation, state))
            calls = self._parse_calls(raw)
            parts: list[str] = []
            done = False
            for call in calls:
                tool = call.get("tool", "done")
                if tool == "search_corpus" and call.get("query"):
                    parts.append(self._do_search(call["query"], state))
                    searches_done += 1
                elif tool == "grep_corpus":
                    parts.append(self._do_grep(call.get("pattern", ""), state))
                elif tool == "read_document":
                    parts.append(self._do_read(call.get("chunk_id", ""), state))
                elif tool == "prune_chunks" and call.get("chunk_ids"):
                    parts.append(self._do_prune(call["chunk_ids"], state))
                elif tool == "done":
                    # Don't let the agent finish before it has actually searched.
                    # A first-turn `done` (often a parse fallback) would otherwise
                    # turn this into single-shot inference. Force a real search
                    # on the raw query instead and keep looping.
                    if searches_done < self.min_searches:
                        logger.info("[harness] premature done at turn %d "
                                    "(%d/%d searches) — forcing a search",
                                    turn + 1, searches_done, self.min_searches)
                        parts.append(self._do_search(query, state))
                        searches_done += 1
                    else:
                        logger.info("[harness] done after %d turn(s): %s",
                                    turn + 1, call.get("reasoning", ""))
                        done = True
                        break
                else:
                    parts.append(f"[harness] Unknown tool: {tool!r}")
            observation = "\n".join(parts) if parts else "[No tool output]"
            if done:
                break
        gathered = state.gathered()
        logger.info("[harness] complete — %d cluster(s) in %d turn(s), "
                    "%d search(es), ~%d tokens",
                    len(gathered), state.turn_count, searches_done,
                    state.tokens_used)
        return gathered
