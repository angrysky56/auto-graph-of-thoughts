# Copyright (c) 2026 angrysky56 (Ty).
#                    All rights reserved.
#
# Use of this source code is governed by the BSD-style license that governs
# this repository (see LICENSE). Original addition; reuses only the graph's own
# language model (no external service, NLI model, or embedding store).
#
# This module ports — in spirit, as an original re-expression — the "deeper than
# the paper" creativity-selection design: set-level divergence with a
# metacognitive explore/commit controller, set-relative (comparative) quality
# scoring, and a multi-persona final judge. The underlying creativity signals
# trace to:
#   Tan Min Sen et al., "Automated Creativity Evaluation of Language Models
#   Across Open-Ended Tasks", ACL 2026.
# Design choices that differ from a literal port, and why:
#   * Clustering is done in ONE LLM call (not O(N^2) pairwise NLI) so it stays
#     responsive against slow remote models and within MCP timeouts.
#   * The multi-persona judge runs independent personas concurrently with no
#     retrieval/vector store (the API backends expose no embeddings); this keeps
#     it self-contained at the cost of the paper's retrieval-discussion rounds.
#   * Quality is scored set-relatively (all candidates in one call) because
#     judging candidates in isolation is a weak signal.

"""Deeper, concurrent creativity-selection operations for Graph of Thoughts.

* :class:`DivergentGenerate` -- explore/commit generator. Samples a population
  concurrently, measures semantic entropy + per-candidate novelty, and if the
  set has collapsed onto one idea, re-samples hotter (controller loop) before
  committing. Solves provider ``n``-fan-out limits and mode collapse at once.
* :class:`ComparativeScore`  -- set-relative quality: ONE call ranks/scores all
  candidates against each other (a stronger convergent signal than per-item).
* :class:`MultiPersonaJudge` -- final quality judge: several critic personas
  judge a finalist concurrently and their verdicts are aggregated.

All three are async (``_execute_async``) and parallelise their LLM calls with
``asyncio.to_thread`` over the synchronous ``lm.query`` (which carries the
per-call ``temperature`` override), so judging runs cold while generation runs
hot — without touching the async LM path.
"""

from __future__ import annotations

import asyncio
import json
import re
from enum import Enum
from typing import Dict, List

from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.operations.creativity import (
    _clone_with_state,
    answer_text,
    novelty_from_clusters,
    semantic_entropy,
)
from graph_of_thoughts.operations.operations import Operation
from graph_of_thoughts.operations.thought import Thought
from graph_of_thoughts.parser import Parser
from graph_of_thoughts.prompter import Prompter


class DeepOpType(Enum):
    """Operation-type tags for the deeper creativity operations."""

    DIVERGENT_GENERATE = 110
    COMPARATIVE_SCORE = 111
    MULTI_PERSONA_JUDGE = 112


_NUM_RE = re.compile(r"[-+]?\d*\.\d+|\d+")
_VERDICT_RE = re.compile(r"\[\[\s*(YES|NO)\s*\]\]", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# Async helpers (concurrency via threads over the sync, temperature-aware query)
# --------------------------------------------------------------------------- #


async def _aquery_one(
    lm: AbstractLanguageModel, prompt: str, temperature: float | None = None
) -> str:
    """Run a single query off-thread and return its text."""
    try:
        resp = await asyncio.to_thread(lm.query, prompt, 1, temperature)
    except TypeError:
        resp = await asyncio.to_thread(lm.query, prompt, 1)
    texts = lm.get_response_texts(resp)
    # a model may return content=None for a choice; coerce so downstream
    # regex/JSON parsing never sees None (the parsers degrade gracefully on "").
    return (texts[0] if texts else "") or ""


async def _gather_queries(
    lm: AbstractLanguageModel, prompts: List[str], temperature: float | None = None
) -> List[str]:
    """Run many queries concurrently; return their texts in order."""
    return await asyncio.gather(*(_aquery_one(lm, p, temperature) for p in prompts))


def _extract_json_list(text: str):
    """Best-effort parse of the first JSON array in text."""
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


_CLUSTER_TMPL = (
    "You are grouping answers by meaning.\n"
    "Answers:\n{listing}\n\n"
    "Assign each answer a group number starting at 1. Answers expressing the "
    "SAME core idea share a number; genuinely different ideas get different "
    "numbers. Return ONLY a JSON array of integers, one per answer in order, "
    "e.g. [1, 1, 2, 3, 2]."
)


async def cluster_semantic_single_call(
    lm: AbstractLanguageModel, texts: List[str]
) -> List[int]:
    """Cluster answers into semantic classes with a single LLM call.

    Returns a 0-based class id per input. Falls back to all-distinct on any
    parse failure. Much cheaper than pairwise entailment and responsive enough
    for remote models.
    """
    n = len(texts)
    if n <= 1:
        return [0] * n
    listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    raw = await _aquery_one(lm, _CLUSTER_TMPL.format(listing=listing), temperature=0.0)
    parsed = _extract_json_list(raw)
    if not isinstance(parsed, list) or len(parsed) != n:
        return list(range(n))  # fall back to "all distinct"
    # renumber to contiguous 0-based ids
    remap: Dict[object, int] = {}
    ids: List[int] = []
    for val in parsed:
        if val not in remap:
            remap[val] = len(remap)
        ids.append(remap[val])
    return ids


# --------------------------------------------------------------------------- #
# DivergentGenerate -- explore/commit controller + set-level novelty
# --------------------------------------------------------------------------- #


class DivergentGenerate(Operation):
    """Generate a *diverse* population, escalating temperature until it spreads.

    Each round samples ``k`` candidates concurrently, clusters them, and scores
    novelty. If the set has collapsed (few non-modal ideas) and rounds remain,
    it raises the temperature and samples again before committing. Output
    thoughts carry ``state[novelty_axis]`` and the set's ``state['_entropy']``.
    """

    operation_type = DeepOpType.DIVERGENT_GENERATE

    def __init__(
        self,
        k: int = 5,
        max_rounds: int = 2,
        base_temperature: float = 0.7,
        temperature_step: float = 0.3,
        max_temperature: float = 1.5,
        diversity_threshold: float = 0.34,
        novelty_eps: float = 0.05,
        novelty_axis: str = "_novelty",
    ) -> None:
        super().__init__()
        self.k = max(2, k)
        self.max_rounds = max(1, max_rounds)
        self.base_temperature = base_temperature
        self.temperature_step = temperature_step
        self.max_temperature = max_temperature
        self.diversity_threshold = diversity_threshold
        self.novelty_eps = novelty_eps
        self.novelty_axis = novelty_axis
        # Optional fast "utility" LM for the cheap structured call (clustering);
        # falls back to the primary LM when unset.
        self.utility_lm: AbstractLanguageModel | None = None
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _diversity(self, novelty: List[float]) -> float:
        """Fraction of non-modal candidates that escaped the modal basin."""
        if len(novelty) <= 1:
            return 0.0
        escaped = [v for v in novelty if v > self.novelty_eps]
        return len(escaped) / len(novelty)

    async def _execute_async(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        previous = self.get_previous_thoughts()
        if not previous and self.predecessors:
            return
        bases = previous if previous else [Thought(state=kwargs)]
        ulm = self.utility_lm or lm  # cheap clustering on the utility model

        for base in bases:
            base_state = base.state
            prompt = prompter.generate_prompt(1, **base_state)

            texts: List[str] = []
            novelty: List[float] = []
            cluster_ids: List[int] = []
            temp = self.base_temperature
            for round_idx in range(self.max_rounds):
                new_raw = await _gather_queries(lm, [prompt] * self.k, temperature=temp)
                for raw in new_raw:
                    for new_state in parser.parse_generate_answer(base_state, [raw]):
                        merged = {**base_state, **new_state}
                        texts.append(answer_text(merged))
                cluster_ids = await cluster_semantic_single_call(ulm, texts)
                novelty = novelty_from_clusters(cluster_ids)
                self.logger.info(
                    "DivergentGenerate op %d round %d: %d candidates, %d classes, "
                    "entropy=%.3f, diversity=%.2f, temp=%.2f",
                    self.id,
                    round_idx,
                    len(texts),
                    len(set(cluster_ids)),
                    semantic_entropy(cluster_ids),
                    self._diversity(novelty),
                    temp,
                )
                if (
                    self._diversity(novelty) >= self.diversity_threshold
                    or round_idx + 1 >= self.max_rounds
                ):
                    break
                temp = min(self.max_temperature, round(temp + self.temperature_step, 3))

            entropy = semantic_entropy(cluster_ids)
            for text, nov in zip(texts, novelty, strict=False):
                state = {**base_state, "current": text}
                state[self.novelty_axis] = round(float(nov), 4)
                state["_entropy"] = round(float(entropy), 4)
                nt = Thought(state)
                nt.score = float(nov)
                self.thoughts.append(nt)

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        asyncio.run(self._execute_async(lm, prompter, parser, **kwargs))


# --------------------------------------------------------------------------- #
# ComparativeScore -- set-relative quality in ONE call
# --------------------------------------------------------------------------- #


class ComparativeScore(Operation):
    """Score all candidates relative to each other in a single LLM call.

    Stronger than per-candidate judging: the model rates each candidate against
    the others on a 0..``scale`` scale, normalised into ``state[axis]``.
    """

    operation_type = DeepOpType.COMPARATIVE_SCORE

    def __init__(
        self,
        criteria: List[str] | None = None,
        axis: str = "_quality",
        problem_key: str = "original",
        scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.criteria = criteria or ["feasibility", "effectiveness", "insightfulness"]
        self.axis = axis
        self.problem_key = problem_key
        self.scale = float(scale)
        # Optional fast "utility" LM for the single relative-scoring call.
        self.utility_lm: AbstractLanguageModel | None = None
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _build_prompt(self, problem: str, texts: List[str]) -> str:
        listing = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
        crit = ", ".join(self.criteria)
        return (
            "Rate each candidate answer to the problem on overall quality, "
            f"considering: {crit}.\n"
            f"PROBLEM: {problem}\n"
            f"CANDIDATES:\n{listing}\n\n"
            f"Use an integer 0-{int(self.scale)} scale ({int(self.scale)} is best). "
            "Score candidates RELATIVE to each other. Return ONLY a JSON array of "
            f"{len(texts)} numbers in order, e.g. [7, 4, 9]."
        )

    async def _execute_async(
        self,
        lm: AbstractLanguageModel,
        _prompter: Prompter,
        _parser: Parser,
        **_kwargs,
    ) -> None:
        if not self.predecessors:
            raise AssertionError("ComparativeScore needs at least one predecessor")
        previous = self.get_previous_thoughts()
        if not previous:
            return
        texts = [answer_text(t.state) for t in previous]
        problem = ""
        if isinstance(previous[0].state, dict):
            problem = str(previous[0].state.get(self.problem_key, ""))

        ulm = self.utility_lm or lm  # relative scoring on the utility model
        raw = await _aquery_one(
            ulm, self._build_prompt(problem, texts), temperature=0.0
        )
        nums = [float(x) for x in _NUM_RE.findall(raw)]
        if len(nums) < len(texts):
            # Parse failure (e.g. a reasoning model returned empty/odd content):
            # fill with a NEUTRAL mid score, not 0, so selection isn't silently
            # zeroed out and keep_pareto's floor still behaves sensibly.
            missing = len(texts) - len(nums)
            self.logger.warning(
                "ComparativeScore: parsed %d/%d scores from utility model; "
                "filling %d with neutral %.1f (raw=%r)",
                len(nums),
                len(texts),
                missing,
                self.scale * 0.5,
                raw[:120],
            )
            nums += [self.scale * 0.5] * missing

        for thought, raw_score in zip(previous, nums[: len(texts)], strict=False):
            quality = max(0.0, min(1.0, raw_score / self.scale))
            nt = _clone_with_state(thought)
            nt.state[self.axis] = round(quality, 4)
            nt.score = float(quality)
            self.thoughts.append(nt)
        self.logger.info(
            "ComparativeScore op %d scored %d candidates relatively",
            self.id,
            len(self.thoughts),
        )

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        asyncio.run(self._execute_async(lm, prompter, parser, **kwargs))


# --------------------------------------------------------------------------- #
# MultiPersonaJudge -- concurrent multi-critic final judge
# --------------------------------------------------------------------------- #

_DEFAULT_PERSONAS = (
    "problem analyst (focus: the problem's real constraints and goals)",
    "solution analyst (focus: the solution's mechanism and coherence)",
    "criterion analyst (focus: strict reading of each criterion)",
)


class MultiPersonaJudge(Operation):
    """Final quality judge: several critic personas vote per criterion.

    For each finalist thought, every persona independently returns a
    ``[[YES]]/[[NO]]`` verdict per criterion (concurrently, cold). Verdicts are
    aggregated by majority; the score is the fraction of criteria passed and is
    written to ``state[axis]``.

    A pragmatic, self-contained stand-in for the paper's retrieval-based
    multi-agent ChatEval judge: independent personas instead of multi-round
    retrieval discussion (the API backends expose no embedding store).
    """

    operation_type = DeepOpType.MULTI_PERSONA_JUDGE

    def __init__(
        self,
        criteria: List[str] | None = None,
        personas: List[str] | None = None,
        axis: str = "_quality",
        problem_key: str = "original",
    ) -> None:
        super().__init__()
        self.criteria = criteria or ["feasibility", "effectiveness", "originality"]
        self.personas = list(personas) if personas else list(_DEFAULT_PERSONAS)
        self.axis = axis
        self.problem_key = problem_key
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _build_prompt(self, role: str, problem: str, solution: str) -> str:
        lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(self.criteria))
        return (
            f"You are a {role}, an impartial but critical judge.\n"
            f"PROBLEM: {problem}\n"
            f"SOLUTION: {solution}\n"
            "Judge whether the SOLUTION fulfils EACH criterion. For each, output "
            "one line in order as '<n>. [[YES]]' or '<n>. [[NO]]'.\n"
            f"CRITERIA:\n{lines}\n"
            "Output only those lines."
        )

    @staticmethod
    def _parse(text: str, n: int) -> List[bool]:
        verdicts = [m.upper() == "YES" for m in _VERDICT_RE.findall(text)]
        if len(verdicts) < n:
            verdicts += [False] * (n - len(verdicts))
        return verdicts[:n]

    async def _execute_async(
        self,
        lm: AbstractLanguageModel,
        _prompter: Prompter,
        _parser: Parser,
        **_kwargs,
    ) -> None:
        if not self.predecessors:
            raise AssertionError("MultiPersonaJudge needs at least one predecessor")
        nc = len(self.criteria)
        for thought in self.get_previous_thoughts():
            problem = ""
            if isinstance(thought.state, dict):
                problem = str(thought.state.get(self.problem_key, ""))
            solution = answer_text(thought.state)

            prompts = [self._build_prompt(r, problem, solution) for r in self.personas]
            responses = await _gather_queries(lm, prompts, temperature=0.0)
            persona_verdicts = [self._parse(r, nc) for r in responses]

            # majority vote per criterion
            breakdown: Dict[str, bool] = {}
            passed = 0
            for ci, crit in enumerate(self.criteria):
                yes = sum(1 for pv in persona_verdicts if pv[ci])
                won = yes * 2 >= len(self.personas)  # majority (ties -> pass)
                breakdown[crit] = won
                passed += int(won)
            quality = passed / nc if nc else 0.0

            nt = _clone_with_state(thought)
            nt.state[self.axis] = round(quality, 4)
            nt.state[f"{self.axis}_judge"] = breakdown
            nt.score = float(quality)
            self.thoughts.append(nt)
        self.logger.info(
            "MultiPersonaJudge op %d judged %d finalists with %d personas",
            self.id,
            len(self.thoughts),
            len(self.personas),
        )

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        asyncio.run(self._execute_async(lm, prompter, parser, **kwargs))
