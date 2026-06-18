# Copyright (c) 2026 angrysky56 (Ty).
#                    All rights reserved.
#
# Use of this source code is governed by the BSD-style license that governs
# this repository (see LICENSE). This module is an original addition; it does
# not reproduce upstream code.
#
# The reference-free creativity signals implemented here (semantic-class
# novelty via bidirectional-entailment clustering, and a per-criterion rubric
# judge) are re-expressions of the measurement framework from:
#   Tan Min Sen et al., "Automated Creativity Evaluation of Language Models
#   Across Open-Ended Tasks", ACL 2026. The implementation below is original
#   and self-contained: it reuses the graph's existing language model rather
#   than any external service or model.

"""Principled, self-contained selection operations for Graph of Thoughts.

Adds three operations that upgrade GoT's weak default scoring/selection:

* :class:`RubricScore`   -- LLM rubric judge with deterministic per-criterion
  ``[[YES]]/[[NO]]`` parsing (a "quality"/convergent axis).
* :class:`NoveltyScore`  -- reference-free novelty: cluster a node's sibling
  thoughts by bidirectional entailment, score each by the normalised surprisal
  of its semantic class (a "novelty"/divergent axis).
* :class:`KeepPareto`    -- multi-axis Pareto selection with a *convergent
  floor*. The floor is the anti-Goodhart guard: novelty is never maximised on
  its own; a thought must clear a minimum quality before it can win on novelty.

Axes are stored on each thought's ``state`` under string keys (default
``_novelty`` and ``_quality``) so several scorers can stack without clobbering
one another, and so the values survive into the JSON results. Each scorer also
sets ``thought.score`` (to its own axis) for compatibility with the stock
``KeepBestN``/``Aggregate`` operations.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Dict, List

from graph_of_thoughts.language_models import AbstractLanguageModel
from graph_of_thoughts.operations.operations import Operation
from graph_of_thoughts.operations.thought import Thought
from graph_of_thoughts.parser import Parser
from graph_of_thoughts.prompter import Prompter


class CreativeOpType(Enum):
    """Operation-type tags for the creativity operations (kept separate from the
    upstream ``OperationType`` enum so upstream files stay untouched)."""

    RUBRIC_SCORE = 100
    NOVELTY_SCORE = 101
    KEEP_PARETO = 102


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

_VERDICT_RE = re.compile(r"\[\[\s*(YES|NO)\s*\]\]", re.IGNORECASE)


def answer_text(state: Dict) -> str:
    """Best-effort extraction of the answer text from a thought state.

    Prefers the conventional ``current`` field; otherwise falls back to the
    longest string value in the state.
    """
    if not isinstance(state, dict):
        return str(state)
    val = state.get("current")
    if isinstance(val, str) and val.strip():
        return val.strip()
    strings = [v for v in state.values() if isinstance(v, str) and v.strip()]
    if strings:
        return max(strings, key=len).strip()
    return str(state)


def get_axis(state: Dict, name: str) -> float | None:
    """Read a stored axis value from a thought state, or None if absent."""
    if isinstance(state, dict) and isinstance(state.get(name), (int, float)):
        return float(state[name])
    return None


def _clone_with_state(thought: Thought) -> Thought:
    """Clone a thought with a shallow-copied state so axis writes do not mutate
    the predecessor's shared state dict."""
    new_thought = Thought.from_thought(thought)
    new_thought.state = (
        {**thought.state} if isinstance(thought.state, dict) else thought.state
    )
    return new_thought


def _query_one(
    lm: AbstractLanguageModel, prompt: str, temperature: float | None = 0.0
) -> str:
    """Query the LM for a single response and return its text.

    Defaults to ``temperature=0.0`` because these calls are *judgements*
    (entailment / rubric verdicts), where randomness only adds noise. Backends
    that predate the temperature override simply ignore the extra argument.
    """
    try:
        responses = lm.get_response_texts(
            # pyrefly: ignore [unexpected-keyword]
            lm.query(prompt, num_responses=1, temperature=temperature)
        )
    except TypeError:
        # Backend without the temperature override.
        responses = lm.get_response_texts(lm.query(prompt, num_responses=1))
    # coerce a possible None content to "" so parsers never see None
    return (responses[0] if responses else "") or ""


# ----------------------------------------------------------------------------
# Entailment clustering + novelty (divergent axis)
# ----------------------------------------------------------------------------

_ENTAIL_TMPL = (
    "You are judging whether two answers express the SAME core idea.\n"
    'Answer A: "{a}"\n'
    'Answer B: "{b}"\n'
    "Do A and B express essentially the same idea, each entailing the other "
    "(ignoring wording, length, and style)? Reply with ONLY YES or NO."
)


def bidirectional_entailment(lm: AbstractLanguageModel, a: str, b: str) -> bool:
    """Return True if a and b are judged to express the same idea (both
    directions), using the graph's own LLM as the judge.

    This approximates the paper's bidirectional-entailment equivalence test
    without adding a dedicated NLI model.
    """
    a, b = a.strip(), b.strip()
    if a == b:
        return True
    text = _query_one(lm, _ENTAIL_TMPL.format(a=a, b=b)).lower()
    return "yes" in text and "no" not in text.split()


def cluster_by_entailment(lm: AbstractLanguageModel, texts: List[str]) -> List[int]:
    """Greedy bidirectional-entailment clustering.

    Each text joins the first existing class whose representative it is
    bidirectionally equivalent to, else it starts a new class. Returns the
    class index for each input text.
    """
    reps: List[str] = []
    cluster_ids: List[int] = []
    for text in texts:
        placed = False
        for idx, rep in enumerate(reps):
            if bidirectional_entailment(lm, text, rep):
                cluster_ids.append(idx)
                placed = True
                break
        if not placed:
            reps.append(text)
            cluster_ids.append(len(reps) - 1)
    return cluster_ids


def novelty_from_clusters(cluster_ids: List[int]) -> List[float]:
    """Per-item novelty = normalised surprisal of the item's semantic class.

    Normalised by the *maximum* surprisal present in the set, so the rarest
    class scores 1.0 and the modal class scores lowest (0 when only one class).
    This matches the creativity-evaluation reference implementation (relative,
    set-internal novelty) rather than an absolute ``ln(N)`` scale.
    """
    n = len(cluster_ids)
    if n <= 1:
        return [0.0] * n
    sizes: Dict[int, int] = {}
    for cid in cluster_ids:
        sizes[cid] = sizes.get(cid, 0) + 1
    surprisal = {cid: -math.log(size / n) for cid, size in sizes.items()}
    max_surprisal = max(surprisal.values())
    if max_surprisal <= 0:
        return [0.0] * n
    return [surprisal[cid] / max_surprisal for cid in cluster_ids]


def semantic_entropy(cluster_ids: List[int]) -> float:
    """Shannon entropy (nats) of the semantic-class distribution."""
    n = len(cluster_ids)
    if n == 0:
        return 0.0
    sizes: Dict[int, int] = {}
    for cid in cluster_ids:
        sizes[cid] = sizes.get(cid, 0) + 1
    return -sum((s / n) * math.log(s / n) for s in sizes.values())


# ----------------------------------------------------------------------------
# Operations
# ----------------------------------------------------------------------------


class NoveltyScore(Operation):
    """Score each predecessor thought by the novelty of its semantic class.

    Writes the value to ``state[axis]`` (default ``_novelty``) and to
    ``thought.score``.
    """

    operation_type = CreativeOpType.NOVELTY_SCORE

    def __init__(self, axis: str = "_novelty") -> None:
        super().__init__()
        self.axis = axis
        # Optional fast "utility" LM for the entailment clustering calls.
        self.utility_lm: AbstractLanguageModel | None = None
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        if len(self.predecessors) < 1:
            raise AssertionError(
                "NoveltyScore operation needs at least one predecessor"
            )
        previous = self.get_previous_thoughts()
        if not previous:
            return
        texts = [answer_text(t.state) for t in previous]
        cluster_ids = cluster_by_entailment(self.utility_lm or lm, texts)
        novelties = novelty_from_clusters(cluster_ids)
        self.logger.info(
            "NoveltyScore op %d: %d thoughts -> %d semantic classes (entropy=%.3f)",
            self.id,
            len(previous),
            len(set(cluster_ids)),
            semantic_entropy(cluster_ids),
        )
        for thought, novelty in zip(previous, novelties, strict=False):
            nt = _clone_with_state(thought)
            nt.state[self.axis] = round(float(novelty), 4)
            nt.score = float(novelty)
            self.thoughts.append(nt)


class RubricScore(Operation):
    """LLM rubric judge: score each thought by the fraction of criteria met.

    For each predecessor thought, prompts for an explicit ``[[YES]]/[[NO]]``
    verdict per criterion and parses them deterministically. Writes the
    fraction to ``state[axis]`` (default ``_quality``), a per-criterion
    breakdown to ``state[axis + "_breakdown"]``, and to ``thought.score``.
    """

    operation_type = CreativeOpType.RUBRIC_SCORE

    def __init__(
        self,
        criteria: List[str],
        axis: str = "_quality",
        problem_key: str = "original",
        answer_key: str = "current",
        num_samples: int = 1,
    ) -> None:
        super().__init__()
        if not criteria:
            raise AssertionError("RubricScore needs at least one criterion")
        self.criteria = list(criteria)
        self.axis = axis
        self.problem_key = problem_key
        self.answer_key = answer_key
        self.num_samples = max(1, num_samples)
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _build_prompt(self, problem: str, answer: str) -> str:
        lines = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(self.criteria))
        return (
            "You are a strict evaluator. Judge the ANSWER against each "
            "criterion independently.\n"
            f"PROBLEM: {problem}\n"
            f"ANSWER: {answer}\n"
            "CRITERIA:\n"
            f"{lines}\n\n"
            "For each criterion, output exactly one line in order, formatted as:\n"
            "<n>. [[YES]]   or   <n>. [[NO]]\n"
            "Output only those lines, nothing else."
        )

    def _parse_verdicts(self, text: str) -> List[bool]:
        verdicts = [m.upper() == "YES" for m in _VERDICT_RE.findall(text)]
        # pad / truncate to the number of criteria
        if len(verdicts) < len(self.criteria):
            verdicts += [False] * (len(self.criteria) - len(verdicts))
        return verdicts[: len(self.criteria)]

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        if len(self.predecessors) < 1:
            raise AssertionError("RubricScore operation needs at least one predecessor")
        for thought in self.get_previous_thoughts():
            problem = ""
            if isinstance(thought.state, dict):
                problem = str(thought.state.get(self.problem_key, ""))
            answer = answer_text(thought.state)
            prompt = self._build_prompt(problem, answer)

            # average over num_samples for a more stable judgement; judge cold
            # for a single sample, slightly warm when sampling several so the
            # average is meaningful.
            judge_temp = 0.0 if self.num_samples == 1 else 0.3
            fracs: List[float] = []
            last_verdicts: List[bool] = []
            for _ in range(self.num_samples):
                verdicts = self._parse_verdicts(_query_one(lm, prompt, judge_temp))
                last_verdicts = verdicts
                fracs.append(sum(verdicts) / len(self.criteria))
            quality = sum(fracs) / len(fracs)

            nt = _clone_with_state(thought)
            nt.state[self.axis] = round(quality, 4)
            nt.state[f"{self.axis}_breakdown"] = {
                c: bool(v) for c, v in zip(self.criteria, last_verdicts, strict=False)
            }
            nt.score = float(quality)
            self.thoughts.append(nt)
        self.logger.info(
            "RubricScore op %d scored %d thoughts on %d criteria",
            self.id,
            len(self.thoughts),
            len(self.criteria),
        )


class KeepPareto(Operation):
    """Keep the Pareto-optimal thoughts across several axes, above a floor.

    A thought survives only if its ``floor_axis`` value is >= ``floor`` (the
    anti-Goodhart guard). Among survivors, keep the non-dominated set: a thought
    is dropped if another scores >= on every axis and strictly > on at least
    one. If ``n`` is given and the frontier is larger, keep the ``n`` with the
    highest axis sum. Never returns empty while any thought exists: if the floor
    excludes everything, the single best thought by ``floor_axis`` is kept.
    """

    operation_type = CreativeOpType.KEEP_PARETO

    def __init__(
        self,
        axes: List[str] | None = None,
        floor_axis: str = "_quality",
        floor: float = 0.5,
        n: int | None = None,
    ) -> None:
        super().__init__()
        self.axes = list(axes) if axes else ["_novelty", "_quality"]
        self.floor_axis = floor_axis
        self.floor = float(floor)
        self.n = n
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        return self.thoughts

    def _vec(self, thought: Thought) -> List[float]:
        return [get_axis(thought.state, ax) or 0.0 for ax in self.axes]

    @staticmethod
    def _dominates(a: List[float], b: List[float]) -> bool:
        """True if a dominates b (>= on all, > on at least one)."""
        return all(x >= y for x, y in zip(a, b, strict=False)) and any(
            x > y for x, y in zip(a, b, strict=False)
        )

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        if len(self.predecessors) < 1:
            raise AssertionError(
                "KeepPareto operation must have at least one predecessor"
            )
        previous = self.get_previous_thoughts()
        if not previous:
            return

        # 1) convergent floor
        survivors = [
            t
            for t in previous
            if (get_axis(t.state, self.floor_axis) or 0.0) >= self.floor
        ]
        if not survivors:
            # floor excluded everything: fall back to best by floor_axis
            best = max(
                previous, key=lambda t: get_axis(t.state, self.floor_axis) or 0.0
            )
            self.thoughts = [_clone_with_state(best)]
            self.logger.info(
                "KeepPareto op %d: floor=%.2f excluded all; kept best-by-%s",
                self.id,
                self.floor,
                self.floor_axis,
            )
            return

        # 2) Pareto frontier among survivors
        vecs = [self._vec(t) for t in survivors]
        frontier = [
            t
            for i, t in enumerate(survivors)
            if not any(
                self._dominates(vecs[j], vecs[i])
                for j in range(len(survivors))
                if j != i
            )
        ]

        # 3) optional cap by axis sum
        if self.n is not None and len(frontier) > self.n:
            frontier = sorted(frontier, key=lambda t: sum(self._vec(t)), reverse=True)[
                : self.n
            ]

        self.thoughts = [_clone_with_state(t) for t in frontier]
        self.logger.info(
            "KeepPareto op %d: %d in -> %d above floor -> %d on frontier",
            self.id,
            len(previous),
            len(survivors),
            len(self.thoughts),
        )
