# Copyright (c) 2026 angrysky56 (Ty).
#
# Unit tests for the self-contained creativity selection operations
# (graph_of_thoughts/operations/creativity.py). They use a fake language model,
# so they make no network calls.

from __future__ import annotations

import math
import re

import pytest

from graph_of_thoughts.operations import (
    KeepPareto,
    NoveltyScore,
    RubricScore,
)
from graph_of_thoughts.operations.creativity import (
    bidirectional_entailment,
    cluster_by_entailment,
    novelty_from_clusters,
    semantic_entropy,
)
from graph_of_thoughts.operations.operations import Operation
from graph_of_thoughts.operations.thought import Thought


# --------------------------------------------------------------------------- #
# Test doubles
# --------------------------------------------------------------------------- #


class FakeLM:
    """Minimal stand-in for AbstractLanguageModel; routes prompts to a handler."""

    def __init__(self, handler):
        self.handler = handler
        self.calls = 0

    def query(self, prompt, num_responses=1, temperature=None):
        self.calls += 1
        return [self.handler(prompt)]

    def get_response_texts(self, responses):
        return list(responses)


class StubPredecessor(Operation):
    """Holds a fixed list of thoughts to feed downstream operations."""

    def __init__(self, thoughts):
        super().__init__()
        self._thoughts = thoughts
        self.executed = True

    def _execute(self, *a, **k):  # pragma: no cover - never run
        pass

    def get_thoughts(self):
        return self._thoughts


def _entail_handler(prompt: str) -> str:
    """YES iff the two embedded answers are textually identical."""
    found = re.findall(r'Answer [AB]: "(.*?)"', prompt, re.DOTALL)
    if len(found) == 2:
        return "YES" if found[0].strip() == found[1].strip() else "NO"
    return "NO"


# --------------------------------------------------------------------------- #
# Pure-function tests
# --------------------------------------------------------------------------- #


def test_novelty_from_clusters_modal_vs_singleton():
    # clusters [0,0,1]: the singleton is maximally novel, the modal pair less so
    nov = novelty_from_clusters([0, 0, 1])
    assert nov[2] == pytest.approx(1.0)
    assert nov[0] == nov[1] < nov[2]
    assert nov[0] == pytest.approx(-math.log(2 / 3) / math.log(3), rel=1e-6)


def test_novelty_all_distinct_and_singleton_edge():
    assert novelty_from_clusters([0, 1, 2]) == [pytest.approx(1.0)] * 3
    assert novelty_from_clusters([0]) == [0.0]  # single candidate -> no novelty
    assert novelty_from_clusters([]) == []


def test_semantic_entropy_bounds():
    assert semantic_entropy([0, 0, 0]) == pytest.approx(0.0)  # collapsed
    assert semantic_entropy([0, 1, 2]) == pytest.approx(math.log(3))  # uniform/max


def test_bidirectional_entailment_shortcut_and_judge():
    lm = FakeLM(lambda p: "NO")
    # identical strings short-circuit without calling the LM
    assert bidirectional_entailment(lm, "same", "same") is True
    assert lm.calls == 0
    assert bidirectional_entailment(lm, "a", "b") is False
    assert lm.calls == 1


def test_cluster_by_entailment_groups_equivalents():
    lm = FakeLM(_entail_handler)
    ids = cluster_by_entailment(lm, ["apple", "apple", "banana"])
    assert ids[0] == ids[1] != ids[2]


# --------------------------------------------------------------------------- #
# Operation tests
# --------------------------------------------------------------------------- #


def test_novelty_score_operation_writes_axis():
    thoughts = [
        Thought({"current": "apple"}),
        Thought({"current": "apple"}),
        Thought({"current": "banana"}),
    ]
    op = NoveltyScore()
    op.add_predecessor(StubPredecessor(thoughts))
    op._execute(FakeLM(_entail_handler), None, None)
    out = op.get_thoughts()
    assert len(out) == 3
    novelties = [t.state["_novelty"] for t in out]
    assert novelties[2] > novelties[0]  # singleton more novel than modal
    assert all(t.scored for t in out)
    # original predecessor states must not be mutated
    assert "_novelty" not in thoughts[0].state


def test_rubric_score_parses_verdicts():
    def handler(_prompt):
        return "1. [[YES]]\n2. [[NO]]\n3. [[YES]]"

    op = RubricScore(criteria=["a", "b", "c"])
    op.add_predecessor(
        StubPredecessor([Thought({"current": "x", "original": "p"})])
    )
    op._execute(FakeLM(handler), None, None)
    t = op.get_thoughts()[0]
    # stored axes are rounded to 4 decimals for clean JSON output
    assert t.state["_quality"] == pytest.approx(2 / 3, abs=1e-3)
    assert t.state["_quality_breakdown"] == {"a": True, "b": False, "c": True}


def test_rubric_score_pads_missing_verdicts():
    op = RubricScore(criteria=["a", "b", "c"])
    op.add_predecessor(StubPredecessor([Thought({"current": "x"})]))
    op._execute(FakeLM(lambda p: "1. [[YES]]"), None, None)
    # only one verdict returned -> remaining padded to NO -> 1/3
    assert op.get_thoughts()[0].state["_quality"] == pytest.approx(1 / 3, abs=1e-3)


def _thought(nov, qual):
    return Thought({"current": f"n{nov}q{qual}", "_novelty": nov, "_quality": qual})


def test_keep_pareto_floor_excludes_low_quality():
    # high novelty but quality below floor -> excluded
    thoughts = [_thought(0.9, 0.2), _thought(0.4, 0.8), _thought(0.6, 0.7)]
    op = KeepPareto(floor=0.5)
    op.add_predecessor(StubPredecessor(thoughts))
    op._execute(None, None, None)
    kept = {t.state["current"] for t in op.get_thoughts()}
    assert "n0.9q0.2" not in kept  # failed the floor
    # both survivors are non-dominated (trade novelty vs quality)
    assert kept == {"n0.4q0.8", "n0.6q0.7"}


def test_keep_pareto_drops_dominated():
    thoughts = [_thought(0.8, 0.8), _thought(0.5, 0.5), _thought(0.6, 0.9)]
    op = KeepPareto(floor=0.0)
    op.add_predecessor(StubPredecessor(thoughts))
    op._execute(None, None, None)
    kept = {t.state["current"] for t in op.get_thoughts()}
    # (0.5,0.5) is dominated by (0.8,0.8); (0.8,0.8) and (0.6,0.9) are a frontier
    assert kept == {"n0.8q0.8", "n0.6q0.9"}


def test_keep_pareto_floor_fallback_never_empty():
    thoughts = [_thought(0.9, 0.1), _thought(0.7, 0.2)]
    op = KeepPareto(floor=0.5)
    op.add_predecessor(StubPredecessor(thoughts))
    op._execute(None, None, None)
    kept = op.get_thoughts()
    assert len(kept) == 1  # falls back to best-by-quality
    assert kept[0].state["current"] == "n0.7q0.2"


def test_keep_pareto_cap_n():
    thoughts = [_thought(0.9, 0.6), _thought(0.6, 0.9), _thought(0.75, 0.75)]
    op = KeepPareto(floor=0.5, n=2)
    op.add_predecessor(StubPredecessor(thoughts))
    op._execute(None, None, None)
    assert len(op.get_thoughts()) == 2
