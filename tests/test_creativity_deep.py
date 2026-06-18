# Copyright (c) 2026 angrysky56 (Ty).
#
# Unit tests for the deeper, concurrent creativity operations
# (graph_of_thoughts/operations/creativity_deep.py). A fake LM routes prompts to
# canned responses, so there are no network calls.

from __future__ import annotations

import asyncio
import json
import re

import pytest

from graph_of_thoughts.operations import (
    ComparativeScore,
    DivergentGenerate,
    MultiPersonaJudge,
)
from graph_of_thoughts.operations.creativity_deep import cluster_semantic_single_call
from graph_of_thoughts.operations.operations import Operation
from graph_of_thoughts.operations.thought import Thought


class FakeLM:
    def __init__(self, handler):
        self.handler = handler

    def query(self, prompt, num_responses=1, temperature=None):
        return [self.handler(prompt)]

    def get_response_texts(self, responses):
        return list(responses)


class StubPredecessor(Operation):
    def __init__(self, thoughts):
        super().__init__()
        self._t = thoughts
        self.executed = True

    def _execute(self, *a, **k):  # pragma: no cover
        pass

    def get_thoughts(self):
        return self._t


class StubPrompter:
    def generate_prompt(self, num_branches, **kwargs):
        return "GEN"


class StubParser:
    def parse_generate_answer(self, state, texts):
        return [{"current": t} for t in texts]


def _count_listing(prompt: str) -> int:
    return len(re.findall(r"^\s*\d+\.", prompt, re.M))


# --------------------------------------------------------------------------- #


def test_cluster_single_call_parses_and_renumbers():
    lm = FakeLM(lambda p: "groups: [5, 5, 7, 9, 7]")
    ids = asyncio.run(cluster_semantic_single_call(lm, ["a", "b", "c", "d", "e"]))
    assert ids == [0, 0, 1, 2, 1]


def test_cluster_single_call_fallback_all_distinct():
    lm = FakeLM(lambda p: "no json here")
    ids = asyncio.run(cluster_semantic_single_call(lm, ["a", "b", "c"]))
    assert ids == [0, 1, 2]


def test_none_content_is_coerced_not_crashing():
    # a model returning content=None must not blow up regex/JSON parsing
    lm = FakeLM(lambda p: None)
    ids = asyncio.run(cluster_semantic_single_call(lm, ["a", "b", "c"]))
    assert ids == [0, 1, 2]  # degrades to all-distinct
    op = ComparativeScore(scale=10.0)
    op.add_predecessor(StubPredecessor([Thought({"current": "a", "original": "p"})]))
    op._execute(lm, None, None)
    # None content -> coerced to "" -> no numbers -> neutral 0.5 (not a crash)
    assert op.get_thoughts()[0].state["_quality"] == 0.5


def test_divergent_generate_commits_when_diverse():
    counter = {"i": 0}

    def handler(prompt):
        if "group number" in prompt:  # the clustering call
            n = _count_listing(prompt)
            return json.dumps(list(range(1, n + 1)))  # all distinct
        counter["i"] += 1  # a generate call -> unique idea
        return f"distinct idea {counter['i']}"

    op = DivergentGenerate(k=3, max_rounds=2)
    op.add_predecessor(StubPredecessor([Thought({"original": "problem", "current": ""})]))
    op._execute(FakeLM(handler), StubPrompter(), StubParser())

    out = op.get_thoughts()
    assert len(out) == 3  # one round, committed (fully diverse)
    assert all("_novelty" in t.state and "_entropy" in t.state for t in out)
    assert all(t.state["_novelty"] == pytest.approx(1.0) for t in out)  # all singletons
    assert counter["i"] == 3  # did not escalate to a second round


def test_divergent_generate_escalates_when_collapsed():
    counter = {"i": 0}

    def handler(prompt):
        if "group number" in prompt:
            n = _count_listing(prompt)
            return json.dumps([1] * n)  # everything collapses to one class
        counter["i"] += 1
        return "same idea"

    op = DivergentGenerate(k=3, max_rounds=2)
    op.add_predecessor(StubPredecessor([Thought({"original": "p", "current": ""})]))
    op._execute(FakeLM(handler), StubPrompter(), StubParser())

    out = op.get_thoughts()
    # collapsed -> ran both rounds -> 3 + 3 = 6 generate calls
    assert counter["i"] == 6
    assert len(out) == 6
    assert all(t.state["_novelty"] == 0.0 for t in out)  # single class -> no novelty


def test_comparative_score_normalises_relative_scores():
    lm = FakeLM(lambda p: "Here are the ratings: [8, 4, 10]")
    op = ComparativeScore(scale=10.0)
    op.add_predecessor(
        StubPredecessor(
            [
                Thought({"current": "a", "original": "p"}),
                Thought({"current": "b", "original": "p"}),
                Thought({"current": "c", "original": "p"}),
            ]
        )
    )
    op._execute(lm, None, None)
    quals = [t.state["_quality"] for t in op.get_thoughts()]
    assert quals == [pytest.approx(0.8), pytest.approx(0.4), pytest.approx(1.0)]


def test_comparative_uses_utility_lm_when_set():
    primary = FakeLM(lambda p: "[0, 0, 0]")  # must NOT be used for scoring
    utility = FakeLM(lambda p: "[10, 10, 10]")
    op = ComparativeScore(scale=10.0)
    op.utility_lm = utility
    op.add_predecessor(
        StubPredecessor(
            [
                Thought({"current": "a", "original": "p"}),
                Thought({"current": "b", "original": "p"}),
                Thought({"current": "c", "original": "p"}),
            ]
        )
    )
    op._execute(primary, None, None)
    quals = [t.state["_quality"] for t in op.get_thoughts()]
    assert quals == [pytest.approx(1.0)] * 3  # utility LM's scores were used


def test_comparative_neutral_fallback_on_parse_failure():
    # utility model returns no numbers -> neutral 0.5, never silent zeros
    lm = FakeLM(lambda p: "the model rambled with no usable scores")
    op = ComparativeScore(scale=10.0)
    op.add_predecessor(
        StubPredecessor(
            [
                Thought({"current": "a", "original": "p"}),
                Thought({"current": "b", "original": "p"}),
            ]
        )
    )
    op._execute(lm, None, None)
    quals = [t.state["_quality"] for t in op.get_thoughts()]
    assert quals == [pytest.approx(0.5)] * 2


def test_multi_persona_judge_majority_vote():
    # every persona: criterion 1 & 2 YES, criterion 3 NO -> 2/3 pass
    lm = FakeLM(lambda p: "1. [[YES]]\n2. [[YES]]\n3. [[NO]]")
    op = MultiPersonaJudge(criteria=["a", "b", "c"])
    op.add_predecessor(StubPredecessor([Thought({"current": "sol", "original": "p"})]))
    op._execute(lm, None, None)
    t = op.get_thoughts()[0]
    assert t.state["_quality"] == pytest.approx(2 / 3, abs=1e-3)
    assert t.state["_quality_judge"] == {"a": True, "b": True, "c": False}
