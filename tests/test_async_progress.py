# Copyright (c) 2026 angrysky56 (Ty).
#
# Verifies AsyncController emits a progress callback once per completed
# operation (the mechanism that keeps long graphs under client timeouts).

from __future__ import annotations

import asyncio

from graph_of_thoughts.controller import AsyncController
from graph_of_thoughts.operations import GraphOfOperations
from graph_of_thoughts.operations.operations import Operation
from graph_of_thoughts.operations.thought import Thought


class Noop(Operation):
    def __init__(self):
        super().__init__()
        self.thoughts = [Thought({"current": "x"})]

    def _execute(self, lm, prompter, parser, **kwargs):
        pass

    def get_thoughts(self):
        return self.thoughts


def test_progress_callback_fires_per_operation():
    a, b = Noop(), Noop()
    b.add_predecessor(a)  # a -> b

    graph = GraphOfOperations()
    graph.add_operation(a)
    graph.add_operation(b)

    ctrl = AsyncController(
        lm=None, graph=graph, prompter=None, parser=None, problem_parameters={}
    )

    events = []

    async def cb(done, total, op):
        events.append((done, total, op.id))

    asyncio.run(ctrl.run(progress_callback=cb))

    assert len(events) == 2  # one per operation
    assert events[-1][0] == 2 and events[-1][1] == 2  # final: 2/2


def test_run_without_callback_still_works():
    a = Noop()
    graph = GraphOfOperations()
    graph.add_operation(a)
    ctrl = AsyncController(
        lm=None, graph=graph, prompter=None, parser=None, problem_parameters={}
    )
    asyncio.run(ctrl.run())  # no callback -> must not raise
    assert a.executed
