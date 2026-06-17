# Copyright (c) 2026 ETH Zurich / Modernized.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

"""
Asynchronous controller for Graph of Thoughts.
"""

import asyncio
from typing import Any, Set

from graph_of_thoughts.operations import Operation

from .controller import Controller


class AsyncController(Controller):
    """
    AsyncController class to manage the concurrent execution flow of the Graph of Operations,
    generating the Graph Reasoning State asynchronously.
    """

    def run(self) -> Any:  # type: ignore[override]
        """
        Run the controller asynchronously.

        :return: A coroutine that can be awaited.
        :rtype: Coroutine
        """
        return self._run()

    async def _run(self) -> None:
        """
        Run the controller asynchronously and execute the operations from the Graph of
        Operations concurrently based on their readiness (DAG topological order with parallelism).
        """
        self.logger.debug("Checking that the program is in a valid state")
        if self.graph.roots is None:
            raise AssertionError("The operations graph has no root")
        self.logger.debug("The program is in a valid state")

        # Reset executed flag to False before execution
        for op in self.graph.operations:
            op.executed = False

        completed_operations: Set[Operation] = set()
        queued_operations: Set[Operation] = set()
        running_tasks: Set[asyncio.Task] = set()

        # Find initial roots that are ready to run
        ready_to_run = [op for op in self.graph.operations if op.can_be_executed()]

        # Helper function to run a single operation
        async def run_operation(op: Operation) -> Operation:
            self.logger.info(
                "Executing operation %d (%s) asynchronously", op.id, op.operation_type
            )
            await op.execute_async(
                self.lm, self.prompter, self.parser, **self.problem_parameters
            )
            self.logger.info(
                "Operation %d (%s) executed asynchronously", op.id, op.operation_type
            )
            return op

        # Create tasks for all initially ready operations
        for op in ready_to_run:
            task = asyncio.create_task(run_operation(op))
            running_tasks.add(task)
            queued_operations.add(op)

        # Loop until all operations are completed and no tasks are running
        while running_tasks:
            # Wait for any of the running tasks to complete
            done, running_tasks = await asyncio.wait(
                running_tasks, return_when=asyncio.FIRST_COMPLETED
            )

            for task in done:
                completed_op = await task
                completed_operations.add(completed_op)

                # Check successors of the completed operation
                for successor in completed_op.successors:
                    if successor not in self.graph.operations:
                        raise AssertionError(
                            "The successor of an operation is not in the operations graph"
                        )

                    if (
                        successor not in completed_operations
                        and successor not in queued_operations
                    ):
                        # If all predecessors are now completed, can_be_executed() returns True
                        if successor.can_be_executed():
                            self.logger.debug(
                                "Successor %d (%s) is now ready to execute",
                                successor.id,
                                successor.operation_type,
                            )
                            new_task = asyncio.create_task(run_operation(successor))
                            running_tasks.add(new_task)
                            queued_operations.add(successor)

        # Sanity check: verify all operations in the graph were executed
        unexecuted = [op for op in self.graph.operations if not op.executed]
        if unexecuted:
            self.logger.warning(
                "Some operations in the graph were not executed: %s",
                [op.id for op in unexecuted],
            )

        self.logger.info("All operations executed asynchronously")
        self.run_executed = True
