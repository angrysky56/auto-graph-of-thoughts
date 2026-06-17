# Copyright (c) 2026 ETH Zurich / Modernized.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from graph_of_thoughts import controller, language_models, operations
from graph_of_thoughts.parser import Parser
from graph_of_thoughts.prompter import Prompter

# Load environment variables from .env if present
load_dotenv()

# Set up logging
logger = logging.getLogger("mcp_server_got")

# Initialize FastMCP Server
mcp = FastMCP("Graph of Thoughts")

# Stateful Sessions Registry
# maps session_id -> dict containing:
# - "graph": GraphOfOperations
# - "lm": AbstractLanguageModel
# - "prompter": Prompter
# - "parser": Parser
# - "initial_params": dict
# - "operations_map": dict mapping client node ID (int/str) to Operation instance
sessions: Dict[str, Dict[str, Any]] = {}


class DynamicPrompter(Prompter):
    """
    DynamicPrompter uses templates provided dynamically by the client.
    """

    def __init__(self, templates: Dict[str, str]) -> None:
        self.templates = templates or {}

    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        template = self.templates.get(
            "generate",
            "Generate {num_branches} thoughts or solutions based on the current state.\n"
            "State: {current}\n"
            "Original Problem: {original}",
        )
        return template.format(num_branches=num_branches, **kwargs)

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        template = self.templates.get(
            "score",
            "Evaluate and score the following thoughts or states. Return a numerical score for each (higher is better).\n"
            "Thoughts: {state_dicts}",
        )
        return template.format(state_dicts=state_dicts, **kwargs)

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        template = self.templates.get(
            "aggregation",
            "Combine or aggregate the following thoughts/states into a single consolidated thought or solution:\n"
            "Thoughts: {state_dicts}",
        )
        return template.format(state_dicts=state_dicts, **kwargs)

    def improve_prompt(self, **kwargs) -> str:
        template = self.templates.get(
            "improve", "Improve the following thought/state:\n" "State: {current}"
        )
        return template.format(**kwargs)

    def validation_prompt(self, **kwargs) -> str:
        template = self.templates.get(
            "validation",
            "Validate whether the following thought/state is correct or valid. Return YES or NO.\n"
            "State: {current}",
        )
        return template.format(**kwargs)


class DynamicParser(Parser):
    """
    DynamicParser parses responses based on a declarative spec or heuristics.
    """

    def __init__(self, parser_spec: Dict[str, Any] = None) -> None:
        self.spec = parser_spec or {}

    def _extract_json(self, text: str) -> Optional[Union[Dict, List]]:
        try:
            # Look for JSON block
            match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except Exception:
            return None

    def _extract_list(self, text: str) -> List[Any]:
        # Tries to parse as python list first, otherwise splits by comma
        try:
            match = re.search(r"(\[.*\])", text)
            if match:
                # safe evaluation of list literal
                import ast

                return ast.literal_eval(match.group(1))
        except Exception:
            pass
        # Fallback to splitting lines/commas
        cleaned = text.replace("[", "").replace("]", "").strip()
        if not cleaned:
            return []
        if "," in cleaned:
            return [item.strip() for item in cleaned.split(",")]
        return [line.strip() for line in cleaned.split("\n") if line.strip()]

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        new_states = []
        parse_type = self.spec.get("generate_type", "default")

        for text in texts:
            if parse_type == "json":
                js = self._extract_json(text)
                if isinstance(js, dict):
                    new_states.append(js)
                elif isinstance(js, list):
                    for item in js:
                        if isinstance(item, dict):
                            new_states.append(item)
                        else:
                            new_states.append({"current": str(item)})
                else:
                    new_states.append({"current": text})
            elif parse_type == "list":
                lst = self._extract_list(text)
                for val in lst:
                    new_states.append({"current": str(val)})
            else:
                # Default behavior: wrap in dict
                new_states.append({"current": text})

        return new_states

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        # Return a list of scores corresponding to each state
        # In combined scoring, texts[0] contains scores for all states
        scores = []

        for text in texts:
            # Find all numbers in the text
            numbers = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", text)]
            if numbers:
                scores.extend(numbers)

        # Ensure we have the correct length
        target_len = len(states)
        if len(scores) < target_len:
            scores.extend([0.0] * (target_len - len(scores)))
        return scores[:target_len]

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        new_states = []
        for text in texts:
            js = self._extract_json(text)
            if isinstance(js, dict):
                new_states.append(js)
            elif isinstance(js, list):
                new_states.append({"current": str(js)})
            else:
                new_states.append({"current": text})
        return new_states[0] if len(new_states) == 1 else new_states

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        text = texts[0] if texts else ""
        js = self._extract_json(text)
        if isinstance(js, dict):
            return js
        return {"current": text}

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        text = texts[0].lower() if texts else ""
        if "yes" in text or "true" in text or "valid" in text:
            return True
        return False


def get_task_prompter_parser(task_name: str, spec: dict = None, templates: dict = None):
    """
    Helper to get the corresponding Prompter and Parser based on task name.
    """
    task_name = task_name.lower()
    if task_name == "sorting":
        from examples.sorting.sorting_032 import SortingParser, SortingPrompter

        return SortingPrompter(), SortingParser()
    elif task_name == "keyword_counting":
        # In keyword_counting, parser is built as abstract or custom.
        # Let's import or fallback to keyword counting prompter / parser
        from examples.keyword_counting.keyword_counting import (
            KeywordCountingParser,
            KeywordCountingPrompter,
        )

        return KeywordCountingPrompter(), KeywordCountingParser()
    elif task_name == "set_intersection":
        from examples.set_intersection.set_intersection_032 import (
            SetIntersectionParser,
            SetIntersectionPrompter,
        )

        return SetIntersectionPrompter(), SetIntersectionParser()
    elif task_name == "doc_merge":
        from examples.doc_merge.doc_merge import DocMergeParser, DocMergePrompter

        return DocMergePrompter(), DocMergeParser()
    else:
        return DynamicPrompter(templates or {}), DynamicParser(spec or {})


# MCP Tools definition


@mcp.tool()
async def create_got_session(
    initial_parameters: dict,
    task_name: str = "custom",
    config_path: str = "",
    model_name: str = "",
    templates: Optional[dict] = None,
    parser_spec: Optional[dict] = None,
) -> str:
    """
    Create a stateful Graph of Thoughts session.

    :param initial_parameters: The dictionary containing the initial variables (e.g., {"original": "[2, 1, 3]", "current": ""}).
    :param task_name: Built-in task name ("sorting", "keyword_counting", "set_intersection", "doc_merge") or "custom".
    :param config_path: Path to config.json. Defaults to empty (uses default config.json).
    :param model_name: Name of model in config.json. Defaults to "chatgpt".
    :param templates: Optional dict containing custom prompt templates (for custom task).
    :param parser_spec: Optional dict containing custom parser instructions (for custom task).
    :return: A session_id string.
    """
    session_id = str(uuid.uuid4())

    import os

    if not model_name:
        model_name = os.getenv("GOT_LANGUAGE_MODEL", "chatgpt")

    # Instantiate Language Model
    if "openrouter" in model_name.lower():
        lm = language_models.OpenRouter(config_path, model_name=model_name)
    else:
        lm = language_models.ChatGPT(config_path, model_name=model_name)

    # Get Prompter and Parser
    prompter, parser = get_task_prompter_parser(task_name, parser_spec, templates)

    # Initialize Operations Graph
    graph = operations.GraphOfOperations()

    # Store session details
    sessions[session_id] = {
        "graph": graph,
        "lm": lm,
        "prompter": prompter,
        "parser": parser,
        "initial_params": initial_parameters,
        "operations_map": {},
    }

    logger.info("Created GoT session: %s", session_id)
    return session_id


@mcp.tool()
async def add_got_operation(
    session_id: str,
    op_type: str,
    client_op_id: Union[int, str],
    predecessor_ids: Optional[List[Union[int, str]]] = None,
    params: Optional[dict] = None,
) -> str:
    """
    Add a node (operation) to a stateful Graph of Thoughts session.

    :param session_id: The ID of the session.
    :param op_type: Type of operation ("generate", "score", "keep_best_n", "keep_valid", "aggregate", "improve", "validate_and_improve", "ground_truth").
    :param client_op_id: A unique ID for this operation node within the client's scope.
    :param predecessor_ids: Optional list of client_op_ids that this operation depends on.
    :param params: Dict of configuration parameters for the operation.
    :return: Confirmation message.
    """
    if session_id not in sessions:
        raise ValueError(f"Session {session_id} not found.")

    session = sessions[session_id]
    graph = session["graph"]
    ops_map = session["operations_map"]

    params = params or {}
    op_type = op_type.lower()

    # Instantiate Operation based on type
    if op_type == "generate":
        op = operations.Generate(
            num_branches_prompt=params.get("num_branches_prompt", 1),
            num_branches_response=params.get("num_branches_response", 1),
        )
    elif op_type == "score":
        # Check if scoring function name is provided (for sorting etc.)
        scoring_fn = None
        fn_name = params.get("scoring_function")
        if fn_name == "sorting_errors":
            from examples.sorting.utils import num_errors

            scoring_fn = num_errors
        elif fn_name == "keyword_counting_errors":
            # keyword counting errors needs parameters, can use wrapper
            from functools import partial

            from examples.keyword_counting.keyword_counting import num_errors

            # Default to some lists
            scoring_fn = partial(num_errors, [])

        op = operations.Score(
            num_samples=params.get("num_samples", 1),
            combined_scoring=params.get("combined_scoring", False),
            scoring_function=scoring_fn,
        )
    elif op_type == "keep_best_n":
        op = operations.KeepBestN(
            n=params.get("n", 1),
            higher_is_better=params.get("higher_is_better", True),
        )
    elif op_type == "keep_valid":
        op = operations.KeepValid()
    elif op_type == "aggregate":
        op = operations.Aggregate(
            num_responses=params.get("num_responses", 1),
        )
    elif op_type == "improve":
        op = operations.Improve()
    elif op_type == "validate_and_improve":
        op = operations.ValidateAndImprove(
            num_samples=params.get("num_samples", 1),
            improve=params.get("improve", True),
            num_tries=params.get("num_tries", 3),
        )
    elif op_type == "ground_truth":
        eval_fn = None
        fn_name = params.get("eval_function")
        if fn_name == "test_sorting":
            from examples.sorting.utils import test_sorting

            eval_fn = test_sorting
        elif fn_name == "test_keyword_counting":
            from examples.keyword_counting.keyword_counting import test_keyword_counting

            eval_fn = test_keyword_counting

        # fallback dummy function
        if eval_fn is None:

            def default_eval_fn(x):
                return True

            eval_fn = default_eval_fn

        op = operations.GroundTruth(ground_truth_evaluator=eval_fn)
    else:
        raise ValueError(f"Unknown operation type: {op_type}")

    # Set predecessors if any
    if predecessor_ids:
        for pred_id in predecessor_ids:
            if pred_id not in ops_map:
                raise ValueError(
                    f"Predecessor operation {pred_id} not found in this session."
                )
            op.add_predecessor(ops_map[pred_id])

    # Add to graph
    graph.add_operation(op)
    ops_map[client_op_id] = op

    return f"Successfully added operation {client_op_id} of type {op_type}."


@mcp.tool()
async def run_got_session(session_id: str) -> dict:
    """
    Run the Graph of Thoughts session asynchronously and return leaf thoughts and cost details.

    :param session_id: The ID of the session.
    :return: A dictionary of results and stats.
    """
    if session_id not in sessions:
        raise ValueError(f"Session {session_id} not found.")

    session = sessions[session_id]
    graph = session["graph"]
    lm = session["lm"]
    prompter = session["prompter"]
    parser = session["parser"]
    initial_params = session["initial_params"]

    # Instantiate AsyncController and run it
    ctrl = controller.AsyncController(
        lm=lm,
        graph=graph,
        prompter=prompter,
        parser=parser,
        problem_parameters=initial_params,
    )

    logger.info("Running GoT session: %s", session_id)
    await ctrl.run()

    # Gather final thoughts
    final_thoughts = []
    for leaf in graph.leaves:
        final_thoughts.append(
            {
                "operation_id": leaf.id,
                "operation_type": (
                    leaf.operation_type.name if leaf.operation_type else "unknown"
                ),
                "thoughts": [thought.state for thought in leaf.get_thoughts()],
                "scores": [
                    thought.score for thought in leaf.get_thoughts() if thought.scored
                ],
                "solved": [
                    thought.solved
                    for thought in leaf.get_thoughts()
                    if thought.compared_to_ground_truth
                ],
            }
        )

    result = {
        "final_thoughts": final_thoughts,
        "prompt_tokens": lm.prompt_tokens,
        "completion_tokens": lm.completion_tokens,
        "cost": lm.cost,
    }

    # Remove session from registry to free memory
    del sessions[session_id]

    return result


@mcp.tool()
async def execute_got_graph(
    initial_parameters: dict,
    graph_def: dict,
    task_name: str = "custom",
    config_path: str = "",
    model_name: str = "",
    templates: Optional[dict] = None,
    parser_spec: Optional[dict] = None,
) -> dict:
    """
    Execute a Graph of Thoughts graph in a single stateless tool call.

    :param initial_parameters: Initial variables for the thought state.
    :param graph_def: Definition of the graph in a format like:
                      {
                          "nodes": [
                              {"id": "gen1", "type": "generate", "params": {"num_branches_prompt": 1, "num_branches_response": 1}},
                              {"id": "score1", "type": "score", "predecessors": ["gen1"], "params": {"scoring_function": "sorting_errors"}}
                          ]
                      }
    :param task_name: Built-in task name or "custom".
    :param config_path: Path to config.json.
    :param model_name: Name of model in config.json.
    :param templates: Dict of custom templates.
    :param parser_spec: Dict of parser specifications.
    :return: A dictionary of results and stats.
    """
    # Create temporary session
    import os

    if not model_name:
        model_name = os.getenv("GOT_LANGUAGE_MODEL", "chatgpt")

    session_id = await create_got_session(
        initial_parameters=initial_parameters,
        task_name=task_name,
        config_path=config_path,
        model_name=model_name,
        templates=templates,
        parser_spec=parser_spec,
    )

    # Add operations
    nodes = graph_def.get("nodes", [])
    for node in nodes:
        node_id = node["id"]
        node_type = node["type"]
        predecessors = node.get("predecessors", [])
        params = node.get("params", {})

        await add_got_operation(
            session_id=session_id,
            op_type=node_type,
            client_op_id=node_id,
            predecessor_ids=predecessors,
            params=params,
        )

    # Run session and return results
    return await run_got_session(session_id)


@mcp.tool()
async def got_get_prompt(
    prompt_type: str,
    task_name: str = "custom",
    variables: Optional[dict] = None,
    templates: Optional[dict] = None,
) -> str:
    """
    Get a formatted prompt for client-side LLM execution.

    :param prompt_type: Type of prompt ("generate", "score", "aggregation", "improve", "validation").
    :param task_name: Task name ("sorting", "keyword_counting", "set_intersection", "doc_merge") or "custom".
    :param variables: Dict containing variables for the prompt (e.g. {"original": "[2, 1]", "current": "", "method": "got"}).
    :param templates: Optional dict containing custom templates if task_name is "custom".
    """
    variables = variables or {}
    prompter, _ = get_task_prompter_parser(task_name, templates=templates)

    p_type = prompt_type.lower()
    if p_type == "generate":
        num_branches = variables.pop("num_branches", 1)
        # some prompters expect original, current, method
        return prompter.generate_prompt(num_branches, **variables)
    elif p_type == "score":
        # pop so state_dicts is not also splatted via **variables (would
        # raise "got multiple values for argument 'state_dicts'").
        state_dicts = variables.pop("state_dicts", [])
        return prompter.score_prompt(state_dicts, **variables)
    elif p_type == "aggregation":
        state_dicts = variables.pop("state_dicts", [])
        return prompter.aggregation_prompt(state_dicts, **variables)
    elif p_type == "improve":
        return prompter.improve_prompt(**variables)
    elif p_type == "validation":
        return prompter.validation_prompt(**variables)
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")


@mcp.tool()
async def got_parse_response(
    parse_type: str,
    responses: List[str],
    task_name: str = "custom",
    variables: Optional[dict] = None,
    parser_spec: Optional[dict] = None,
) -> Union[List[dict], List[float], dict, bool]:
    """
    Parse LLM response texts for client-side LLM execution.

    :param parse_type: Type of parsing ("generate", "score", "aggregation", "improve", "validation").
    :param responses: List of raw string responses from the LLM.
    :param task_name: Task name ("sorting", "keyword_counting", "set_intersection", "doc_merge") or "custom".
    :param variables: Dict containing variables / input states used to generate the prompt (e.g. {"state": {...}} or {"states": [...]}).
    :param parser_spec: Optional dict containing custom parsing specification if task_name is "custom".
    """
    variables = variables or {}
    _, parser = get_task_prompter_parser(task_name, spec=parser_spec)

    # NOTE: several built-in parsers (sorting, keyword_counting,
    # set_intersection) intentionally leave parse_score_answer unimplemented
    # (they score with deterministic functions, not by parsing text), so it
    # returns None. The tool's return type cannot be None, so we coerce None
    # results to a sensible empty value per parse type. For text-based scoring
    # use the "doc_merge" task or a custom task with a parser_spec.
    p_type = parse_type.lower()
    if p_type == "generate":
        state = variables.get("state", {})
        return parser.parse_generate_answer(state, responses) or []
    elif p_type == "score":
        states = variables.get("states", [])
        return parser.parse_score_answer(states, responses) or []
    elif p_type == "aggregation":
        states = variables.get("states", [])
        result = parser.parse_aggregation_answer(states, responses)
        return result if result is not None else {}
    elif p_type == "improve":
        state = variables.get("state", {})
        return parser.parse_improve_answer(state, responses) or {}
    elif p_type == "validation":
        state = variables.get("state", {})
        return bool(parser.parse_validation_answer(state, responses))
    else:
        raise ValueError(f"Unknown parse type: {parse_type}")


if __name__ == "__main__":
    mcp.run()
