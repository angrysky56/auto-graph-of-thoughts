# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Nils Blach

"""
Document Merge example utilizing the Graph of Thoughts framework.
This module executes various methods (IO, CoT, ToT, GoT) to merge multiple NDA documents.
"""

import csv
import datetime
import json
import logging
import os
import re
from statistics import fmean
from typing import Callable, Dict, List, Union

from graph_of_thoughts import controller, language_models, operations, parser, prompter


class DocMergePrompter(prompter.Prompter):
    """
    DocMergePrompter provides the generation of prompts specific to the document
    merge example for the language models.

    Inherits from the Prompter class and implements its abstract methods.
    """

    merge_doc_prompt_start = (
        "Merge the following {num} NDA documents <Doc1> - <Doc{num}> into a single NDA, "
        "maximizing retained information and minimizing redundancy. Output only the created "
        "NDA between the tags <Merged> and </Merged>, without any additional text.\n"
        "Here are NDAs <Doc1> - <Doc{num}>\n"
    )
    merge_doc_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    merge_doc_prompt_cot_start = (
        "Merge the following {num} NDA documents <Doc1> - <Doc{num}> into a single NDA, "
        "maximizing retained information and minimizing redundancy.\n"
        "You can generate any intermediate thoughts and documents you want, but the final output "
        "should be the merged NDA, placed between the two tags <Merged> and </Merged>.\n"
        "For instance you might want to follow this approach:\n"
        "1. Split each NDA into their logical subparts.\n"
        "2. Merge the subparts of the {num} NDAs.\n"
        "3. Combine the merged subparts into a single NDA.\n"
        "4. Place the merged NDA between the tags <Merged> and </Merged>.\n\n"
        "Here are NDAs <Doc1> - <Doc{num}>:\n"
    )

    improve_summary_prompt_start = """The following NDA <S> merges initial NDAs <Doc1> - <Doc{num}>.
Please improve the summary NDA <S> by adding more information and removing redundancy. Output only the improved NDA, placed between the two tags <Merged> and </Merged>, without any additional text.

Here are NDAs <Doc1> - <Doc{num}>:
"""

    improve_summary_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    improve_summary_prompt_end = """
Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

    score_prompt_base = """The following NDA <S> merges NDAs <Doc1> - <Doc{num}>.
Please score the merged NDA <S> in terms of how much redundant information is contained, independent of the original NDAs, as well as how much information is retained from the original NDAs.
A score of 10 for redundancy implies that absolutely no information is redundant, while a score of 0 implies that at least half of the information is redundant (so everything is at least mentioned twice).
A score of 10 for retained information implies that all information from the original NDAs is retained, while a score of 0 implies that no information is retained.
You may provide reasoning for your scoring, but the final score for redundancy should be between the tags <Redundancy> and </Redundancy>, and the final score for retained information should be between the tags <Retained> and </Retained>, without any additional text within any of those tags.

Here are NDAs <Doc1> - <Doc{num}>:
"""

    score_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    score_prompt_end = """
Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

    validation_prompt_base = (
        "The following NDA <S> merges NDAs <Doc1> - <Doc{num}>.\n"
        "Please evaluate if the merged NDA <S> is valid. A merged NDA is valid if it "
        "represents a coherent agreement and retains the core terms of the original NDAs.\n"
        "Output only <Valid>YES</Valid> if it is valid, and <Valid>NO</Valid> if it is invalid, "
        "without any additional text.\n\n"
        "Here are NDAs <Doc1> - <Doc{num}>:\n"
    )

    validation_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    validation_prompt_end = """
Here is the summary NDA <S>:
<S>
{summary}
</S>
"""

    improve_prompt_base = (
        "The following NDA <S> merges NDAs <Doc1> - <Doc{num}>.\n"
        "Please improve the merged NDA <S> to ensure it represents a coherent agreement, "
        "removes any redundant sections, and retains all key details of the original NDAs.\n"
        "Output only the improved NDA between the tags <Merged> and </Merged>, "
        "without any additional text.\n\n"
        "Here are NDAs <Doc1> - <Doc{num}>:\n"
    )

    improve_prompt_block = """
<Doc{num}>
{document}
</Doc{num}>
"""

    improve_prompt_end = """
Here is the current summary NDA <S>:
<S>
{summary}
</S>
"""

    aggregate_full_prompt_base = (
        "The following NDAs <S1> - <S{num_ndas_summary}> each merge the initial NDAs "
        "<Doc1> - <Doc{num_ndas}>.\n"
        "Combine the merged NDAs <S1> - <S{num_ndas_summary}> into a new one, maximizing "
        "their advantages and overall information retention, while minimizing redundancy.\n"
        "Output only the new NDA between the tags <Merged> and </Merged>, without any "
        "additional text.\n\n"
        "Here are the original NDAs <Doc1> - <Doc{num_ndas}>:\n"
    )

    aggregate_full_prompt_block1 = """
<Doc{num}>
{document}
</Doc{num}>
"""
    aggregate_full_prompt_mid = """
Here are the summary NDAs <S1> - <S{num_ndas_summary}>:
"""

    aggregate_full_prompt_block2 = """
<S{num}>
{summary}
</S{num}>
"""

    aggregate_sub_prompt_base = (
        "The following NDAs <S1> - <S{num_ndas}> are summaries of some other NDAs.\n"
        "Combine them into a new one, make sure to maximize their advantages and overall "
        "information retention, while minimizing redundancy.\n"
        "Output only the new NDA between the tags <Merged> and </Merged>, without any "
        "additional text.\n\n"
        "Here are NDAs <S1> - <S{num_ndas}>:\n"
    )

    aggregate_sub_prompt_generate = """
NDA <S{num}>:
{nda}
</S{num}>
"""

    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate an aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        """

        if len(state_dicts[0]["parts"]) > 0 and len(state_dicts[0]["parts"]) < len(
            state_dicts[0]["documents"]
        ):
            prompt = self.aggregate_sub_prompt_base.format(
                num_ndas=len(state_dicts),
            )
            for i, state_dict in enumerate(state_dicts):
                prompt += self.aggregate_sub_prompt_generate.format(
                    nda=state_dict["current"], num=i + 1
                )
            return prompt
        else:
            prompt = self.aggregate_full_prompt_base.format(
                num_ndas=len(state_dicts[0]["documents"]),
                num_ndas_summary=len(state_dicts),
            )
            for i, document in enumerate(state_dicts[0]["documents"]):
                prompt += self.aggregate_full_prompt_block1.format(
                    document=document, num=i + 1
                )
            prompt += self.aggregate_full_prompt_mid.format(
                num_ndas_summary=len(state_dicts),
            )
            for i, state_dict in enumerate(state_dicts):
                prompt += self.aggregate_full_prompt_block2.format(
                    summary=state_dict["current"], num=i + 1
                )
            return prompt

    def generate_prompt(
        self,
        num_branches: int,
        **kwargs,
    ) -> str:
        """
        Generate a generate prompt for the language model.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param kwargs: Additional keyword arguments containing:
                       - documents: List[str]
                       - method: str
                       - parts: Set[str]
                       - current: str
        :return: The generate prompt.
        :rtype: str
        :raise AssertionError: If method is not implemented yet.
        """

        documents = kwargs.get("documents")
        method = kwargs.get("method")
        parts = kwargs.get("parts")
        current = kwargs.get("current")

        prompt = ""
        if method.startswith("io") or method.startswith("cot"):
            if method.startswith("io"):
                prompt += self.merge_doc_prompt_start.format(num=len(documents))
            else:
                prompt += self.merge_doc_prompt_cot_start.format(num=len(documents))
            for i, document in enumerate(documents):
                prompt += self.merge_doc_prompt_block.format(
                    document=document, num=i + 1
                )
            return prompt
        elif method.startswith("tot"):
            if current is None or current == "":
                prompt += self.merge_doc_prompt_start.format(num=len(documents))
                for i, document in enumerate(documents):
                    prompt += self.merge_doc_prompt_block.format(
                        document=document, num=i + 1
                    )
                return prompt
            else:
                prompt += self.improve_summary_prompt_start.format(
                    num=len(documents),
                )
                for i, document in enumerate(documents):
                    prompt += self.improve_summary_prompt_block.format(
                        document=document, num=i + 1
                    )
                prompt += self.improve_summary_prompt_end.format(summary=current)
                return prompt
        elif method.startswith("got"):
            parts = (
                sorted(list(parts)) if len(parts) > 0 else list(range(len(documents)))
            )
            if current is None or current == "":
                prompt += self.merge_doc_prompt_start.format(num=len(parts))
                for i, part in enumerate(sorted(list(parts))):
                    prompt += self.merge_doc_prompt_block.format(
                        document=documents[part], num=i + 1
                    )
                return prompt
            else:
                prompt += self.improve_summary_prompt_start.format(
                    num=len(parts),
                )
                for i, part in enumerate(sorted(list(parts))):
                    prompt += self.improve_summary_prompt_block.format(
                        document=documents[part], num=i + 1
                    )
                prompt += self.improve_summary_prompt_end.format(summary=current)
                return prompt
        else:
            raise AssertionError("Not implemented yet.")

    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        :raise AssertionError: If more than one thought state is supplied.
        """

        if len(state_dicts) > 1:
            raise AssertionError("Not implemented yet.")
        else:
            # perform individual scoring
            parts = (
                [
                    state_dicts[0]["documents"][part]
                    for part in sorted(list(state_dicts[0]["parts"]))
                ]
                if len(state_dicts[0]["parts"]) > 0
                else state_dicts[0]["documents"]
            )
            prompt = self.score_prompt_base.format(
                num=len(parts),
            )
            for i, part in enumerate(parts):
                prompt += self.score_prompt_block.format(document=part, num=i + 1)
            prompt += self.score_prompt_end.format(
                summary=state_dicts[0]["current"],
            )
            return prompt

    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.

        :param kwargs: Additional keyword arguments containing:
                       - documents: List[str]
                       - current: str
                       - parts: Set[str]
        :return: The improve prompt.
        :rtype: str
        """
        documents = kwargs.get("documents")
        current = kwargs.get("current")
        parts = kwargs.get("parts")

        parts_list = (
            [documents[part] for part in sorted(list(parts))]
            if parts and len(parts) > 0
            else documents
        )

        prompt = self.improve_prompt_base.format(num=len(parts_list))
        for i, part in enumerate(parts_list):
            prompt += self.improve_prompt_block.format(document=part, num=i + 1)
        prompt += self.improve_prompt_end.format(summary=current)
        return prompt

    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.

        :param kwargs: Additional keyword arguments containing:
                       - documents: List[str]
                       - current: str
                       - parts: Set[str]
        :return: The validation prompt.
        :rtype: str
        """
        documents = kwargs.get("documents")
        current = kwargs.get("current")
        parts = kwargs.get("parts")

        parts_list = (
            [documents[part] for part in sorted(list(parts))]
            if parts and len(parts) > 0
            else documents
        )

        prompt = self.validation_prompt_base.format(num=len(parts_list))
        for i, part in enumerate(parts_list):
            prompt += self.validation_prompt_block.format(document=part, num=i + 1)
        prompt += self.validation_prompt_end.format(summary=current)
        return prompt


class DocMergeParser(parser.Parser):
    """
    DocMergeParser provides the parsing of language model reponses specific to the
    document merge example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

    def strip_answer_helper(self, text: str, tag: str = "") -> str:
        """
        Helper function to remove tags from a text.

        :param text: The input text.
        :type text: str
        :param tag: The tag to be stripped. Defaults to "".
        :type tag: str
        :return: The stripped text.
        :rtype: str
        """

        text = text.strip()
        if "Output:" in text:
            text = text[text.index("Output:") + len("Output:") :].strip()
        if tag != "":
            start = text.rfind(f"<{tag}>")
            end = text.rfind(f"</{tag}>")
            if start != -1 and end != -1:
                text = text[start + len(f"<{tag}>") : end].strip()
            elif start != -1:
                logging.warning(
                    "Only found the start tag <%s> in answer: %s. Returning everything after the tag.",
                    tag,
                    text,
                )
                text = text[start + len(f"<{tag}>") :].strip()
            elif end != -1:
                logging.warning(
                    "Only found the end tag </%s> in answer: %s. Returning everything before the tag.",
                    tag,
                    text,
                )
                text = text[:end].strip()
            else:
                logging.warning(
                    "Could not find any tag %s in answer: %s. Returning the full answer.",
                    tag,
                    text,
                )
        return text

    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for an aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: Union[Dict, List[Dict]]
        """

        new_states = []
        for text in texts:
            if len(states[0]["parts"]) < len(states[0]["documents"]):
                # subpart aggregation
                text = self.strip_answer_helper(text, "Merged")
                new_state = states[0].copy()
                new_state["current"] = text
                new_state["parts"] = set()
                for state in states:
                    new_state["parts"] = new_state["parts"] | state["parts"]

                new_states.append(new_state)
            else:
                # full NDA aggregation
                text = self.strip_answer_helper(text, "Merged")
                new_state = states[0].copy()
                new_state["current"] = text
                new_states.append(new_state)
        return new_states

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """
        new_states = []
        for text in texts:
            text = self.strip_answer_helper(text, "Merged")
            new_state = state.copy()
            new_state["current"] = text
            new_states.append(new_state)
        return new_states

    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        :raise AssertionError: If the number of thought states is not one.
        """
        if len(states) != 1:
            raise AssertionError("Only one state is allowed for scoring.")
        if len(states) == 1:
            # individual scoring
            redundancy_scores = []
            retain_scores = []
            for text in texts:
                answer = self.strip_answer_helper(text, "Redundancy")
                res = re.findall(r"\d+\.?\d*", answer)
                if len(res) == 1:
                    redundancy_scores.append(float(res[0]))
                elif len(res) > 1:
                    logging.warning(
                        "Found multiple redundancy scores in answer: %s. Returning the last one.",
                        text,
                    )
                    redundancy_scores.append(float(res[-1]))
                else:
                    logging.warning(
                        "Could not find any redundancy score in answer: %s. Ignoring this answer.",
                        text,
                    )
                answer = self.strip_answer_helper(text, "Retained")
                res = re.findall(r"\d+\.?\d*", answer)
                if len(res) == 1:
                    retain_scores.append(float(res[0]))
                elif len(res) > 1:
                    logging.warning(
                        "Found multiple retained scores in answer: %s. Returning the last one.",
                        text,
                    )
                    retain_scores.append(float(res[-1]))
                else:
                    logging.warning(
                        "Could not find any retained score in answer: %s. Ignoring this answer.",
                        text,
                    )
            if len(redundancy_scores) == 0 or len(retain_scores) == 0:
                logging.warning(
                    "Could not find any valid score in any answer. Returning 0.0."
                )
                return [0.0]
            mean_redundancy = fmean(redundancy_scores)
            mean_retain = fmean(retain_scores)
            f1 = 2 * mean_redundancy * mean_retain / (mean_redundancy + mean_retain)
            return [f1]

    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the responses from the language model.
        :rtype: Dict
        """
        if len(texts) != 1:
            raise AssertionError("Expected exactly 1 text for improve answer.")
        text = self.strip_answer_helper(texts[0], "Merged")
        new_state = state.copy()
        new_state["current"] = text
        return new_state

    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        for text in texts:
            answer = self.strip_answer_helper(text, "Valid")
            if "YES" in answer.upper():
                return True
        return False


def io() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the IO method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(3, False))

    return operations_graph


def cot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the CoT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 1))
    operations_graph.append_operation(operations.Score(3, False))

    return operations_graph


def tot() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the ToT method.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    branch_factor = 10

    operations_graph.append_operation(operations.Generate(1, branch_factor))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best_1 = operations.KeepBestN(1, True)
    operations_graph.append_operation(keep_best_1)

    for _ in range(2):
        operations_graph.append_operation(operations.Generate(1, branch_factor))
        operations_graph.append_operation(operations.Score(3, False))
        keep_best_2 = operations.KeepBestN(1, True)
        keep_best_2.add_predecessor(keep_best_1)
        operations_graph.append_operation(keep_best_2)
        keep_best_1 = keep_best_2

    return operations_graph


def got() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT method, where full documents
    are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    operations_graph.append_operation(operations.Generate(1, 5))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best = operations.KeepBestN(3, True)
    operations_graph.append_operation(keep_best)
    operations_graph.append_operation(operations.Aggregate(5))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best2 = operations.KeepBestN(1, True)
    keep_best2.add_predecessor(keep_best)
    operations_graph.append_operation(keep_best2)
    operations_graph.append_operation(operations.Generate(1, 10))
    operations_graph.append_operation(operations.Score(3, False))
    keep_best3 = operations.KeepBestN(1, True)
    keep_best3.add_predecessor(keep_best2)
    operations_graph.append_operation(keep_best3)

    return operations_graph


def got2() -> operations.GraphOfOperations:
    """
    Generates the Graph of Operations for the GoT2 method, where partial
    documents are merged.

    :return: Graph of Operations
    :rtype: GraphOfOperations
    """
    operations_graph = operations.GraphOfOperations()

    sub_parts = []
    for i in range(0, 4, 2):  # should be at most 16 parts
        sub_text = operations.Selector(
            lambda thoughts, list_id=i: [
                operations.Thought(
                    state={**thoughts[0].state, "parts": {list_id, list_id + 1}}
                )
            ]
        )
        operations_graph.add_operation(sub_text)
        gen_nda = operations.Generate(1, 5)
        gen_nda.add_predecessor(sub_text)
        operations_graph.add_operation(gen_nda)
        score_nda = operations.Score(3, False)
        score_nda.add_predecessor(gen_nda)
        operations_graph.add_operation(score_nda)
        keep_best_nda = operations.KeepBestN(1, True)
        keep_best_nda.add_predecessor(score_nda)
        operations_graph.add_operation(keep_best_nda)

        sub_parts.append(keep_best_nda)

    while len(sub_parts) > 1:
        new_sub_parts = []
        for i in range(0, len(sub_parts), 2):
            if i + 1 == len(sub_parts):
                new_sub_parts.append(sub_parts[i])
                continue
            aggregate = operations.Aggregate(5)
            aggregate.add_predecessor(sub_parts[i])
            aggregate.add_predecessor(sub_parts[i + 1])
            operations_graph.add_operation(aggregate)
            score = operations.Score(3, False)
            score.add_predecessor(aggregate)
            operations_graph.add_operation(score)
            keep_best = operations.KeepBestN(1, True)
            keep_best.add_predecessor(score)
            operations_graph.add_operation(keep_best)

            gen_nda = operations.Generate(1, 5)
            gen_nda.add_predecessor(keep_best)
            operations_graph.add_operation(gen_nda)
            score_nda = operations.Score(3, False)
            score_nda.add_predecessor(gen_nda)
            operations_graph.add_operation(score_nda)
            keep_best_nda = operations.KeepBestN(1, True)
            keep_best_nda.add_predecessor(score_nda)
            keep_best_nda.add_predecessor(keep_best)
            operations_graph.add_operation(keep_best_nda)

            new_sub_parts.append(keep_best_nda)
        sub_parts = new_sub_parts

    return operations_graph


def run(
    data_ids: List[int],
    methods: List[Callable[[], operations.GraphOfOperations]],
    budget: float,
    lm_name: str,
) -> float:
    """
    Controller function that executes each specified method for each specified
    sample while the budget is not exhausted.

    :param data_ids: Indices of the sample to be run.
    :type data_ids: List[int]
    :param methods: List of functions to generate Graphs of Operations.
    :type methods: Each function generates a Graph of Operation.
    :param budget: Language model budget for the execution in dollars.
    :type budget: float
    :param lm_name: Name of the language model to be used.
    :type lm_name: str
    :return: Spent budget in dollars.
    :rtype: float
    """

    orig_budget = budget
    data_path = os.path.join(os.path.dirname(__file__), "documents.csv")
    data = []
    with open(data_path, "r", encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            row[0] = int(row[0])
            data.append(row)

    if data_ids is None or len(data_ids) == 0:
        data_ids = list(range(len(data)))
    selected_data = [data[i] for i in data_ids]

    results_dir = os.path.join(os.path.dirname(__file__), "results")

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    extra_info = f"{lm_name}_{'-'.join([method.__name__ for method in methods])}"
    folder_name = f"{extra_info}_{timestamp}"
    results_folder = os.path.join(results_dir, folder_name)
    os.makedirs(results_folder)

    config = {
        "data": selected_data,
        "methods": [method.__name__ for method in methods],
        "lm": lm_name,
        "budget": budget,
    }
    with open(os.path.join(results_folder, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)

    logging.basicConfig(
        filename=os.path.join(results_folder, "log.log"),
        filemode="w",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )

    for method in methods:
        os.makedirs(os.path.join(results_folder, method.__name__))

    for data in selected_data:
        logging.info("Running data %s: %s", data[0], data[1])
        if budget <= 0.0:
            logging.error(
                "Budget has been depleted, stopping. Data %s has not been run.",
                data[0],
            )
            break
        for method in methods:
            logging.info("Running method %s", method.__name__)
            logging.info("Budget left: %s", budget)
            if budget <= 0.0:
                logging.error(
                    "Budget has been depleted, stopping. Method %s has not been run.",
                    method.__name__,
                )
                break
            lm = language_models.ChatGPT(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../graph_of_thoughts/language_models/config.json",
                ),
                model_name=lm_name,
                cache=True,
            )
            operations_graph = method()
            executor = controller.Controller(
                lm,
                operations_graph,
                DocMergePrompter(),
                DocMergeParser(),
                {
                    "documents": [data[2], data[3], data[4], data[5]],
                    "parts": set(),
                    "current": "",
                    "method": method.__name__,
                },
            )
            try:
                executor.run()
            except Exception as e:  # pylint: disable=broad-exception-caught
                logging.error("Exception: %s", e)
            path = os.path.join(
                results_folder,
                method.__name__,
                f"{data[0]}.json",
            )
            for operation in operations_graph.operations:
                for thought in operation.thoughts:
                    thought.state["parts"] = list(thought.state["parts"])
            executor.output_graph(path)
            budget -= lm.cost

    return orig_budget - budget


if __name__ == "__main__":
    # Input (x1, x2, x3, x4): Four NDAs
    # Output (y): A new combined NDA
    # Evaluation: According to information coverage without repetition (scored by the LLM)
    BUDGET = 30
    samples = list(range(0, 50))
    approaches = [io, cot, tot, got, got2]

    spent = run(samples, approaches, BUDGET, "chatgpt")

    logging.info("Spent %s out of %s budget.", spent, BUDGET)
