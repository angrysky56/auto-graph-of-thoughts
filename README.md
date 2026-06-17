# Graph of Thoughts (GoT)

<p align="center">
  <img src="paper/pics/preview.svg">
</p>

This is the official implementation of **[Graph of Thoughts: Solving Elaborate Problems with Large Language Models](https://arxiv.org/pdf/2308.09687.pdf)**.

This modernized version enables solving complex problems by modeling them as a **Graph of Operations (GoO)**, which can be executed concurrently via an asynchronous engine. It features support for **OpenRouter** and **OpenAI**, and exposes a **Model Context Protocol (MCP)** server so that clients (like Claude, Gemini, or other agents) can dynamically build and execute graphs.

---

## Key Modernizations

1. **Asynchronous Execution Engine (`AsyncController`)**:
   Reasoning is modeled as a Directed Acyclic Graph (DAG). The framework now features an async scheduler that executes independent operations concurrently using `asyncio`, drastically reducing latency.
2. **OpenRouter Support**:
   Exposes first-class support for OpenRouter, enabling access to Llama 3, Claude 3.5, Gemini 1.5, and thousands of other open-source and proprietary models through a single API connection.
3. **Model Context Protocol (MCP) Server**:
   Exposes tools for both server-side execution and client-side prompt formatting/parsing, allowing agents to execute cognitive graphs dynamically.
4. **Python 3.10+ & package updates**:
   Fully compatible with Python 3.10 through 3.13, using `uv` for virtual environment management, modern type hints, and formatted/linted using Ruff.

---

## Setup Guide

To use this framework, you need to have a working installation of Python 3.10 or newer (tested up to 3.13). We recommend using `uv`.

### Installing GoT

Clone the repository and install in editable mode:

```bash
git clone https://github.com/angrysky56/auto-graph-of-thoughts.git
cd auto-graph-of-thoughts
uv venv
uv pip install -e .
```

If you want to use the HuggingFace local LLaMA execution (heavyweight), install optional dependencies:

```bash
uv pip install -e .[hf]
```

### Configuring the LLM

We use `python-dotenv` to manage configurations. You can configure your environment in two ways:

1. **OS Environment Variables**: Simply export them in your shell (e.g. `~/.bashrc`):

   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   export GOT_LANGUAGE_MODEL="openrouter"
   ```

2. **.env File**: Create a `.env` file in the project root:
   ```env
   OPENROUTER_API_KEY=your-api-key
   OPENROUTER_MODEL_ID=meta-llama/llama-3-70b-instruct
   GOT_LANGUAGE_MODEL=openrouter
   ```

You can configure more complex model specifics (e.g. costs, parameters) in `config.json` (see `graph_of_thoughts/language_models/config_template.json` for structure), but for standard usage, the `.env` approach is recommended.

---

## Model Context Protocol (MCP) Server

Graph of Thoughts can run as an MCP server, exposing tools to your AI assistant. An example setup file is provided in [mcp_config.json](mcp_config.json).

### Running the Server (Non-MCP Config Usage)

Start the server using `uv`:

```bash
uv run python -m graph_of_thoughts.mcp_server
```

### Server Execution Modes

The MCP server supports two different ways to handle the actual "thinking" (the LLM calls).

#### 1. The MCP Server Calls the LLM (Requires API Keys)

In this mode, the GoT framework handles everything. It builds the graph and makes the network requests to OpenRouter/OpenAI using the API keys you provided in your `.env`.

- **Stateless Tool (`execute_got_graph`)**: Send a complete JSON DAG definition to the server. The server runs the entire graph, queries the LLM, and returns the final result.
- **Stateful Tools (`create_got_session`, `add_got_operation`, `run_got_session`)**: Build graphs node-by-node programmatically, then tell the server to run them using its own LLM connection.

#### 2. The AI Assistant Acts as the LLM (No API Keys Required on Server)

If you (or your AI assistant) don't want the server to make its own API calls, the AI assistant can act as the LLM itself! The server will just provide the formatting and parsing logic.

- **Prompt Helper (`got_get_prompt`)**: The AI assistant asks the server "What prompt should I use to generate the next thought?" The server returns the formatted text.
- **Parse Helper (`got_parse_response`)**: After the AI assistant generates the thoughts using its own internal model, it passes the raw text back to the server. The server parses the text and extracts the structured scores/updates.

---

## Quick Start (Async Python API)

Here is a quick example of running a sorting problem with 32 numbers using the async controller:

```python
import asyncio
from examples.sorting.sorting_032 import SortingPrompter, SortingParser, got, utils
from graph_of_thoughts import controller, language_models

async def main():
    # Problem input
    to_be_sorted = "[0, 2, 6, 3, 8, 7, 1, 1, 6, 7, 7, 7, 7, 9, 3, 0, 1, 7, 9, 1, 3, 5, 1, 3, 6, 4, 5, 4, 7, 3, 5, 7]"

    # Retrieve the Graph of Operations
    gop = got()

    # Configure OpenRouter model (reads from config.json or environment)
    lm = language_models.OpenRouter("config.json", model_name="openrouter")

    # Create the AsyncController
    ctrl = controller.AsyncController(
        lm,
        gop,
        SortingPrompter(),
        SortingParser(),
        {
            "original": to_be_sorted,
            "current": "",
            "phase": 0,
            "method": "got"
        }
    )

    # Run concurrently and output
    await ctrl.run()
    ctrl.output_graph("output_got.json")

    print("Execution completed! Spent:", lm.cost)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Examples

The [examples](examples) directory contains tasks solved via GoT:

- **Sorting**: Split, sort, and merge-sort numbers.
- **Keyword Counting**: Split a text, count keywords, and merge frequency dictionaries.
- **Set Intersection**: Find common elements between lists.
- **Doc Merge**: Aggregating documents.

Run examples directly:

```bash
python -m examples.sorting.sorting_032
python -m examples.keyword_counting.keyword_counting
```

---

## Use Cases (for AI agents via MCP)

Once the MCP server is connected, an agent (Claude, Gemini, etc.) can treat GoT
as a **reasoning subroutine**: instead of answering a hard problem in one shot,
it builds a graph that generates many candidates, scores them, prunes to the
best, and merges the survivors. The two modes make this useful in different
situations:

- **Client-side (no API key, no cost)** — the agent is the LLM. It calls
  `got_get_prompt` to get a formatted step, produces the thought with its own
  model, then `got_parse_response` to structure the result. Use this to impose a
  disciplined branch/score/prune loop on the agent's _own_ reasoning.
- **Server-side (uses your `.env` key)** — the engine runs the whole DAG
  concurrently against any OpenRouter/OpenAI model via `execute_got_graph` or a
  stateful session. Use this to offload many branches to a cheaper/faster model
  and get back a single scored, merged answer.

### What it's good for

- **Strategy / option generation with scoring.** Generate N distinct approaches
  to a problem, have the model score each on your own rubric (e.g.
  _impact × feasibility_), keep the top few, then aggregate them into one plan.
  A complete, runnable example ships at
  [`examples/custom_tasks/rag_antihallucination.json`](examples/custom_tasks/rag_antihallucination.json)
  (`generate×4 → score → keep_best_n(2) → aggregate → improve`).
- **Decompose → solve → recombine over large inputs.** Split a long document or
  list into chunks, solve each branch independently and in parallel, then
  merge — the original paper's wins on sorting, keyword counting, and set
  intersection are this pattern.
- **Answer / document synthesis.** Produce several drafts, score for accuracy or
  coverage, and `aggregate` the best into a single consolidated output
  (see the `doc_merge` task).
- **Self-critique and refinement loops.** Chain `validate_and_improve` or
  `improve` nodes so a draft is checked and rewritten until it passes, instead
  of trusting a single pass.
- **Verifiable tasks with ground truth.** Attach a `ground_truth` evaluator so
  the graph reports whether each surviving thought is actually correct — useful
  for benchmarking a model or gating an automated pipeline.

### Minimal agent recipe

Write a `generate` template that asks for **one** candidate, a `score` template
that demands a **bare integer**, then wire:

```
generate (×N)  →  score  →  keep_best_n  →  aggregate  →  improve
```

Pass it to `execute_got_graph` with `task_name: "custom"`. For deeper guidance
on authoring templates and parser specs, see the bundled agent skill at
[`.claude/skills/graph-of-thoughts/SKILL.md`](.claude/skills/graph-of-thoughts/SKILL.md).

> Note on scoring: the built-in `sorting`, `keyword_counting`, and
> `set_intersection` tasks score with deterministic functions, so their text
> `score` prompts are intentionally absent. For LLM-judged scoring, use
> `doc_merge` or a `custom` task (as in the example above).

---

## Citations

If you find this repository valuable, please cite the AAAI paper:

```bibtex
@article{besta2024got,
  title = {{Graph of Thoughts: Solving Elaborate Problems with Large Language Models}},
  author = {Besta, Maciej and Blach, Nils and Kubicek, Ales and Gerstenberger, Robert and Gianinazzi, Lukas and Gajda, Joanna and Lehmann, Tomasz and Podstawski, Micha{\l} and Niewiadomski, Hubert and Nyczyk, Piotr and Hoefler, Torsten},
  year = 2024,
  month = {Mar},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = 38,
  number = 16,
  pages = {17682-17690},
  publisher = {AAAI Press},
  doi = {10.1609/aaai.v38i16.29720},
  url = {https://ojs.aaai.org/index.php/AAAI/article/view/29720}
}
```
