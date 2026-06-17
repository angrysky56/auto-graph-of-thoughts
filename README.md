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

1. **OpenRouter**: Set the environment variable `OPENROUTER_API_KEY`:
   ```bash
   export OPENROUTER_API_KEY="your-api-key"
   ```
2. **OpenAI**: Set the environment variable `OPENAI_API_KEY`:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

You can configure model specifics (e.g. model ID, costs, temperature) in `config.json` (see `graph_of_thoughts/language_models/config_template.json` for structure).

---

## Model Context Protocol (MCP) Server

Graph of Thoughts can run as an MCP server, exposing tools to your AI assistant. An example setup file is provided in [mcp_config.json](file:///home/ty/Repositories/ai_workspace/auto-graph-of-thoughts/mcp_config.json).

### Running the Server

Start the server using `uv`:

```bash
uv run python -m graph_of_thoughts.mcp_server
```

### Server Execution Modes

The MCP server supports two execution modes:

#### 1. Server-Side Execution (Requires API keys on the server)

- **Stateless Tool (`execute_got_graph`)**: Send a complete JSON DAG definition and input variables, and receive the final solved state.
- **Stateful Tools (`create_got_session`, `add_got_operation`, `run_got_session`)**: Dynamically build graphs node-by-node (useful for agents constructing custom structures programmatically) and execute them.

#### 2. Client-Side Execution (No API keys or internet connection required by the server)

If the client agent (like Claude) wants to query the LLM itself (e.g., using its own API provider):

- **Prompt Helper (`got_get_prompt`)**: Get the formatted prompt (e.g., Generate, Score, Aggregate) for a specific step.
- **Parse Helper (`got_parse_response`)**: Pass the raw LLM output text back to the server to extract structured thought updates or validation scores.

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
