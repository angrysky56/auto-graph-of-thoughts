---
name: graph-of-thoughts
description: "Use the Graph of Thoughts (GoT) MCP server to solve elaborate problems by modeling reasoning as a Directed Acyclic Graph of operations (generate, score, keep-best, aggregate, improve, validate). Trigger when a task benefits from branching/exploring multiple candidate solutions, scoring and pruning them, and merging partial results — e.g. sorting/merging large lists, counting/aggregating over chunked text, set operations, document merging, or any decompose-solve-recombine workflow. Covers both server-side execution (the server calls the LLM) and client-side execution (the agent acts as the LLM via get_prompt/parse_response), plus authoring custom tasks with templates and parser specs."
---

# Graph of Thoughts (GoT) — agent usage

Model reasoning as a **Graph of Operations (GoO)**: a DAG whose nodes transform
"thoughts" (state dicts). The async engine runs independent branches
concurrently. Use GoT when single-shot prompting is weak but the problem
decomposes into *generate many candidates → score → prune → recombine*.

This skill targets the MCP server in this repo (`graph_of_thoughts.mcp_server`),
exposed with the `mcp__graph-of-thought__*` tools.

## Two execution modes — pick one first

1. **Server-side (server calls the LLM).** The GoT engine builds the graph and
   makes its own API calls (OpenRouter/OpenAI) using keys from `.env`. Use when
   you want a full automated run and the server has a valid key.
   Tools: `execute_got_graph` (one-shot) or `create_got_session` +
   `add_got_operation` + `run_got_session` (build node-by-node).

2. **Client-side (you, the agent, are the LLM).** The server only formats
   prompts and parses responses; *you* produce the "thoughts." No API key
   needed, no cost. Use when you want to keep reasoning inside this model, or
   the server has no key. Tools: `got_get_prompt` then `got_parse_response`.

## Tool quick reference

| Tool | Mode | Purpose |
|------|------|---------|
| `create_got_session` | server | Start a stateful session; returns `session_id`. |
| `add_got_operation` | server | Append one operation node to a session's graph. |
| `run_got_session` | server | Execute the session async; returns leaf thoughts + cost. Consumes (deletes) the session. |
| `execute_got_graph` | server | Stateless: pass a full `graph_def` + params, run it, return results. |
| `got_get_prompt` | client | Get the formatted prompt text for a step. |
| `got_parse_response` | client | Parse your raw response text into structured state/score. |

## Operation types (for `add_got_operation` / `graph_def` nodes)

`op_type` → params:

- `generate` — `num_branches_prompt`, `num_branches_response` (fan-out width).
- `score` — `num_samples`, `combined_scoring` (bool), `scoring_function`
  (string name; built-ins: `sorting_errors`, `keyword_counting_errors`).
- `keep_best_n` — `n`, `higher_is_better` (bool). For error-style scores set
  `higher_is_better: false` (fewer errors = better).
- `keep_valid` — drop invalid thoughts.
- `aggregate` — `num_responses` (merge predecessors' thoughts).
- `improve` — refine a thought.
- `validate_and_improve` — `num_samples`, `improve`, `num_tries`.
- `ground_truth` — `eval_function` (built-ins: `test_sorting`,
  `test_keyword_counting`); marks thoughts solved/unsolved.

**Canonical GoT shape:** `generate → score → keep_best_n → (aggregate | improve)
→ score → keep_best_n → ground_truth`.

### Principled selection ops (native, self-contained)

These upgrade GoT's weak default scoring. They need no extra models — they reuse
the graph's own LLM — and they judge at low temperature automatically.

- `rubric_score` — LLM rubric judge. Params: `criteria` (list of strings),
  `axis` (default `_quality`), `num_samples`. Asks for a per-criterion
  `[[YES]]/[[NO]]` verdict and parses it deterministically; writes the fraction
  met to `state[axis]`. A reliable **quality / convergent** axis (replaces the
  fragile "grab any number" parser).
- `novelty_score` — reference-free **novelty / divergent** axis. Clusters a
  node's sibling thoughts by bidirectional entailment and scores each by the
  normalised surprisal of its semantic class; writes to `state[axis]` (default
  `_novelty`). Run it directly on a `generate` node's branches.
- `keep_pareto` — multi-axis Pareto selection with a **convergent floor**
  (anti-Goodhart guard). Params: `axes` (default `["_novelty","_quality"]`),
  `floor_axis` (default `_quality`), `floor` (default `0.5`), `n` (optional cap).
  Drops anything below the floor, keeps the non-dominated frontier, never
  returns empty.

**Creative selection shape:** `generate(wide) → novelty_score → rubric_score →
keep_pareto → improve`. The floor stops novelty from being maximised into
incoherence; the Pareto frontier keeps genuinely different good answers rather
than collapsing to one. (Novelty method credited in-code to the ACL 2026
creativity-evaluation paper; implementation is original and API-only.)

### Deeper pipeline (concurrent, set-level) — preferred

The simple ops above judge candidates one at a time, which is a weak signal and
can time out (many serial calls). These three are async, parallelise their LLM
calls, and judge at the **set** level. Prefer them for real creative work.

- `divergent_generate` — explore/commit generator. Samples `k` candidates
  **concurrently**, clusters them in a single call, scores set-level novelty +
  entropy, and if the set collapsed onto one idea, **raises temperature and
  re-samples** (controller loop) before committing. Params: `k`, `max_rounds`,
  `base_temperature`, `temperature_step`, `diversity_threshold`. Fixes both the
  provider `n`-fan-out limit and mode collapse. Needs a `generate` template.
- `comparative_score` — scores all candidates **relative to each other in ONE
  call** (params: `criteria`, `scale`) → `_quality`. Stronger than per-candidate
  rubric, and one call instead of N.
- `multi_persona_judge` — final judge: several critic personas vote per
  criterion **concurrently**, aggregated by majority → `_quality` (params:
  `criteria`, `personas`). A self-contained stand-in for the paper's
  retrieval-based multi-agent judge (no embedding store).

**Preferred creative shape:** `divergent_generate → comparative_score →
keep_pareto → multi_persona_judge → improve`. Runnable example:
[`examples/custom_tasks/creative_selection.json`](../../../examples/custom_tasks/creative_selection.json).

Note: per-call cold judging needs the `temperature` override on the LM `query`
(present in this repo's `OpenRouter`/`ChatGPT`); older backends degrade silently.

## Server-side: stateless one-shot (`execute_got_graph`)

```jsonc
{
  "task_name": "sorting",
  "model_name": "openrouter",
  "initial_parameters": {"original": "[3,1,2,5,4,0,2,7]", "current": "", "phase": 0, "method": "io"},
  "graph_def": {"nodes": [
    {"id": "gen1",   "type": "generate",    "params": {"num_branches_prompt": 1, "num_branches_response": 3}},
    {"id": "score1", "type": "score",       "predecessors": ["gen1"],   "params": {"scoring_function": "sorting_errors"}},
    {"id": "kb1",    "type": "keep_best_n", "predecessors": ["score1"], "params": {"n": 1, "higher_is_better": false}},
    {"id": "gt1",    "type": "ground_truth","predecessors": ["kb1"],    "params": {"eval_function": "test_sorting"}}
  ]}
}
```

Node `id`s are client-scoped; `predecessors` reference earlier `id`s. The result
includes `final_thoughts` (leaf states, scores, solved flags) plus
`prompt_tokens`, `completion_tokens`, `cost`.

## Server-side: stateful session

1. `create_got_session(initial_parameters=..., task_name=..., model_name="openrouter")` → `session_id`.
2. `add_got_operation(session_id, op_type, client_op_id, predecessor_ids=[...], params={...})` per node.
3. `run_got_session(session_id)` → results (this deletes the session).

Use this when you want to inspect/branch the graph build programmatically. The
one-shot `execute_got_graph` is just sugar over these three.

## Client-side: agent-as-LLM loop

This is the zero-cost path. You drive the cycle; the server does formatting and
parsing only.

1. **Prompt:** `got_get_prompt(prompt_type="generate", task_name, variables={...})`.
   - `generate` variables: `original`, `current`, optional `method` (`io`/`cot`/`got`),
     optional `num_branches`.
   - `score`/`aggregation` variables: put the states under `state_dicts`.
2. **Think:** produce the answer text yourself (this is the "LLM call").
3. **Parse:** `got_parse_response(parse_type, responses=["...your text..."], task_name, variables, parser_spec)`.
   - `generate` → pass `variables={"state": {...}}`; returns list of new state dicts.
   - `score` → pass `variables={"states": [...]}`; returns list of floats.
   - `aggregation` → `variables={"states": [...]}`.
4. Repeat for the next operation, carrying the chosen state forward as `current`.

### Worked client-side example (custom task, verified)

```jsonc
// get_prompt
{"prompt_type": "generate", "task_name": "custom",
 "variables": {"original": "Brainstorm names for a graph-reasoning library", "current": "", "num_branches": 3},
 "templates": {"generate": "Propose {num_branches} distinct one-word names as a JSON list of strings.\nProject: {original}\nWork so far: {current}"}}
// → returns the formatted prompt; you then answer, e.g. ["Synapse","Lattice","Weave"]

// parse_response
{"parse_type": "generate", "task_name": "custom",
 "responses": ["[\"Synapse\", \"Lattice\", \"Weave\"]"],
 "parser_spec": {"generate_type": "list"}}
// → [{"current":"Synapse"},{"current":"Lattice"},{"current":"Weave"}]
```

## Authoring custom tasks (`task_name: "custom"`)

When no built-in fits, supply your own `templates` (prompter) and `parser_spec`
(parser). These flow through every tool that takes them.

### `templates` keys

`generate`, `score`, `aggregation`, `improve`, `validation`. Each is a Python
`.format()` string. Available placeholders are whatever you pass in `variables`,
plus `{num_branches}` (generate) and `{state_dicts}` (score/aggregation).

```jsonc
"templates": {
  "generate":    "Solve. Problem: {original}\nCurrent: {current}\nGive {num_branches} options as a JSON list.",
  "score":       "Rate each candidate 0-10 (higher better). Candidates: {state_dicts}\nReturn one number per candidate.",
  "aggregation": "Merge these into one best answer as JSON: {state_dicts}",
  "improve":     "Improve this answer, return JSON: {current}",
  "validation":  "Is this correct? Answer YES or NO. State: {current}"
}
```

### `parser_spec`

- `generate_type`: `"default"` (wrap whole text as `{"current": text}`),
  `"list"` (split a list → one state per item), or `"json"` (parse JSON
  object/array into state dict(s)).
- `score` parsing extracts numbers from the text in order, one per state
  (missing scores default to `0.0`).
- `validation` returns true if the text contains `yes`/`true`/`valid`.

**Rule of thumb:** make `generate` templates emit JSON or a clean list and set
`generate_type` accordingly — it's the most reliable to parse.

**Verified runnable example:** `examples/custom_tasks/rag_antihallucination.json`
is a complete, tested custom task (`generate×4 → score → keep_best_n(2) →
aggregate → improve`). It demonstrates **working text-based scoring** — the
`score` template tells the LLM to emit a single integer, the `DynamicParser`
extracts it, and `keep_best_n` prunes on it. Copy it as the starting template
for new custom tasks: edit `original`, the four templates, and the graph shape.

## Built-in tasks and their gotchas

Built-ins: `sorting`, `keyword_counting`, `set_intersection`, `doc_merge`.

- **Scoring is deterministic for `sorting`, `keyword_counting`,
  `set_intersection`.** Their *parsers* intentionally do not parse scores from
  text (`parse_score_answer` is a no-op). So in **server-side** mode give the
  `score` node a `scoring_function` (e.g. `sorting_errors`); in **client-side**
  mode `got_parse_response(parse_type="score")` returns `[]` for these tasks —
  compute the score yourself or use a custom task.
- **`doc_merge`** *does* parse scores from text, so client-side scoring works.
- Built-in `generate`/`aggregation`/`improve`/`validation` parsing works in
  client-side mode for all four tasks.

## Setup / config

- `.env` (project root) or shell env: `OPENROUTER_API_KEY`,
  `OPENROUTER_MODEL_ID` (e.g. `deepseek/deepseek-v4-flash`),
  `GOT_LANGUAGE_MODEL=openrouter`. Env vars override `config.json`.
- `model_name` selects a block in
  `graph_of_thoughts/language_models/config.json` (`openrouter`, `chatgpt`,
  `chatgpt4`, …). If omitted, the server falls back to `GOT_LANGUAGE_MODEL`,
  else `chatgpt`.
- **Server-side modes need a real key.** A 401 "Missing Authentication header"
  means the key is missing or still the placeholder — fix `.env`, then restart
  the MCP server so it reloads. Client-side mode needs no key.

## Decision guide

- Need a fast, free reasoning aid and you trust this model's outputs →
  **client-side** (`got_get_prompt` + `got_parse_response`).
- Need an automated multi-branch run with a (possibly cheaper/other) model and
  you have a key → **server-side** (`execute_got_graph`).
- Problem is one of the four built-ins → set `task_name` and reuse its
  prompter/parser/scoring.
- Novel problem → `task_name: "custom"` with `templates` + `parser_spec`.
