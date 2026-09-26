# mini-agent

A minimal ReAct-style tool-using research agent in ~370 lines of Python.
Weekend project, built to demonstrate the three patterns every production
agent system needs: **the agent loop**, **tool use**, and **evaluation**.

## What it is

Ask a question in plain English. The agent plans, calls tools, observes the
results, and synthesizes an answer — the same plan → act → observe loop that
powers LangGraph / CrewAI / AutoGen-style systems, just without the framework.

## Architecture

```
                    ┌─────────────────────┐
                    │       agent.py      │
                    │    ReAct loop:      │
  question ────────▶│ plan → act → observe│──▶ final answer + trace
                    └────────┬────────────┘
              JSON action     │  observation
                     ┌────────▼────────┐   ┌──────────────────┐
                     │    tools.py     │   │   LLM backend    │
                     │  • web_search  │   │  langchain-openai│
                     │  • calculator   │◀──│  (or openai SDK  │
                     │  • file_qa     │   │   fallback)      │
                     └────────────────┘   └──────────────────┘

                    ┌─────────────────────┐
                    │      eval.py        │
                    │ 5 tasks → pass/fail │
                    │ + latency report    │
                    └─────────────────────┘
```

## How to run (3 commands)

```bash
pip install -r requirements.txt
cp .env.example .env   # then put your key in .env
python agent.py "What is 15% of 240, and what is the latest stable Python release?"
```

More ways to run:

```bash
python agent.py            # interactive mode
python eval.py             # eval harness — works WITHOUT an API key (stub mode)
```

## Example session

See [examples/sample_transcript.md](examples/sample_transcript.md) for a full
transcript. Short version:

```
$ python agent.py "What is 15% of 240, and what vector database do my notes mention?"

[step 1] calculator(0.15 * 240)            -> 36.0
[step 2] file_qa(examples/notes.txt | ...) -> ...FAISS vector database...
[step 3] final(...)

Answer: 15% of 240 is 36. Your notes say the project uses FAISS.
(Tools used: calculator, file_qa)
```

## Tools

| Tool         | What it does | Notes |
|--------------|--------------|-------|
| `web_search` | Web search via DuckDuckGo | No API key; uses `ddgs` if installed, else plain `httpx` against the Instant Answer API |
| `calculator` | Safe math evaluation | AST whitelist — no `eval()`, no imports, no attribute access |
| `file_qa`    | Q&A over a local `.txt`/`.md` file | Input: `"path/to/file | your question"`; keyword retrieval over chunks |

## Evaluation

`eval.py` runs 5 tasks (arithmetic, web fact, file Q&A, multi-hop, no-tool)
and reports pass/fail + latency per task. A task passes when all expected
keywords appear in the final answer. Without `OPENAI_API_KEY` it runs in stub
mode — canned ReAct scripts that still exercise the real loop, real tools, and
the scoring logic. Set the key to evaluate against a live model.

## Design decisions

- **ReAct with JSON actions, not framework magic.** The model outputs one JSON
  object per step (`{"action": ..., "input": ...}`). Easy to log, easy to trace,
  easy to unit-test — this is the observability pattern I'd want in production.
- **Pluggable LLM backend.** `langchain-openai` if available, otherwise the raw
  OpenAI SDK. The agent loop doesn't care which; swapping in another provider
  means writing one small `chat()` method.
- **Tools as data, not decorators.** A `Tool` dataclass (name, description,
  input hint, function) keeps the registry explicit and makes the tool docs in
  the system prompt impossible to drift from the implementation.
- **Tools never crash the loop.** Every tool returns error strings, and the loop
  wraps each call in `try/except` — a flaky tool degrades to a bad observation,
  not a dead agent. Same principle as production agent infra.
- **Keyword evals over vibes.** Pass/fail on expected-answer keywords plus
  latency per task is the smallest harness that still catches regressions. Stub
  mode keeps it runnable in CI without secrets.
- **Deliberately small.** ~370 lines, no agent framework, no vector DB, no
  server. The point is the loop, the tool contract, and the eval discipline —
  all of which transfer directly to larger systems.
