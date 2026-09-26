"""
Tiny eval harness for mini-agent.

Runs 5 sample tasks through the agent loop and reports pass/fail per task
plus latency. A task passes when every expected keyword appears in the
final answer (case-insensitive).

Two modes:
  - Live:  set OPENAI_API_KEY and the agent uses a real model.
  - Stub:   no key set -> canned ReAct scripts exercise the loop, the
            tools, and the scoring logic. Useful for CI / graders.

Usage:  python eval.py
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from agent import LLMBackend, StubBackend, run

TASKS: list[dict] = [
    {
        "id": "arithmetic",
        "question": "What is 15% of 240?",
        "keywords": ["36"],
        "script": [
            '{"action": "calculator", "input": "0.15 * 240"}',
            '{"action": "final", "input": "15% of 240 is 36."}',
        ],
    },
    {
        "id": "web_fact",
        "question": "What is the latest stable release of Python?",
        "keywords": ["python", "3."],
        "script": [
            '{"action": "web_search", "input": "latest stable Python release version"}',
            '{"action": "final", "input": "The latest stable Python release is in the 3.x series."}',
        ],
    },
    {
        "id": "file_qa",
        "question": "examples/notes.txt | What does the notes file say about vector databases?",
        "keywords": ["FAISS", "embeddings"],
        "script": [
            '{"action": "file_qa", "input": "examples/notes.txt | vector databases"}',
            '{"action": "final", "input": "Per the notes, the project stores embeddings in a FAISS vector database."}',
        ],
    },
    {
        "id": "multi_hop",
        "question": "If a model costs $0.02 per 1K tokens and I use 50K tokens, what do I pay?",
        "keywords": ["1.00"],
        "script": [
            '{"action": "calculator", "input": "0.02 * 50"}',
            '{"action": "final", "input": "50K tokens at $0.02 per 1K tokens costs $1.00."}',
        ],
    },
    {
        "id": "no_tool_needed",
        "question": "What does RAG stand for in the context of LLMs?",
        "keywords": ["retrieval", "augmented", "generation"],
        "script": [
            '{"action": "final", "input": "RAG stands for Retrieval-Augmented Generation."}',
        ],
    },
]


@dataclass
class EvalResult:
    task_id: str
    passed: bool
    latency_s: float
    tool_calls: int
    detail: str
    answer: str = ""


def run_task(task: dict, backend: LLMBackend | None) -> EvalResult:
    try:
        result = run(task["question"], backend=backend, verbose=False)
    except Exception as e:
        return EvalResult(task["id"], False, 0.0, 0, f"error: {e}")
    missing = [k for k in task["keywords"] if k.lower() not in result.answer.lower()]
    return EvalResult(
        task_id=task["id"],
        passed=not missing,
        latency_s=result.latency_s,
        tool_calls=result.tool_calls,
        detail="ok" if not missing else "missing keywords: " + ", ".join(missing),
        answer=result.answer,
    )


def main() -> int:
    live = bool(os.getenv("OPENAI_API_KEY"))
    print(f"mini-agent eval — {len(TASKS)} tasks "
          f"(backend: {'live model' if live else 'stub (no API key)'})\n")
    results: list[EvalResult] = []
    for task in TASKS:
        backend: LLMBackend | None = None if live else StubBackend(task["script"])
        r = run_task(task, backend)
        results.append(r)
        mark = "PASS" if r.passed else "FAIL"
        print(f"[{mark}] {r.task_id:<14} {r.latency_s:>6.2f}s  "
              f"{r.tool_calls} tool call(s)  {r.detail}")
    passed = sum(r.passed for r in results)
    total_latency = sum(r.latency_s for r in results)
    print(f"\n{passed}/{len(results)} passed, "
          f"total latency {total_latency:.2f}s")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
