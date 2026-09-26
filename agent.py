"""
mini-agent: a minimal ReAct-style tool-using research agent.

Weekend project demonstrating the core agent loop used in production
agent systems: plan -> act (tool call) -> observe -> repeat -> answer.

Usage:
    python agent.py "What is 15% of 240 and what is the latest stable Python release?"
    python agent.py            # interactive mode

Requires OPENAI_API_KEY in the environment (see .env.example).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from tools import TOOLS, Tool

SYSTEM_PROMPT = """You are a careful research assistant that solves questions step by step using tools.

Available tools:
{tool_docs}

Rules:
- Think step by step. On each step output EXACTLY one JSON object and nothing else:
    {{"action": "<tool_name>", "input": "<tool input>"}}   to call a tool, or
    {{"action": "final", "input": "<your final answer>"}}  to finish.
- Use a tool whenever you need a fact, a calculation, or file contents. Do not guess.
- After each tool result, keep reasoning until you can answer confidently.
- Keep the final answer concise and say which tools you used.
- Stop after at most {max_steps} tool calls.
"""

TOOL_DOCS = "\n".join(
    f"- {t.name}: {t.description} Input hint: {t.input_hint}" for t in TOOLS
)


# ---------------------------------------------------------------- backends

class LLMBackend:
    """Pluggable chat backend. Subclass and implement chat()."""

    def chat(self, messages: list[dict]) -> str:
        raise NotImplementedError


class LangChainBackend(LLMBackend):
    """Preferred backend: langchain-openai (pip install langchain-openai)."""

    def __init__(self) -> None:
        from langchain_openai import ChatOpenAI

        self.llm = ChatOpenAI(
            model=os.getenv("MINI_AGENT_MODEL", "gpt-4o-mini"), temperature=0
        )

    def chat(self, messages: list[dict]) -> str:
        return self.llm.invoke(messages).content  # type: ignore[no-any-return]


class OpenAIBackend(LLMBackend):
    """Fallback backend: direct OpenAI SDK (pip install openai)."""

    def __init__(self) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = os.getenv("MINI_AGENT_MODEL", "gpt-4o-mini")

    def chat(self, messages: list[dict]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        return resp.choices[0].message.content or ""


class StubBackend(LLMBackend):
    """Canned ReAct responses. Lets the loop and eval harness run without an API key."""

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.i = 0

    def chat(self, messages: list[dict]) -> str:
        out = self.script[min(self.i, len(self.script) - 1)]
        self.i += 1
        return out


def get_backend() -> LLMBackend:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example)")
    try:
        return LangChainBackend()
    except ImportError:
        pass
    try:
        return OpenAIBackend()
    except ImportError:
        raise RuntimeError(
            "Install a backend: pip install langchain-openai  (or: pip install openai)"
        )


# ---------------------------------------------------------------- agent loop

def parse_action(raw: str) -> dict | None:
    """Extract the first JSON object containing an 'action' key from model output."""
    for m in re.finditer(r"\{.*?\}", raw, re.DOTALL):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            return obj
    return None


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)
    latency_s: float = 0.0
    tool_calls: int = 0


def run(question: str, backend: LLMBackend | None = None,
        max_steps: int = 6, verbose: bool = True) -> AgentResult:
    """Run the ReAct loop on a question. Returns the final answer plus a trace."""
    backend = backend or get_backend()
    tools: dict[str, Tool] = {t.name: t for t in TOOLS}
    system = SYSTEM_PROMPT.format(tool_docs=TOOL_DOCS, max_steps=max_steps)
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    tool_calls = 0
    answer = "I couldn't reach an answer within the step limit."
    start = time.time()

    for step in range(1, max_steps + 1):
        raw = backend.chat(messages)
        action = parse_action(raw)
        if action is None:
            trace.append({"step": step, "error": "could not parse action", "raw": raw})
            break
        name, action_input = action.get("action"), str(action.get("input", ""))

        if name == "final":
            answer = action_input
            trace.append({"step": step, "action": "final", "answer": answer})
            break

        if name not in tools:
            observation = f"Error: unknown tool '{name}'. Valid tools: {sorted(tools)}."
        else:
            tool_calls += 1
            try:
                observation = tools[name].func(action_input)
            except Exception as e:  # tools must never crash the loop
                observation = f"Error running {name}: {e}"
        trace.append(
            {"step": step, "action": name, "input": action_input,
             "observation": observation[:800]}
        )
        if verbose:
            print(f"[step {step}] {name}({action_input[:80]})")
            print(f"           -> {observation[:160]}")
        messages.append({"role": "assistant", "content": raw})
        messages.append({
            "role": "user",
            "content": f"Tool result:\n{observation}\n\nContinue. Reply with exactly one JSON action.",
        })

    return AgentResult(
        answer=answer, trace=trace,
        latency_s=round(time.time() - start, 2), tool_calls=tool_calls,
    )


def main() -> None:
    import sys

    if len(sys.argv) > 1:
        result = run(" ".join(sys.argv[1:]))
        print("\nAnswer:", result.answer)
        return
    print("mini-agent — ask a question (Ctrl+C to quit)")
    while True:
        try:
            q = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        result = run(q)
        print("Agent:", result.answer)


if __name__ == "__main__":
    main()
