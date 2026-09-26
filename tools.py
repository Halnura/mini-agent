"""
Tools for mini-agent. Each tool is a plain function wrapped in a Tool
dataclass with a name, description, and input hint shown to the model.

Tools must never crash the agent loop: they return error strings, and
agent.py additionally wraps every call in try/except.
"""
from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    input_hint: str
    func: Callable[[str], str]


# ---------------------------------------------------------------- web_search

def web_search(query: str, num_results: int = 5) -> str:
    """Search the web via DuckDuckGo. No API key required."""
    query = query.strip()
    if not query:
        return "Error: empty search query."
    # Rich results via the ddgs package when installed (pip install ddgs).
    try:
        from ddgs import DDGS

        results = list(DDGS().text(query, max_results=num_results))
        lines = [
            f"- {r.get('title', '')}: {r.get('body', '')} ({r.get('href', '')})"
            for r in results
        ]
        return "\n".join(lines) if lines else "No results found."
    except ImportError:
        pass  # fall through to the no-dependency endpoint
    except Exception as e:
        return f"Search error: {e}"
    # Fallback: DuckDuckGo Instant Answer API (works with plain httpx).
    try:
        import httpx

        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1},
            timeout=15,
        )
        data = resp.json()
        lines = []
        if data.get("AbstractText"):
            lines.append(f"- {data['AbstractText']} ({data.get('AbstractURL', '')})")
        for topic in (data.get("RelatedTopics") or [])[:num_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                lines.append(f"- {topic['Text']}")
        return "\n".join(lines) if lines else "No results found."
    except Exception as e:
        return f"Search error: {e}"


# ---------------------------------------------------------------- calculator

_ALLOWED_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_ALLOWED_FUNCS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan",
                 "floor", "ceil", "fabs", "pow")
}
_ALLOWED_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max})
_ALLOWED_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval(node: ast.AST) -> float:
    """Evaluate an AST with a strict whitelist — no attribute access, no imports."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func = _ALLOWED_FUNCS.get(node.func.id)
        if func is None or node.keywords:
            raise ValueError(f"function not allowed: {node.func.id}")
        return func(*[_safe_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _ALLOWED_CONSTS:
        return _ALLOWED_CONSTS[node.id]
    raise ValueError(f"expression not allowed: {ast.dump(node)[:60]}")


def calculator(expression: str) -> str:
    """Safely evaluate a math expression (no eval(), AST whitelist only)."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------- file_qa

def file_qa(query: str) -> str:
    """Answer a question over a local .txt/.md file via simple keyword retrieval.

    Input format:  "path/to/file.txt | your question"
    """
    if "|" not in query:
        return "Error: use the format 'path/to/file.txt | your question'."
    path, question = (part.strip() for part in query.split("|", 1))
    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as e:
        return f"Error reading file: {e}"
    chunks = [c.strip() for c in re.split(r"\n\s*\n", text) if c.strip()]
    if not chunks:
        return "Error: file is empty."
    q_words = set(re.findall(r"\w+", question.lower()))
    ranked = sorted(
        chunks,
        key=lambda c: len(set(re.findall(r"\w+", c.lower())) & q_words),
        reverse=True,
    )
    top = [c for c in ranked[:3] if set(re.findall(r"\w+", c.lower())) & q_words]
    if not top:
        return "No relevant content found in the file."
    return "\n---\n".join(top)


# ---------------------------------------------------------------- registry

TOOLS: list[Tool] = [
    Tool(
        name="web_search",
        description="Search the web for current facts, docs, and news.",
        input_hint="a search query string, e.g. 'latest stable Python release'",
        func=web_search,
    ),
    Tool(
        name="calculator",
        description="Evaluate a math expression safely.",
        input_hint="an expression, e.g. '0.15 * 240' or 'sqrt(144) + 2**10'",
        func=calculator,
    ),
    Tool(
        name="file_qa",
        description="Answer a question over a local text/markdown file.",
        input_hint="'path/to/file.txt | your question'",
        func=file_qa,
    ),
]
