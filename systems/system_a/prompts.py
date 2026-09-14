"""Prompt assembly.

The template is IDENTICAL for Leg 1 and Leg 2 except the task instruction
(spec section 4.6). Changing it invalidates cross-leg comparability.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

SYSTEM_PROMPT = (
    "You are a precise software engineering assistant. Answer strictly using the "
    "retrieved repository context provided. If the context is insufficient, say so "
    "explicitly rather than inventing APIs, files, or behaviour."
)

CONTEXT_TEMPLATE = """### Retrieved repository context
{context}

### Task
{instruction}

### Query
{query}
"""

QA_INSTRUCTION = (
    "Answer the question about this repository. Be specific: name the exact classes, "
    "methods, and files involved, and explain the relationship between them."
)

GEN_INSTRUCTION = (
    "Complete or generate the requested code so that it is consistent with the "
    "repository's existing APIs and conventions. Return ONLY the code inside a single "
    "fenced code block, with no explanation before or after."
)


def format_context(chunks: Sequence[Tuple[str, str]], max_chars: int = 24000) -> str:
    """chunks: (source_label, chunk_text) in retrieval-rank order."""
    parts: List[str] = []
    used = 0
    for rank, (label, text) in enumerate(chunks, 1):
        block = f"[chunk {rank}] source: {label}\n```\n{text}\n```"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts) if parts else "(no context retrieved)"


def build_prompt(query: str, chunks: Sequence[Tuple[str, str]], task: str = "qa",
                 max_context_chars: int = 24000) -> str:
    return CONTEXT_TEMPLATE.format(
        context=format_context(chunks, max_context_chars),
        instruction=QA_INSTRUCTION if task == "qa" else GEN_INSTRUCTION,
        query=query,
    )
