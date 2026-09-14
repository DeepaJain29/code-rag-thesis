"""Sandboxed execution of generated code for pass@k (CoderEval / RepoExec).

Each candidate runs in its own subprocess with a wall-clock timeout, so an
infinite loop or crash in generated code cannot take the run down.

NOTE: CoderEval's official harness runs inside its Docker image with the target
repos installed. This module gives you (a) a self-contained runner for tasks that
ship their own test code, and (b) the interface to swap in the official Docker
harness - keep evaluate_samples' signature and return shape if you do.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


@dataclass
class ExecResult:
    passed: bool
    stdout: str
    stderr: str
    timed_out: bool


def run_python_snippet(program: str, timeout: int = 15) -> ExecResult:
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "candidate.py"
        f.write_text(program, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, str(f)], capture_output=True,
                                  text=True, timeout=timeout, cwd=td)
            return ExecResult(proc.returncode == 0, proc.stdout[-4000:],
                              proc.stderr[-4000:], False)
        except subprocess.TimeoutExpired:
            return ExecResult(False, "", f"timeout after {timeout}s", True)


def build_program(candidate_code: str, test_code: str, preamble: str = "") -> str:
    return f"{preamble}\n\n{candidate_code}\n\n{test_code}\n"


def evaluate_samples(samples: Sequence[str], test_code: str, preamble: str = "",
                     timeout: int = 15) -> Dict:
    """Run every sample for one task; returns n_samples / n_correct for pass@k."""
    outcomes: List[ExecResult] = [
        run_python_snippet(build_program(s, test_code, preamble), timeout) for s in samples
    ]
    return {"n_samples": len(samples),
            "n_correct": sum(1 for o in outcomes if o.passed),
            "details": [{"passed": o.passed, "timed_out": o.timed_out,
                         "stderr": o.stderr[-500:]} for o in outcomes]}


def extract_code_block(text: str, language: str = "python") -> str:
    """Pull the first fenced code block out of a chat model's answer."""
    m = re.search(rf"```(?:{language}|py)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()
