"""Comment/docstring stripping for the RQ4 (undocumented-code) condition.

Writes a derived snapshot to data/derived/<repo>-stripped/ so the original clone
stays pristine and every system indexes the identical stripped tree.
"""
from __future__ import annotations

import io
import re
import tokenize
from pathlib import Path
from typing import Iterable

from . import state
from .paths import paths


def strip_python(source: str) -> str:
    """Remove comments and docstrings."""
    try:
        out, prev_toktype, last_col, last_lineno = [], tokenize.INDENT, 0, -1
        for ttype, tstring, (slineno, scol), (elineno, ecol), _ in \
                tokenize.generate_tokens(io.StringIO(source).readline):
            if slineno > last_lineno:
                last_col = 0
            if scol > last_col:
                out.append(" " * (scol - last_col))
            if ttype == tokenize.COMMENT:
                pass
            elif ttype == tokenize.STRING and prev_toktype in (tokenize.INDENT,
                                                               tokenize.NEWLINE):
                pass                      # docstring
            else:
                out.append(tstring)
            prev_toktype = ttype
            last_col, last_lineno = ecol, elineno
        text = "".join(out)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        text = re.sub(r"#.*", "", source)
    return "\n".join(line for line in text.splitlines() if line.strip())


_JAVA_COMMENTS = re.compile(r"//.*?$|/\*.*?\*/", re.DOTALL | re.MULTILINE)


def strip_java(source: str) -> str:
    text = _JAVA_COMMENTS.sub("", source)
    return "\n".join(line for line in text.splitlines() if line.strip())


STRIPPERS = {".py": strip_python, ".java": strip_java}


def build_stripped_snapshot(repo_dir: Path, name: str, include_ext: Iterable[str],
                            exclude_dirs: Iterable[str], force: bool = False) -> Path:
    """Write a comment/docstring-free copy once; reused by every system."""
    from .repos import iter_source_files

    dest = paths.derived / f"{name}-stripped"
    key = f"derived:{name}-stripped"
    if force:
        state.clear(key)
    if state.is_done(key, check_path=dest):
        print(f"  [skip] stripped snapshot for {name} exists")
        return dest

    n = 0
    for src in iter_source_files(repo_dir, include_ext, exclude_dirs):
        rel = src.relative_to(repo_dir)
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        fn = STRIPPERS.get(src.suffix, lambda s: s)
        try:
            out.write_text(fn(src.read_text(encoding="utf-8", errors="ignore")),
                           encoding="utf-8")
            n += 1
        except OSError:
            continue
    state.mark_done(key, path=str(dest), files=n, source=str(repo_dir))
    print(f"  [ok] stripped snapshot {name}: {n} files -> {dest}")
    return dest
