"""Chunking - pinned to arXiv:2601.08773's vector-only baseline.

RecursiveCharacterTextSplitter, chunk_size=1000 CHARACTERS, overlap=100.
Not tokens. Changing these means recording it under 'known deviations'.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.repos import iter_source_files


@dataclass
class ChunkStats:
    repo: str
    n_files: int = 0
    n_chunks: int = 0
    n_chars: int = 0
    skipped_files: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"repo": self.repo, "files": self.n_files, "chunks": self.n_chunks,
                "chars": self.n_chars, "skipped": len(self.skipped_files)}


def make_splitter(chunk_size: int = 1000, chunk_overlap: int = 100):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,       # characters
        add_start_index=True,
    )


def chunk_repo(repo_name: str, repo_dir: Path, include_ext: Iterable[str],
               exclude_dirs: Iterable[str], chunk_size: int = 1000,
               chunk_overlap: int = 100, max_file_bytes: int = 1_000_000):
    """Walk one repo -> LangChain Documents with stable chunk ids.

    chunk_id format: '<repo>::<relative/path.java>::<ordinal>'
    Systems B and C reuse this id scheme, so retrieval precision/recall is
    computed against identical ground-truth ids across all three systems.
    """
    splitter = make_splitter(chunk_size, chunk_overlap)
    stats = ChunkStats(repo=repo_name)
    docs: List[Document] = []

    for path in iter_source_files(repo_dir, include_ext, exclude_dirs, max_file_bytes):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            stats.skipped_files.append(str(path))
            continue
        if not text.strip():
            continue

        rel = str(path.relative_to(repo_dir))
        stats.n_files += 1
        stats.n_chars += len(text)

        for i, piece in enumerate(splitter.split_text(text)):
            docs.append(Document(
                page_content=piece,
                metadata={"chunk_id": f"{repo_name}::{rel}::{i}",
                          "repo": repo_name, "source": rel, "abs_path": str(path),
                          "ordinal": i, "n_chars": len(piece)},
            ))

    stats.n_chunks = len(docs)
    return docs, stats
