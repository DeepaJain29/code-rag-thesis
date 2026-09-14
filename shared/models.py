"""Model download-once + in-process singleton loading.

Both models are GLOBAL assets: Systems A, B and C load the same weights from the
same local cache. Nothing here re-downloads once the state marker exists - normal
runs are offline by default (see shared/env.py).
"""
from __future__ import annotations

import functools
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from . import state
from .paths import paths

_LOCK = threading.Lock()


# --------------------------------------------------------------------------- #
# Download (bootstrap only)
# --------------------------------------------------------------------------- #
def download_model(repo_id: str, revision: str = "main", force: bool = False) -> str:
    """Snapshot-download a model into data/models/hf. Idempotent."""
    key = f"model:{repo_id}"
    if force:
        state.clear(key)
    if state.is_done(key, check_path=paths.hf_cache):
        print(f"  [skip] model {repo_id} already cached")
        return (state.read(key) or {}).get("commit", "")

    from huggingface_hub import snapshot_download

    print(f"  [download] {repo_id} -> {paths.hf_cache}")
    local = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(paths.hf_cache),
        ignore_patterns=["*.pth", "*.msgpack", "*.h5", "*.onnx"],   # safetensors only
    )
    try:
        from huggingface_hub import HfApi
        sha = HfApi().model_info(repo_id, revision=revision).sha
    except Exception:
        sha = revision
    state.save_pin(f"model:{repo_id}", revision=revision, commit=sha)
    state.mark_done(key, repo_id=repo_id, revision=revision, commit=sha, local_dir=local)
    print(f"  [ok] {repo_id} @ {sha}")
    return sha


# --------------------------------------------------------------------------- #
# Embeddings - LangChain-compatible wrapper over Qwen3-Embedding
# --------------------------------------------------------------------------- #
try:
    from langchain_core.embeddings import Embeddings as _LCEmbeddings
except Exception:                                    # keeps unit tests importable
    class _LCEmbeddings:                             # type: ignore
        pass


class QwenEmbeddings(_LCEmbeddings):
    """Queries get the model's instruct prompt, documents do not (Qwen3-Embedding
    is asymmetric). Vectors are L2-normalised, so inner product == cosine."""

    def __init__(self, repo_id: str, max_seq_length: int = 1024, batch_size: int = 32,
                 normalize: bool = True, query_prompt: str = "", device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer

        self.repo_id = repo_id
        self.batch_size = batch_size
        self.normalize = normalize
        self.query_prompt = query_prompt
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(
            repo_id, cache_folder=str(paths.hf_cache), device=self.device,
            model_kwargs={"torch_dtype": torch.float16} if self.device == "cuda" else {},
        )
        self.model.max_seq_length = max_seq_length
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.model.encode(list(texts), batch_size=self.batch_size,
                                 convert_to_numpy=True,
                                 normalize_embeddings=self.normalize,
                                 show_progress_bar=len(texts) > 256)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode([self.query_prompt + text], batch_size=1,
                                convert_to_numpy=True,
                                normalize_embeddings=self.normalize,
                                show_progress_bar=False)[0]
        return vec.tolist()

    def count_tokens(self, texts: Sequence[str]) -> int:
        tok = self.model.tokenizer
        return sum(len(tok.encode(t, add_special_tokens=False)) for t in texts)


@functools.lru_cache(maxsize=2)
def get_embedder(repo_id: str, max_seq_length: int = 1024, batch_size: int = 32,
                 normalize: bool = True, query_prompt: str = "") -> QwenEmbeddings:
    """Process-wide singleton - weights load into RAM/VRAM once per run."""
    with _LOCK:
        print(f"[models] loading embedder {repo_id}")
        return QwenEmbeddings(repo_id, max_seq_length, batch_size, normalize, query_prompt)


# --------------------------------------------------------------------------- #
# Generation - Qwen2.5-Coder (identical across A/B/C: the key control)
# --------------------------------------------------------------------------- #
@dataclass
class GenOutput:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    samples: List[str] = field(default_factory=list)


class Generator:
    def __init__(self, repo_id: str, dtype: str = "bfloat16", load_in_4bit: bool = False,
                 max_new_tokens: int = 1024, temperature: float = 0.2, top_p: float = 0.95):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.repo_id = repo_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        kwargs: Dict[str, Any] = {
            "cache_dir": str(paths.hf_cache),
            "device_map": "auto" if torch.cuda.is_available() else None,
            "torch_dtype": getattr(torch, dtype, torch.float32),
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs.pop("torch_dtype", None)
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)

        self.tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=str(paths.hf_cache))
        self.model = AutoModelForCausalLM.from_pretrained(repo_id, **kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def generate(self, system_prompt: str, user_prompt: str, n_samples: int = 1,
                 temperature: float | None = None) -> GenOutput:
        import time

        import torch

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        temp = self.temperature if temperature is None else temperature

        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=temp > 0,
                temperature=max(temp, 1e-5),
                top_p=self.top_p,
                num_return_sequences=n_samples,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.perf_counter() - t0

        prompt_len = inputs["input_ids"].shape[-1]
        samples = [self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                   for seq in out]
        completion_tokens = int(sum(len(seq) - prompt_len for seq in out))
        return GenOutput(text=samples[0], prompt_tokens=int(prompt_len),
                         completion_tokens=completion_tokens, latency_s=latency,
                         samples=samples)


@functools.lru_cache(maxsize=1)
def get_generator(repo_id: str, dtype: str = "bfloat16", load_in_4bit: bool = False,
                  max_new_tokens: int = 1024, temperature: float = 0.2,
                  top_p: float = 0.95) -> Generator:
    with _LOCK:
        print(f"[models] loading generator {repo_id}")
        return Generator(repo_id, dtype, load_in_4bit, max_new_tokens, temperature, top_p)


# --------------------------------------------------------------------------- #
def embedder_from_config(cfg) -> QwenEmbeddings:
    e = cfg.get_path("models.embedding")
    return get_embedder(e["repo_id"], e.get("max_seq_length", 1024),
                        e.get("batch_size", 32), e.get("normalize", True),
                        e.get("query_prompt", ""))


def generator_from_config(cfg) -> Generator:
    g = cfg.get_path("models.generation")
    return get_generator(g["repo_id"], g.get("dtype", "bfloat16"),
                         g.get("load_in_4bit", False), g.get("max_new_tokens", 1024),
                         g.get("temperature", 0.2), g.get("top_p", 0.95))
