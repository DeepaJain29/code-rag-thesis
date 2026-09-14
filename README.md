# Structural vs. LLM-Extracted Graph Retrieval for Repository-Level Code Generation

Implementation repo for the M.Tech thesis. Three systems share one set of global
assets (models, repo snapshots, datasets, query sets) so that the A/B/C comparison
is controlled by construction:

| System | What it is | Status |
|---|---|---|
| **A** | Vanilla chunk → embed → retrieve RAG (this build) | implemented |
| **B** | LLM-extraction GraphRAG adapted to code | later, `systems/system_b/` |
| **C** | Structural (AST + call/import graph) RAG | later, `systems/system_c/` |

System A is pinned to arXiv:2601.08773's vector-only baseline: 1000-character
`RecursiveCharacterTextSplitter` chunks with 100-character overlap, FAISS, top-k = 10,
no re-ranking, no graph, no tuning.

---

## 1. Folder architecture

```
code-rag-thesis/
├── README.md
├── Makefile                     shortcuts for every command below
├── requirements.txt
├── configs/
│   ├── global.yaml              models, repos, datasets  ← shared by A, B, C
│   ├── system_a.yaml            System A's pinned parameters
│   └── pins.lock.json           auto-written: exact commit/sha of every asset
├── shared/                      ← the layer B and C will reuse unchanged
│   ├── env.py                   cache redirection + offline enforcement
│   ├── paths.py                 every path in the project, one place
│   ├── config.py                YAML loading
│   ├── state.py                 download-once markers + pin lock
│   ├── repos.py                 pinned git clones, source-file walker
│   ├── datasets_setup.py        CoderEval / RepoExec / CrossCodeEval acquisition
│   ├── models.py                Qwen3-Embedding + Qwen2.5-Coder, downloaded once,
│   │                            loaded once per process (singleton)
│   ├── textproc.py              comment/docstring stripping (RQ4 condition)
│   ├── exec_eval.py             sandboxed execution of generated code
│   ├── metrics.py               pass@k, EM, ES, retrieval P/R, latency percentiles
│   └── logging_utils.py         per-query JSONL logging (spec §4.8)
├── systems/
│   └── system_a/
│       ├── chunking.py          the pinned splitter + stable chunk ids
│       ├── index_build.py       CLI: build one FAISS index per repo
│       ├── retriever.py         top-k cosine retrieval, no re-ranking
│       ├── prompts.py           one template, two task instructions
│       ├── pipeline.py          retrieve → prompt → single LLM call → log
│       ├── run_leg1.py          CLI: 45-question QA replication
│       ├── run_leg2.py          CLI: pass@k / EM / ES / retrieval P-R
│       ├── report.py            CLI: builds the six-section report
│       └── eval/leg1_questions.json   45 questions (answer keys to fill in)
├── scripts/
│   ├── bootstrap.py             ONE-TIME downloads (the only networked script)
│   ├── doctor.py                what's downloaded / indexed / missing
│   └── build_query_set.py       CrossCodeEval-style held-out queries for Frappe
├── data/                        GLOBAL, git-ignored, downloaded once
│   ├── .state/                  completion markers → the download-once guard
│   ├── models/hf/               model weights (shared by A, B, C)
│   ├── repos/<name>/            pinned clones
│   ├── datasets/raw|hf/         benchmark data
│   ├── derived/                 stripped snapshots, built query sets
│   └── indexes/<system>/<repo>/ vector indexes, one per repo per system
└── results/<system>/            run logs + metric tables + final report
```

**Why `data/` sits at the root and not inside `systems/system_a/`:** models and repo
snapshots are the controlled variables of the experiment. Systems B and C must use the
identical weights and the identical commit, so they live one level above the systems and
are referenced through `shared/paths.py`. Only the *indexes* are per-system
(`data/indexes/system_a/…`), because each system builds its own.

---

## 2. Install (once)

```bash
git clone <your-repo> code-rag-thesis && cd code-rag-thesis
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU runs, install the CUDA build of torch first (from pytorch.org), then the rest.
On <16GB VRAM set `load_in_4bit: true` under `models.generation` in `configs/global.yaml`,
or drop to `Qwen/Qwen2.5-Coder-7B-Instruct` and `Qwen/Qwen3-Embedding-0.6B`.

---

## 3. First run (downloads happen exactly once)

```bash
# 1. pull models, repos, datasets, and the stripped snapshots  (the ONLY network step)
python scripts/bootstrap.py

# 2. confirm what landed
python scripts/doctor.py

# 3. build System A's FAISS indexes (one per repo; the slow GPU step)
python -m systems.system_a.index_build --leg all

# 4. fill reference_answer + keywords in systems/system_a/eval/leg1_questions.json, then:
python -m systems.system_a.run_leg1

# 5. generation leg
python scripts/build_query_set.py --repo frappe --n 200
python -m systems.system_a.run_leg2 --dataset frappe
python -m systems.system_a.run_leg2 --dataset codereval --index-repo frappe
python -m systems.system_a.run_leg2 --dataset repoexec  --index-repo frappe

# 6. assemble the report
python -m systems.system_a.report          # → results/system_a/report_system_a.md
```

Smoke-test first if you like: `--limit 3` on either leg runs three queries end to end.

## 4. Every run after that

Nothing re-downloads and nothing re-indexes. Just run the piece you want:

```bash
python -m systems.system_a.run_leg1
python -m systems.system_a.run_leg2 --dataset frappe
python -m systems.system_a.report
```

`scripts/bootstrap.py` and `index_build` are safe to re-run at any time — they detect the
existing state markers and skip. Via make: `make bootstrap`, `make index`, `make leg1`,
`make leg2-frappe`, `make report`, `make doctor`.

### What makes it download-once

1. **Cache redirection** — `shared/env.py` sets `HF_HOME`, `HF_HUB_CACHE`,
   `TRANSFORMERS_CACHE`, `SENTENCE_TRANSFORMERS_HOME`, `HF_DATASETS_CACHE` and
   `TORCH_HOME` to `data/models` and `data/datasets` *before* any ML library is imported.
   Weights land in the project, not in `~/.cache`, so they survive across systems and
   move with the folder.
2. **State markers** — every completed download writes `data/.state/<key>.json`. Later
   calls check the marker *and* that the artefact still exists; if you delete the files,
   the marker is invalidated automatically.
3. **Offline by default** — every entry point except `bootstrap.py` sets
   `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`. A run physically cannot fetch weights;
   if something is missing you get a clear error instead of a silent 20GB download.
   Override deliberately with `CODERAG_ALLOW_DOWNLOAD=1`.
4. **Pin lock** — `configs/pins.lock.json` records the resolved commit sha of every repo,
   dataset and model. Commit this file: it is what lets Systems B and C index the exact
   same snapshot, and what makes the thesis numbers reproducible.

### Re-downloading on purpose

```bash
python scripts/bootstrap.py --force models     # or repos / datasets / derived
```

### Moving the whole thing (e.g. to Colab)

Copy the folder, or set `CODERAG_HOME=/content/drive/MyDrive/code-rag-thesis`. All paths
derive from that one variable, so mounting Drive and pointing `CODERAG_HOME` at it means
Colab re-uses the same downloads and indexes instead of pulling them again each session.

---

## 5. Pinning the repo snapshots

`configs/global.yaml` ships `commit: null` for every repo, meaning "resolve HEAD of `ref`
on first clone and lock it". After the first bootstrap, copy the shas from
`configs/pins.lock.json` back into `commit:` in the YAML. That is what makes chunk counts
stable and what you cite in the report when your counts differ from the paper's
(5,403 / 32,428 / 11,495 chunks — those came from an earlier snapshot).

---

## 6. Where the spec's requirements live in the code

| Spec item | Code |
|---|---|
| §2 chunking 1000/100 chars | `configs/system_a.yaml` → `systems/system_a/chunking.py` |
| §2 Qwen3-Embedding via LangChain wrapper | `shared/models.QwenEmbeddings` (implements `langchain_core.embeddings.Embeddings`) |
| §2 FAISS, top-k = 10, cosine | `index_build.py` (`MAX_INNER_PRODUCT` + `normalize_L2`), `retriever.py` |
| §2 Qwen2.5-Coder generation | `shared/models.Generator`, shared with B and C |
| §3 datasets + pinned commits | `configs/global.yaml`, `shared/repos.py`, `shared/datasets_setup.py` |
| §4.4 separate index per repo/leg | `data/indexes/system_a/<repo>/` |
| §4.6 identical prompt, different instruction | `systems/system_a/prompts.py` |
| §4.8 mandatory per-query logging | `shared/logging_utils.QueryRecord` → `results/system_a/**/queries.jsonl` |
| §5 Leg 1 scoring | `shared/metrics.qa_rubric_score` + the `human_score` field; question set in `eval/leg1_questions.json` |
| §5 pass@k / EM / ES / retrieval P-R | `shared/metrics.py`, `shared/exec_eval.py` |
| §5 cost metrics | index metadata + `latency_percentiles` over the query log |
| §6 six-section report | `systems/system_a/report.py` |
| §7 non-goals (no graph, no re-rank, no tuning) | enforced by config: `rerank: none`, no tuning CLI |

---

## 7. Honest gaps you still have to close by hand

These are deliberate placeholders, not bugs — each one is a judgement call the spec says
to document rather than guess:

1. **The 45 Leg-1 questions are written, but their answer keys are not.** The paper
   doesn't publish its question set, so `eval/leg1_questions.json` ships a *reconstructed
   comparable set* — 15 per repo, 5 each across repository-semantics, multi-hop-trace and
   system-reasoning. The questions are phrased as questions, so nothing is asserted about
   code that hasn't been read; `reference_answer`, `keywords` and `gold_files` are empty
   and you must fill them from the source at the pinned commit. Say in §6 of the report
   that the set is reconstructed, not the paper's.
2. **Leg-1 grading** is automatic keyword coverage only — and it reads the `keywords` you
   fill in above, so it scores 0 until then. Fill `human_score` (or wire an LLM judge)
   before quoting any accuracy number.
3. **CoderEval/RepoExec field names** vary between releases. `run_leg2._normalise_task`
   maps them; print one record and fix the mapping if a field comes back empty.
4. **CoderEval's official harness** runs in Docker with the target repos installed.
   `shared/exec_eval.py` is a local subprocess runner — fine for self-contained tasks,
   but state which one you used.
5. **Retrieval ground truth** for the held-out set is "every chunk of the file that
   defines the called symbol". That is generous; tighten it to the defining function's
   chunk span if you want a stricter precision number, and say which you used.

---

## 8. Adding System B and System C later

Create `systems/system_b/` next to `system_a/` and import the same shared layer:

```python
import shared                                   # configures caches, must come first
from shared.config import load_config
from shared.models import embedder_from_config, generator_from_config
from shared.repos import ensure_repo            # same pinned commit as System A
from shared.logging_utils import RunLogger, QueryRecord
from shared.metrics import aggregate_pass_at_k
```

Rules that keep the comparison valid:

- Never re-clone or re-download. `ensure_repo` returns the snapshot System A indexed.
- Reuse `chunk_id` strings (`<repo>::<path>::<ordinal>`) as the unit of retrieval ground
  truth, even where B and C retrieve nodes rather than chunks — map node → covering
  chunk ids so precision/recall is computed on one scale.
- Write indexes to `data/indexes/system_b/…`, results to `results/system_b/…`.
- Add `configs/system_b.yaml`; leave `configs/global.yaml` alone, since changing a global
  value silently changes System A's meaning too.
- Keep the same generation model and the same `prompts.SYSTEM_PROMPT`. If B or C needs a
  different prompt shape, record it as a deviation.
