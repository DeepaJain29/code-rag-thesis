# System A — Vanilla RAG: Implementation & Evaluation Spec

Hand this whole file to Gemini (or any LLM coding assistant) as the task brief. It specifies exact parameters, datasets, evaluation, and report output — matched to arXiv:2601.08773's vector-only baseline for valid replication, extended with a code-generation evaluation leg.

---

## 1. Objective
Build "System A" — a vanilla chunk-embed-retrieve RAG baseline for code — and evaluate it on two legs:
- **Leg 1 (QA replication)**: same setup as 2601.08773's No-Graph baseline, same 3 Java repos, same 45 questions.
- **Leg 2 (generation, this project's actual RQ1)**: Pass@k code generation on CoderEval/RepoExec, then on Frappe/ERPNext.

---

## 2. Exact pipeline parameters (do not deviate without documenting why)

| Component | Exact spec |
|---|---|
| Chunking | `langchain.text_splitter.RecursiveCharacterTextSplitter`, `chunk_size=1000`, `chunk_overlap=100` (characters, not tokens) |
| Embedding model | Google Gemini embedding model (e.g. `text-embedding-004` or current Gemini embedding endpoint — pin exact model name/version used, record in report) |
| Framework | LangChain wrappers for embedding + vector store integration |
| Vector store | FAISS (local), or LangChain's default local vector store if simpler — document choice |
| Retrieval | Top-k cosine similarity, **k = 10** |
| Generation LLM | Pin one model/version, used identically for every query in every leg — record exact name |

---

## 3. Datasets to acquire

### Leg 1 — QA replication (matches 2601.08773 exactly)
| Repo | Files | Chunks (expected, ~1000-char chunking) |
|---|---|---|
| Shopizer | 1,210 Java files | ~5,403 |
| ThingsBoard | 4,521 Java files | ~32,428 |
| OpenMRS Core | 1,258 Java files | ~11,495 |

Source: public GitHub repos (search and pin exact commit/tag used — repos evolve, indexing a different snapshot than the paper used will change chunk counts). 15 questions per repo (45 total) targeting repository semantics, multi-hop architectural tracing, system-level reasoning — reconstruct or approximate from the paper's methodology; if the exact question set isn't published, note that clearly and construct a comparable set instead of guessing at the original wording.

### Leg 2 — Generation (this project's core RQ1)
- **CoderEval + RepoExec** — standard benchmark repos/tasks, download from their respective public releases.
- **Frappe/ERPNext** — comment/docstring-stripped snapshot (pin exact commit), plus a constructed held-out query set (CrossCodeEval-style: hide a call/reference, verify statically that the held-out code needs cross-file/cross-function context).

---

## 4. Build steps (what to actually code)

1. **Ingestion**: clone/download each repo at a pinned commit; walk the file tree; read source files.
2. **Chunking**: apply `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)` to each file's raw text.
3. **Embedding**: embed every chunk with the pinned Gemini embedding model; store chunk text + metadata (source file, start/end offset) alongside each vector.
4. **Indexing**: build a FAISS index per repo (or per leg — keep Leg 1's 3 repos and Leg 2's repos in separate indexes, never mixed).
5. **Query-time retrieval**: embed the incoming query with the same embedding model; retrieve top-10 nearest chunks by cosine similarity.
6. **Prompt assembly**: concatenate retrieved chunks (with source-file labels) into a fixed prompt template; keep the template identical across Leg 1 and Leg 2 except for the task instruction (QA vs. "complete/generate this code").
7. **Generation**: single call to the pinned generation LLM per query.
8. **Logging**: for every query, log — retrieved chunk IDs, retrieval latency, generation latency, input/output token counts, raw model output. This is required for the cost metrics in §7 of the main thesis doc, not optional.

---

## 5. Evaluation

### Leg 1 (QA correctness)
- Score each of the 45 answers against expected/reference answers (human judgment or a scoring rubric — document whichever is used).
- Report per-repo and overall accuracy, alongside chunk-count/coverage stats, for direct comparison against 2601.08773's published numbers.

### Leg 2 (generation quality)
- **Pass@1 / Pass@3 / Pass@5** on CoderEval/RepoExec (execute generated code against the benchmark's test harness).
- **Exact Match / Edit Similarity** on the Frappe/ERPNext held-out completions.
- **Retrieval precision/recall** against the statically-verified ground-truth relevant chunks for each held-out query.

### Cost metrics (both legs)
- Indexing wall-clock time, indexing token cost, query latency (p50/p95), query token cost, index size/storage — per repo.

---

## 6. Report output (what Gemini should produce at the end)

A structured report containing:
1. **Setup summary** — exact model names/versions, chunk counts per repo (compare to 2601.08773's reported counts — flag any mismatch and explain, e.g. repo updated since the paper's snapshot).
2. **Leg 1 results table** — per-repo accuracy, compared side-by-side with 2601.08773's own reported System-A numbers.
3. **Leg 2 results table** — Pass@k / EM / ES per dataset.
4. **Cost table** — indexing time/cost, query latency/cost, per repo.
5. **Retrieval quality table** — precision/recall on held-out queries.
6. **Known deviations** — anything that couldn't be matched exactly to 2601.08773 (e.g. repo snapshot, missing exact QA wording) and why.

---

## 7. Explicit non-goals (keep Gemini from scope-creeping this task)
- No graph construction — System A has zero graph logic, that's Systems B/C.
- No re-ranking beyond plain top-k cosine similarity.
- No hyperparameter tuning of k or chunk size beyond the pinned values — this is a fixed baseline, not something to optimize.
