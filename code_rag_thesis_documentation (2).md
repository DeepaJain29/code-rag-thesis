# Structural vs. LLM-Extracted Graph Retrieval for Repository-Level Code Generation

**M.Tech Project-III — Project Plan / PRD — Data Science & AI, IIIT Dharwad**

---

## 0. Honest framing up front

The core idea "use an AST/call-graph instead of an LLM-extracted graph for code RAG" is **not novel on its own**. A cluster of 2024–2026 papers already builds structural code graphs (calls/imports/inheritance) and retrieves from them instead of using vector-similarity chunk retrieval — GraphCoder, RepoGraph, CodexGraph, RANGER, LARGER, CGM, the Partial Dependency Graph paper, and others (full list in §3/§12).

More specifically: **arXiv:2601.08773**, *"Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs"* (Jan 2026), already runs close to the exact three-way comparison this project is built around — vector-only RAG, LLM-extracted knowledge-graph RAG, and a deterministic Tree-sitter-derived knowledge-graph RAG — on real Java codebases (Shopizer, ThingsBoard, OpenMRS Core), reporting indexing time, query latency, coverage, cost, and correctness. Their finding: the AST-derived graph wins on cost/reliability and matches or beats the LLM-extracted graph on correctness. **This is the result this project set out to demonstrate**, so presenting it as-is would be a reproduction, not new work.

**What survives as a genuine contribution, checked specifically against 2601.08773:**

1. **Task type**: 2601.08773 evaluates architecture/code-tracing **QA correctness**. This project targets **code generation quality** (Pass@k) — a different downstream task, where retrieved noise (per AllianceCoder's own finding, §3.3) actively hurts output in a way QA-correctness scoring can't see.
2. **Retrieval-scoping ablation (RQ3)**: not covered by 2601.08773's traversal method at all.
3. **Undocumented-code condition (RQ4)**: 2601.08773's repos are reasonably well-maintained open-source projects. This project explicitly tests a comment/docstring-stripped codebase to approximate real legacy code.
4. **Language/ecosystem**: Java/Spring-only in 2601.08773 vs. Python/Frappe here — and, as a stretch objective, multiple languages within this project itself (RQ5, §2).

**Scope**: a **replication-and-extension study** — confirm 2601.08773's cost/reliability finding in a new task setting and extend it along three axes it doesn't cover (ablation, undocumented-code, cross-language). This is a legitimate, standard research pattern and realistic for a two-semester M.Tech thesis. Everything below is written as a forward-looking plan (problem → design → build → evaluate) and should be read that way end to end, regardless of what stage any individual piece happens to be at when read.

---

## 1. Problem Statement

LLMs generate incorrect or non-compilable code on real repositories because they lack repository-specific context — internal APIs, call relationships, type definitions — needed to write code consistent with the existing codebase. RAG supplies this context, but the dominant text-domain approach (chunk, embed, retrieve by cosine similarity) is a poor fit for code: it breaks functions/classes across chunk boundaries, has no notion of call/import relationships, and cannot resolve cross-file dependencies. GraphRAG fixes this for text via LLM-extracted entity graphs, at large indexing cost and multi-stage hallucination risk — largely unnecessary for code, since code's dependency structure is exactly recoverable via parsing and static analysis, without any LLM inference.

**Question this project answers**: for repository-level code generation on a large, undocumented codebase, how do (a) vanilla chunk-embed RAG, (b) LLM-extraction GraphRAG (adapted to code), and (c) a deterministic, static-analysis-derived code graph compare — in generation quality, retrieval precision, indexing cost, query latency, and token consumption — under one controlled protocol, and does the comparison hold across languages?

---

## 2. Research Questions

- **RQ1 (Quality)**: Does structural (AST + call/import graph) retrieval produce higher Pass@k and higher retrieval precision/recall than vanilla RAG and LLM-extraction GraphRAG, for repository-level code generation and cross-file impact-analysis queries?
- **RQ2 (Cost)**: What is the indexing-time, indexing-token-cost, query-latency, and query-token-cost differential between the three paradigms, on the same codebase?
- **RQ3 (Retrieval scoping, generalizable)**: How does retrieval quality vary with traversal strategy — hop depth, edge-type filtering, embedding-based re-ranking — and does the *optimal* strategy correlate with structural properties of the codebase (e.g. average node degree, clustering coefficient, call-graph diameter), rather than being a fixed number tied to one repo? Evaluated across at least two structurally different codebases, not one.
- **RQ4 (Undocumented setting)**: Does the relative advantage of structural graph retrieval hold, shrink, or grow when the codebase lacks docstrings/comments, compared to well-documented repositories?
- **RQ5 (Cross-language generalization — stretch objective)**: Does the RQ1/RQ2 result hold consistently across languages with different semantics (e.g. Python's dynamic typing vs. a statically-typed language), or does the structural-graph advantage vary by language? Treated as secondary/time-permitting — RQ1–RQ4 are the core commitment; RQ5 should not be allowed to compress the time budgeted for them.

---

## 3. Related Work (summary — full chapter in the companion Phase 0 document)

The complete related-work chapter (organized prose, all citations) lives in `phase0_scoping_and_related_work.md` and should be treated as the literature-review source of truth; this section is a condensed pointer, not a duplicate.

- **§3.0 — Most directly overlapping work**: arXiv:2601.08773 (see §0). Read and cite first.
- **§3.1 — Foundational RAG/GraphRAG (text domain)**: Lewis et al. (RAG), Edge et al. (Microsoft GraphRAG), LazyGraphRAG, Fast-GraphRAG, HippoRAG, LightRAG, RAPTOR, KGP, KET-RAG, PathRAG, GraphRAG-Bench, and related surveys/evaluations (Han et al., RAG-vs-GraphRAG systematic evaluation, NodeRAG, ContextRAG).
- **§3.2 — Structural/graph-based code retrieval**: GraphCoder, RepoGraph, CodexGraph, CGM, RANGER, LARGER, Partial Dependency Graph paper, ICSE'25 Knowledge-Graph-Based Repo-Level Code Generation, Context-Augmented Programming Knowledge Graphs, RepoMind.
- **§3.3 — What to retrieve / learned retrievers**: AllianceCoder (arXiv:2503.20589 — the key finding that retrieved *similar code* hurts generation while in-context code + API descriptions help), RepoCoder, De-Hallucinator, RepoMinCoder, R2C2-Coder, CoCoMIC, RLCoder, AlignCoder, Adaptive Critical Token-Aware Retrieval, SPENCER, SECRET, GraphCodeBERT, UniXcoder.
- **§3.4 — Benchmarks**: CoderEval, RepoExec, CrossCodeEval, RepoBench, ComplexCodeEval, and the 2025/2026 survey of retrieval-augmented code generation.

43 sources total, numbered in §12. Re-verify against live literature within 1–2 weeks of submission — this field moves fast, as 2601.08773's own recency demonstrates.

---

## 4. What this project contributes

- A **replication of 2601.08773's cost/reliability finding**, on a **different task** (code generation, not QA) and a **different ecosystem** (Python/Frappe, with Java/2601.08773's own numbers as an external reference point rather than a full from-scratch rebuild).
- An **undocumented-code condition** neither 2601.08773 nor the standard benchmarks test.
- A **retrieval-scoping ablation** designed to produce a generalizable relationship (optimal traversal vs. codebase structure) rather than a single repo's magic number — addressing a real weakness flagged during scoping of an earlier, single-repo version of this ablation.
- A **stretch cross-language check** (RQ5), feasible within this project's scope because the parsing layer is designed for multi-language support from the outset (see §6, §8) rather than being added as an afterthought.

---

## 5. Datasets

| Use | Dataset | Notes |
|---|---|---|
| Primary generation-quality benchmark | **CoderEval + RepoExec** | Used by AllianceCoder — enables direct Pass@k comparison against a published number. |
| Cross-file dependency benchmark | **CrossCodeEval** (Python/Java/TS/C# subsets) | Provides a static-analysis-verified query-construction method to mirror for the Frappe/ERPNext query set. |
| QA-replication leg (optional, direct number-for-number check) | **Shopizer (1,210 files/5,403 chunks), ThingsBoard (4,521/32,428), OpenMRS Core (1,258/11,495)** — 2601.08773's own repos, 15 QA questions each (45 total) | Only covers architecture/multi-hop QA, not generation — use to sanity-check System A/B/C against 2601.08773's own numbers; does not substitute for the Pass@k generation evaluation (RQ1). |
| Legacy/undocumented setting (RQ4) | **Comment/docstring-stripped Frappe/ERPNext** | Decided: the real target legacy codebase isn't publishable, so a large, structurally complex, multi-module public repo stands in as a reproducible proxy. Comment/docstring stripping done programmatically (Python `ast` + `tokenize`, or Tree-sitter for a language-agnostic version). |
| RQ3 multi-repo ablation | **Frappe/ERPNext + at least one structurally different second repo** (e.g. Django, for a more "library" than "business app" call-graph shape) | Needed so the traversal-scoping finding is a relationship (optimal config vs. graph structure), not a single-repo number. |
| RQ5 stretch (cross-language) | **A statically-typed-language repo of comparable scale** (Java candidate, enabling partial cross-reference with 2601.08773's own Java numbers where task types allow) | Time-permitting only — do not let this delay RQ1–RQ4's core evaluation. |

---

## 6. Systems compared (three arms)

### System A — Vanilla RAG (baseline)
Exact parameters, matched to 2601.08773's own vector-only baseline for a valid replication:
- **Chunking**: `RecursiveCharacterTextSplitter`, chunk size 1,000 characters, overlap 100 characters (note: characters, not tokens).
- **Embedding + stack**: Google Gemini embedding model family, via LangChain.
- **Vector store**: local vector store (FAISS acceptable; document exact choice).
- **Top-k**: k = 10.
- **Generation**: retrieved chunks concatenated into prompt, single LLM call.
No graph, no structure. Cheapest, weakest baseline — establishes the floor.

### System B — LLM-extraction GraphRAG adapted to code (baseline)
Reproduce the Microsoft GraphRAG pipeline (Edge et al.) applied to code chunks: LLM extracts entities/relations from code chunks (prompted, not parsed) → graph built from extracted triples → Leiden clustering → LLM community summarization → dual local/global embedding index → query-time retrieval. Instrument extraction coverage explicitly (files/chunks skipped or malformed) — this directly replicates 2601.08773's indexing-incompleteness finding and is one of the two headline metrics being replicated.

### System C — Structural graph RAG (this project's focus system)
The build sequence, as a set of project phases (§10), not a status report of what exists:

1. **Multi-language AST parsing** (Tree-sitter) — file/class/function-level nodes, across the target languages for this project (Python for the core Frappe/ERPNext work; additional languages for the RQ5 stretch objective).
2. **Symbol resolution** — build a symbol table (functions/classes/files indexed by name, containing file, and module path) and resolve each raw call/import/inheritance reference against it, classifying each resolution outcome (clean match, module-qualified match, external/library reference, superclass/inheritance-based match, ambiguous — multiple candidates, unresolved — no candidate found). This is the step that turns "a function named `process` is called somewhere" into "this specific call refers to this specific definition" — see the note on why this matters in §11 (dynamic dispatch / DI limitation).
3. **Framework-aware entry-point detection** — identify route handlers / entry points for the frameworks present in the target codebase (Frappe's own hook/decorator conventions for the primary dataset; FastAPI and Spring Boot conventions if the RQ5 stretch languages are pursued), used to anchor dependency-closure-style retrieval.
4. **Hierarchical embeddings** — function, class, and file level, each with its own embedding (file/class level = compact structural rollup: imports, contained signatures — not raw concatenated code, to stay within embedding-model token limits).
5. **Hop-scoped retrieval** — embed query, nearest-match to a starting node, k-hop BFS filtered by edge type, candidates re-ranked by embedding similarity to the query, top-N retained. Hop depth / edge-type filter / re-ranking choice is the RQ3 ablation's variable; a single fixed configuration is used for the RQ1/RQ2 headline comparison (kept clearly separate from the ablation's own results, per §9's scope-lock).
6. **Single LLM call for final generation** — retrieved context (signatures + snippets + resolved dependencies) assembled into the prompt. Zero LLM calls anywhere upstream of this step.

### Shared / controlled variables across A, B, and C
Same final-generation LLM (pinned model/version), same embedding model wherever embeddings are used, same codebase snapshot, same query set, same hardware/budget accounting for cost metrics. Any quality difference between systems must trace to retrieval design, not to an uncontrolled variable.

---

## 7. Metrics

**Quality**: Pass@1/@3/@5 (CoderEval/RepoExec, comparable to AllianceCoder's published numbers); Exact Match / Edit Similarity (CrossCodeEval-style, comparable to RepoBench/Qwen2.5-Coder numbers); retrieval precision/recall against statically-verified ground truth; API precision/recall.

**Cost** (the axis most prior structural-graph papers under-report relative to an LLM-extraction baseline, since most don't build one): indexing wall-clock time; indexing token cost ($ and raw tokens — trivial for A/C, the expensive number for B); query latency (p50/p95); query token cost; peak memory/storage.

**Reliability**: provenance traceability audit for System B (can every retrieved "fact" be traced back to source without contradiction — sampled manual check, cross-referenced against 2601.08773's file-skipping finding); trivially 100% for System C by construction, stated plainly as part of the honest comparison rather than a claim needing its own experiment.

**RQ3 ablation**: retrieval precision/recall and Pass@k as a function of hop depth × edge-type filter × re-ranking method, run on **at least two structurally distinct codebases**, reported alongside a structural-property measurement (node degree, clustering coefficient, graph diameter) per codebase so the finding can be framed as a trend/correlation, not a single number.

---

## 8. Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| Parsing | Tree-sitter, multi-language grammars (Python primary; additional languages for RQ5) | Mature, consistent grammar interface across languages — the reason a cross-language stretch objective is realistic within this project's scope. |
| Symbol resolution | Custom resolver: symbol-table construction (by name/file/module) + edge resolution pass with explicit resolution-type classification (clean/module-qualified/external/inheritance-based/ambiguous/unresolved) | Language-specific import/scope semantics mean this layer needs per-language logic even though parsing itself is language-agnostic — budget accordingly per language attempted. |
| Framework detection | Pattern-based entry-point detectors (decorator/route-pattern matching) per framework | Anchors closure-based retrieval to real entry points rather than treating every node as equally likely to be queried. |
| Graph construction | NetworkX → evaluate migration to a graph DB (e.g. Neo4j) if node count strains in-memory performance at Frappe/ERPNext's scale | Benchmark both once actual node count is known; don't assume either is sufficient in advance. |
| Community detection (System B only) | Leiden (`leidenalg`/`python-igraph`) | Standard choice, matches Edge et al.'s original pipeline. |
| Embedding model | Compact code-specific embedding model (~1024-dim class), identical across Systems A and C | Isolates the retrieval-structure variable — embedding quality held constant. |
| Vector index | FAISS (HNSW) | Standard at this scale; tune HNSW parameters once actual node count is known. |
| Sparse/exact index | BM25 / inverted index over identifiers | For hybrid dense+sparse retrieval on exact symbol-name queries. |
| LLM (extraction, System B) | One capable instruction model, pinned before Phase 3 | Indexing-cost figures are meaningless without the exact model/version documented. |
| LLM (final generation, all systems) | Same model/version across A/B/C | Critical control. |
| Evaluation harness | Pass@k / EM / ES scorer, built to CoderEval/RepoExec/CrossCodeEval conventions | Budget real implementation time — this is a full component of the project, not a wrapper. |
| Compute | VS Code (dev) + Google Colab (GPU-heavy embedding/indexing runs) | Existing working setup. |

---

## 9. Architecture (high level)

```
                        +---------------------------------------+
                        |          Target codebase(s)            |
                        |  Frappe/ERPNext (comment-stripped);     |
                        |  + second repo for RQ3; + RQ5 language  |
                        +-------------------+---------------------+
                                            |
        +-------------------+--------------+-------------+-------------------+
        |                   |                             |                   |
   System A            System B                       System C           (shared)
  Vanilla RAG    LLM-extraction GraphRAG        Structural graph RAG     Eval harness
        |                   |                             |
  chunk+embed        LLM extract entities/        AST parse -> symbol
                      relations -> graph ->        resolution -> entry-
                      Leiden -> LLM summarize       point detection ->
                      -> dual embed index           hierarchical embed ->
                                                     hop-scoped retrieval
        |                   |                             |
        +-------------------+--------------+-------------+-------------------+
                                            |
                                 Same final-generation LLM
                                            |
                         Pass@k / EM / ES / retrieval P-R /
                         indexing time+cost / query latency+cost /
                         provenance audit / RQ3 structural-property
                         correlation / RQ5 cross-language check
```

---

## 10. Experimental flow / phases

1. **Phase 0 — Scoping & literature freeze**: citation list finalized, related-work chapter drafted, System A/B/C definitions locked (see companion Phase 0 document).
2. **Phase 1 — System A (vanilla RAG)**: chunking + embedding + retrieval, sanity-checked on CoderEval/RepoExec before touching the target codebase(s).
3. **Phase 2 — System C build-out**: multi-language AST parsing → symbol resolution (symbol tables + edge resolution with explicit resolution-type classification) → framework-aware entry-point detection → hierarchical embeddings → hop-scoped retrieval. This is the largest phase — sequence it as parsing first (fast to validate), then resolution (the part most likely to need iteration per language/framework), then embeddings/retrieval on top.
4. **Phase 3 — System B (LLM-extraction GraphRAG for code)**: implement/adapt the Microsoft GraphRAG pipeline for code chunks. Budget generously — most expensive and error-prone system to get right, and per 2601.08773, extraction-coverage bugs (skipped files/chunks) are a documented failure mode to instrument for, not a hypothetical.
5. **Phase 4 — Query set construction**: CrossCodeEval-style held-out query sets for Frappe/ERPNext and the RQ3 second repo, with static verification that each query genuinely requires cross-file/cross-function context.
6. **Phase 5 — Full three-way evaluation**: quality + cost metrics (§7) across A/B/C, on both standard benchmarks and the target codebase(s). Uses System C's fixed default retrieval configuration — not the ablation's best-performing config (kept separate, see Phase 6).
7. **Phase 6 — RQ3 ablation**: hop-depth × edge-filter × re-ranking grid, run on Frappe/ERPNext and the second RQ3 repo, reported alongside each repo's structural properties.
8. **Phase 7 — Provenance/hallucination audit**: sampled manual audit of System B's extracted facts vs. source; System C's audit is near-trivial by construction but documented plainly for the comparison.
9. **Phase 8 — RQ5 stretch (time-permitting)**: repeat a scoped subset of Phase 2/5 for one additional language, cross-referencing 2601.08773's own Java numbers where the task types allow direct comparison.
10. **Phase 9 — Write-up**: results, honest discussion of where the replication of 2601.08773 holds/breaks on the new axes, limitations, future work.

Two semesters remaining: Phases 1–3 (all three systems reaching a working state) fill semester one; Phases 4–9 (query construction, evaluation, ablation, audit, optional stretch, write-up) fill semester two. Phase 8 (RQ5) is explicitly the first thing to cut if semester one overruns — RQ1–RQ4 are the core commitment.

---

## 11. Limitations to state explicitly (do not hide)

- **Dynamic dispatch / dependency injection**: symbol resolution as scoped in §6/§8 is inherently incomplete for polymorphic calls, reflection, and framework-wired dependency injection (e.g. FastAPI's `Depends(...)`, Spring's `@Autowired`) — these connections aren't literal identifiers in the AST, so a generic resolver misses them. State this as a known source of recall loss in System C rather than presenting the structural graph as ground truth; extending framework-specific resolvers to catch these patterns is a natural extension but is not committed scope unless time permits (candidate for a "future work" note rather than a required phase).
- **Comment-stripped proxy validity**: using a comment/docstring-stripped public repo as a stand-in for genuinely undocumented legacy code is not a perfect match — naming conventions, code quality, and architectural consistency in curated open-source repos tend to be better than in real legacy systems. State this as a threat to validity.
- **System B fidelity**: exact adherence to Edge et al.'s original GraphRAG implementation depends on engineering choices (prompt design, extraction model) not fully specified in the original paper — document adaptation decisions clearly.
- **Replication framing discipline**: the primary result (System C beating System B on cost/reliability) is expected to be consistent with 2601.08773, not a surprise — write the results chapter as "replicates on a new axis," not "we discovered X." If results diverge (e.g. System B performs relatively better on Python/Frappe than on Java), that divergence is the more interesting finding and should be highlighted, not treated as noise.
- **RQ3 generalizability**: a single-codebase ablation would only support a case-study-level claim ("hop=2 worked best here"); running it across at least two structurally different repos and correlating the optimal configuration with a structural property is what makes RQ3 a generalizable finding rather than a repo-specific number — do not collapse back to a single-repo ablation under time pressure without adjusting the claim's strength accordingly.
- **Timeline risk**: three full systems plus a multi-repo ablation plus an optional stretch objective is a substantial two-semester scope — RQ5 (§2) and the framework-specific DI resolver (above) are the two components explicitly marked as first-to-cut if Phase 2 or Phase 3 overruns.

---

## 12. Future Work — Deployment Extension (explicitly beyond thesis scope)

After the thesis experiments (Phases 1–8) are complete, System C's pipeline can be packaged into a standalone, deployable tool — a project layer sitting on top of the research, not part of the graded thesis work itself:

- **Storage**: migrate the graph from NetworkX to a proper graph database (**Neo4j**), enabling persistent, queryable storage instead of an in-memory research artifact.
- **Query interface**: expose retrieval/traversal via **Cypher queries** — e.g. "find all functions calling X within 2 hops," "show the dependency closure of entry point Y" — directly usable by a human or another tool, not just internally by the retrieval pipeline.
- **Delivery form**: package as an extension/app (e.g. IDE extension, CLI tool, or a small service) that lets a user point at a codebase and extract graph-based features/context from it on demand — a practical continuation of the entry-point-anchored closure idea already used for retrieval.
- **Status**: explicitly **out of scope for the thesis deliverable itself** — this belongs in the thesis's "Future Work" chapter as a natural next step the research enables, not as a phase with its own evaluation/timeline commitment. Do not let this pull effort away from Phases 1–8 (§10); revisit only after the core thesis is complete.

---

## 13. Full reference list

0. Anon., 2026, arXiv:2601.08773 — Reliable Graph-RAG for Codebases: AST-Derived Graphs vs LLM-Extracted Knowledge Graphs — the single most important citation in this project; read and cite first.
1. Lewis et al., 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
2. Edge et al., 2024, arXiv:2404.16130 — From Local to Global: A Graph RAG Approach to Query-Focused Summarization (Microsoft GraphRAG)
3. LazyGraphRAG, Microsoft Research, 2024
4. Fast-GraphRAG, CircleMind-AI, 2024
5. HippoRAG, Gutierrez et al., 2024
6. LightRAG, Guo et al., 2024
7. RAPTOR, Sarthi et al., 2024
8. KGP, Wang et al., 2024
9. KET-RAG, Huang et al., 2024/2025
10. PathRAG, Chen et al.
11. Xiang et al., 2025/2026, arXiv:2506.05690 — When to use Graphs in RAG: A Comprehensive Analysis for GraphRAG (GraphRAG-Bench, ICLR'26)
12. Liu et al., ASE 2024 — GraphCoder: Enhancing Repository-Level Code Completion via Coarse-to-fine Retrieval Based on Code Context Graph
13. Ouyang et al., 2024, arXiv:2410.14684 — RepoGraph: Enhancing AI Software Engineering with Repository-Level Code Graph
14. 2024, arXiv:2408.03910 — CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases
15. 2025, arXiv:2505.16901 — Code Graph Model (CGM): A Graph-Integrated Large Language Model for Repository-Level Software Engineering Tasks
16. 2025, arXiv:2509.25257 — RANGER: Repository-Level Agent for Graph-Enhanced Retrieval
17. 2026, arXiv:2605.16352 — LARGER: Lexically Anchored Repository Graph Exploration and Retrieval
18. 2026, arXiv:2608.01927 — Effective and Efficient Context Retrieval via Partial Dependency Graph for Repository-Level Code Generation
19. ICSE 2025, LLM4Code workshop — Knowledge Graph Based Repository-Level Code Generation
20. ACL 2025 — Context-Augmented Code Generation Using Programming Knowledge Graphs
21. ICPC 2026, DOI 10.1145/3794763.3794823 — RepoMind: Enhancing Repository-Level Code Generation via LLM Reasoning over Structured Repository Documentation
22. Gu et al., 2025, arXiv:2503.20589 — What to Retrieve for Effective Retrieval-Augmented Code Generation? An Empirical Study and Beyond (AllianceCoder)
23. Zhang et al., 2023 — RepoCoder
24. De-Hallucinator (cited via RepoMinCoder related work)
25. Li et al., 2024 — RepoMinCoder: Improving Repository-Level Code Generation Based on Information Loss Screening
26. 2024 — R2C2-Coder
27. 2022 — CoCoMIC: Code Completion By Jointly Modeling In-file and Cross-file Context
28. Wang et al., 2024 — RLCoder (reward-guided retriever training, CrossCodeEval)
29. Jiang et al., ASE 2025 — AlignCoder: Aligning Retrieval with Target Intent for Repository-Level Code Completion
30. 2026, arXiv:2609.01601 — Adaptive Critical Token-Aware Retrieval for Repository-Level Code Generation
31. Gu et al. — SPENCER: Self-Adaptive Model Distillation for Efficient Code Retrieval
32. Gu et al., 2024, arXiv:2412.11728 — SECRET: Towards Scalable and Efficient Code Retrieval via Segmented Deep Hashing
33. Guo et al., 2020, arXiv:2009.08366 — GraphCodeBERT: Pre-training Code Representations with Data Flow
34. Guo et al., 2022, arXiv:2203.03850 — UniXcoder: Unified Cross-Modal Pre-training for Code Representation
35. Ding et al., NeurIPS 2023 (D&B), arXiv:2310.11248 — CrossCodeEval: A Diverse and Multilingual Benchmark for Cross-File Code Completion
36. Liu et al., 2023 — RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems
37. Feng et al., 2024 — ComplexCodeEval
38. Tao et al., 2025/2026, arXiv:2510.04905 — Retrieval-Augmented Code Generation: A Survey with Focus on Repository-Level Approaches
39. Qwen2.5-Coder Technical Report, 2024, arXiv:2409.12186
40. Han et al., 2024/2025, arXiv:2501.00309 — Retrieval-Augmented Generation with Graphs (GraphRAG survey/taxonomy)
41. arXiv:2502.11371 — RAG vs. GraphRAG: A Systematic Evaluation and Key Insights
42. 2025, arXiv:2504.11544 — NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes
43. 2026, arXiv:2605.19735 — ContextRAG: Extraction-Free Hierarchical Graph Construction for Retrieval-Augmented Generation

*All citations were gathered via web search and should be re-verified (exact author lists, venues, final arXiv versions) directly from each paper before submission — search-snippet metadata is not always complete or final. Re-run a targeted search for "AST-derived graph code RAG" and "structural code graph vs LLM-extracted knowledge graph" within 1–2 weeks of the actual deadline, given how recently 2601.08773 itself appeared.*
