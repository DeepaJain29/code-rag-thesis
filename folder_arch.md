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