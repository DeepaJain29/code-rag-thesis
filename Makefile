PY ?= python

.PHONY: install bootstrap doctor index leg1 leg2-codereval leg2-repoexec leg2-frappe report clean-index-a

install:                ## install dependencies
	$(PY) -m pip install -r requirements.txt

bootstrap:              ## ONE TIME: download models + repos + datasets (only network step)
	$(PY) scripts/bootstrap.py

doctor:                 ## show what is downloaded / indexed / missing
	$(PY) scripts/doctor.py

index:                  ## build System A FAISS indexes (once per repo)
	$(PY) -m systems.system_a.index_build --leg all

leg1:                   ## System A, Leg 1 QA replication (45 questions)
	$(PY) -m systems.system_a.run_leg1

leg2-codereval:         ## pass@1/3/5 on CoderEval
	$(PY) -m systems.system_a.run_leg2 --dataset codereval --index-repo frappe

leg2-repoexec:          ## pass@1/3/5 on RepoExec
	$(PY) -m systems.system_a.run_leg2 --dataset repoexec --index-repo frappe

leg2-frappe:            ## EM/ES + retrieval P-R on the Frappe held-out set
	$(PY) scripts/build_query_set.py --repo frappe
	$(PY) -m systems.system_a.run_leg2 --dataset frappe

report:                 ## assemble results/system_a/report_system_a.md
	$(PY) -m systems.system_a.report

clean-index-a:          ## drop System A indexes only (keeps models/repos/datasets)
	rm -rf data/indexes/system_a data/.state/index_system_a*
