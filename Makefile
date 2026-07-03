.PHONY: test smoke validate probe

test:
	python -m pytest -q

smoke:
	python scripts/smoke_test_kaggle_path.py

validate:
	python -m biohub_ct.submission.validator submission.csv

probe:
	python -m biohub_ct.metrics.probes

