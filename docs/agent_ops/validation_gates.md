# Validation Gates

Run before claiming work is complete:

```bash
python -m pytest -q
python scripts/smoke_test_kaggle_path.py
python scripts/make_submission.py --data-dir <small_or_mock_test_dir> --output submission.csv --debug
python -m biohub_ct.submission.validator submission.csv
```

For metric changes, also run:

```bash
python -m biohub_ct.metrics.probes
```

If optional official metric dependencies are installed, compare local synthetic probes against organizer code before trusting local CV.

