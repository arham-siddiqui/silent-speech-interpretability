.PHONY: test manifest baseline cv cleanup

test:
	python3 -m pytest -q

manifest:
	python3 scripts/01_build_manifest.py --config configs/defaults.yaml

baseline:
	python3 scripts/02_reproduce_baseline.py --config configs/defaults.yaml

cv:
	python3 scripts/03_run_speaker_cv.py --config configs/defaults.yaml

report:
	python3 scripts/04_make_baseline_report.py --config configs/defaults.yaml

cleanup:
	python3 scripts/00_cleanup_repo.py
