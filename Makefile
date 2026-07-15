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

compare:
	python3 scripts/05_compare_legacy_baseline.py --config configs/defaults.yaml

sanity:
	python3 scripts/06_evaluation_sanity_audit.py --config configs/defaults.yaml

prepare-true-cv:
	python3 scripts/07_prepare_true_encoder_cv.py --config configs/defaults.yaml

true-cv:
	python3 scripts/08_run_true_encoder_cv.py --config configs/defaults.yaml

lip-fold:
	python3 scripts/09_train_lip_fold_embeddings.py --config configs/defaults.yaml --fold 0

laser-fold:
	python3 scripts/10_train_laser_fold_embeddings.py --config configs/defaults.yaml --fold 0

cleanup:
	python3 scripts/00_cleanup_repo.py
