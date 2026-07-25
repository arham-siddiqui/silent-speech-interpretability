.PHONY: test manifest baseline cv cleanup hubert-student-cv hubert-interpretability hubert-feature-causality hubert-temporal-interpretability hubert-temporal-sensors hubert-temporal-multitask hubert-temporal-attention prompt-manifest phonetic-alignment phonetic-probes audio-phonetic-batch wav2vec2-teacher-comparison phone-ctc-alignment phone-ctc-probes phone-boundary-prepare phone-boundary-audit phone-boundary-import phone-boundary-analysis external-radar-fetch external-radar-extract external-radar-audit external-radar-features external-radar-hubert-pilot external-radar-replication

EXTERNAL_RADAR_DIR := artifacts/external/radar_command_words
EXTERNAL_RADAR_ARCHIVE := $(EXTERNAL_RADAR_DIR)/wagner-2022-scientific-reports-supplement.zip
EXTERNAL_RADAR_ARCHIVE_BYTES := 1426657701

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

mouth-fold:
	python3 scripts/11_train_mouth_fold_embeddings.py --config configs/defaults.yaml --fold 0

uwb-fold:
	python3 scripts/12_train_uwb_fold_embeddings.py --config configs/defaults.yaml --fold 0

mmwave-fold:
	python3 scripts/13_train_mmwave_fold_embeddings.py --config configs/defaults.yaml --fold 0

true-cv-artifacts:
	python3 scripts/14_run_true_encoder_artifacts.py --config configs/real_embeddings.local.yaml

cleanup:
	python3 scripts/00_cleanup_repo.py

hubert-student-cv:
	python3 scripts/21_run_teacher_student_cv.py

hubert-interpretability:
	python3 scripts/22_probe_hubert_student.py
	python3 scripts/23_run_hubert_modality_attribution.py
	python3 scripts/24_generate_hubert_interpretability_report.py

hubert-feature-causality:
	python3 scripts/25_train_bottleneck_sae.py
	python3 scripts/26_rank_bottleneck_features.py
	python3 scripts/27_run_bottleneck_causal_ablation.py
	python3 scripts/28_generate_bottleneck_feature_report.py

hubert-temporal-interpretability:
	python3 scripts/29_analyze_sparse_feature_exemplars.py
	python3 scripts/30_extract_temporal_hubert_targets.py --local-files-only
	python3 scripts/31_run_temporal_hubert_student_cv.py
	python3 scripts/22_probe_hubert_student.py --teacher-targets artifacts/teacher_targets/facebook_hubert-base-ls960_temporal4_targets.npz --student-dir artifacts/students/hubert_temporal4_cv --activations-dir artifacts/activations/hubert_temporal4_cv --results-output reports/results/hubert_temporal_student_probe_results.csv --summary-output reports/results/hubert_temporal_student_probe_summary.csv --report-output reports/hubert_temporal_student_probes.md
	python3 scripts/25_train_bottleneck_sae.py --activations-dir artifacts/activations/hubert_temporal4_cv --output-dir artifacts/sae/hubert_temporal4_bottleneck --results-output reports/results/hubert_temporal_bottleneck_sae_results.csv
	python3 scripts/26_rank_bottleneck_features.py --activations-dir artifacts/activations/hubert_temporal4_cv --sae-dir artifacts/sae/hubert_temporal4_bottleneck --output reports/results/hubert_temporal_bottleneck_feature_rankings.csv
	python3 scripts/27_run_bottleneck_causal_ablation.py --activations-dir artifacts/activations/hubert_temporal4_cv --student-dir artifacts/students/hubert_temporal4_cv --sae-dir artifacts/sae/hubert_temporal4_bottleneck --rankings reports/results/hubert_temporal_bottleneck_feature_rankings.csv --probe-results reports/results/hubert_temporal_student_probe_results.csv --output reports/results/hubert_temporal_bottleneck_causal_ablation.csv
	python3 scripts/33_generate_temporal_feature_report.py
	python3 scripts/32_generate_temporal_interpretability_report.py

hubert-temporal-sensors:
	python3 scripts/34_extract_temporal_sensor_activations.py
	python3 scripts/35_run_temporal_sensor_student_cv.py
	python3 scripts/36_probe_temporal_articulation.py
	python3 scripts/37_generate_temporal_sensor_report.py

hubert-temporal-multitask:
	python3 scripts/38_run_multitask_temporal_sensor_cv.py
	python3 scripts/36_probe_temporal_articulation.py
	python3 scripts/37_generate_temporal_sensor_report.py

hubert-temporal-attention:
	python3 scripts/38_run_multitask_temporal_sensor_cv.py --model-type modality_attention --experiment-name Modality-Attention --previous-label "Multitask states" --current-label "Attention states" --checkpoint-suffix temporal_sensor_attention --progress-label ATTENTION_TEMPORAL_CV --output-dir artifacts/students/temporal_sensor_attention_cv --output reports/results/temporal_sensor_attention_cv.csv --sweep-output reports/results/temporal_sensor_attention_sweep.csv --baseline-results reports/results/temporal_sensor_multitask_cv.csv --report-output reports/temporal_sensor_attention.md --figure-output reports/figures/temporal_sensor_attention_tradeoff.svg
	python3 scripts/39_analyze_temporal_attention.py
	python3 scripts/36_probe_temporal_articulation.py
	python3 scripts/37_generate_temporal_sensor_report.py

prompt-manifest:
	python3 scripts/43_audit_audio_prompt_cohorts.py --local-files-only
	python3 scripts/40_build_prompt_manifest.py
	python3 scripts/41_build_pronunciation_manifest.py

phonetic-alignment: prompt-manifest
	python3 scripts/42_align_audio_prompts.py --skip-vowels --local-files-only
	python3 scripts/44_build_phonetic_segment_targets.py

phonetic-probes: phonetic-alignment
	python3 scripts/45_probe_temporal_phonetics.py
	python3 scripts/46_generate_phonetic_probe_report.py

audio-phonetic-batch: phonetic-probes

wav2vec2-teacher-comparison:
	python3 scripts/30_extract_temporal_hubert_targets.py --model-name facebook/wav2vec2-base-960h --local-files-only --output artifacts/teacher_targets/facebook_wav2vec2-base-960h_temporal4_targets.npz --audit-output reports/results/wav2vec2_temporal_target_audit.csv
	python3 scripts/35_run_temporal_sensor_student_cv.py --teacher-targets artifacts/teacher_targets/facebook_wav2vec2-base-960h_temporal4_targets.npz --output-dir artifacts/students/wav2vec2_temporal_sensor_cv --output reports/results/wav2vec2_temporal_sensor_student_cv.csv --report-output artifacts/wav2vec2_temporal_sensor_alignment_draft.md
	python3 scripts/47_generate_audio_teacher_comparison.py

phone-ctc-alignment: phonetic-alignment
	python3 scripts/48_align_audio_phones.py --local-files-only
	python3 scripts/49_build_phone_ctc_segment_targets.py
	python3 scripts/49_build_phone_ctc_segment_targets.py --phone-intervals artifacts/forced_alignment/interpolated_phone_intervals.csv --output artifacts/forced_alignment/uniform_sentence_targets_matched.npz --report-output reports/results/uniform_sentence_target_audit_matched.md
	python3 scripts/49_build_phone_ctc_segment_targets.py --minimum-confidence 0.02 --maximum-phone-error 1.0 --maximum-fallback-fraction 0.5 --output artifacts/forced_alignment/phone_ctc_sentence_targets_lenient.npz --report-output reports/results/phone_ctc_target_audit_lenient.md
	python3 scripts/49_build_phone_ctc_segment_targets.py --minimum-confidence 0.15 --maximum-phone-error 0.5 --minimum-center-in-word 0.5 --maximum-fallback-fraction 0.25 --output artifacts/forced_alignment/phone_ctc_sentence_targets_strict_cv.npz --report-output reports/results/phone_ctc_target_audit_strict_cv.md

phone-ctc-probes: phone-ctc-alignment
	python3 scripts/45_probe_temporal_phonetics.py --targets artifacts/forced_alignment/phone_ctc_sentence_targets.npz --minimum-confidence 0 --output reports/results/phone_ctc_sentence_probe_results.csv --summary-output reports/results/phone_ctc_sentence_probe_summary.csv
	python3 scripts/45_probe_temporal_phonetics.py --targets artifacts/forced_alignment/uniform_sentence_targets_matched.npz --minimum-confidence 0 --output reports/results/uniform_sentence_probe_results_matched.csv --summary-output reports/results/uniform_sentence_probe_summary_matched.csv
	python3 scripts/45_probe_temporal_phonetics.py --targets artifacts/forced_alignment/phone_ctc_sentence_targets_lenient.npz --minimum-confidence 0 --output reports/results/phone_ctc_sentence_probe_results_lenient.csv --summary-output reports/results/phone_ctc_sentence_probe_summary_lenient.csv
	python3 scripts/45_probe_temporal_phonetics.py --targets artifacts/forced_alignment/phone_ctc_sentence_targets_strict_cv.npz --minimum-confidence 0 --output reports/results/phone_ctc_sentence_probe_results_strict_cv.csv --summary-output reports/results/phone_ctc_sentence_probe_summary_strict_cv.csv
	python3 scripts/50_link_sparse_features_to_phones.py
	python3 scripts/51_generate_phone_ctc_report.py

phone-boundary-prepare:
	python3 scripts/52_prepare_phone_boundary_audit.py

phone-boundary-audit:
	python3 scripts/53_serve_phone_boundary_audit.py

phone-boundary-import:
	python3 scripts/54_import_phone_boundary_audit.py

phone-boundary-analysis: phone-boundary-import
	python3 scripts/49_build_phone_ctc_segment_targets.py --phone-intervals artifacts/forced_alignment/phone_ctc_intervals_audited.csv --output artifacts/forced_alignment/phone_ctc_sentence_targets_audited.npz --report-output reports/results/phone_ctc_target_audit_manually_audited.md
	python3 scripts/45_probe_temporal_phonetics.py --targets artifacts/forced_alignment/phone_ctc_sentence_targets_audited.npz --minimum-confidence 0 --output reports/results/phone_ctc_sentence_probe_results_audited.csv --summary-output reports/results/phone_ctc_sentence_probe_summary_audited.csv
	python3 scripts/49_build_phone_ctc_segment_targets.py --phone-intervals artifacts/forced_alignment/phone_ctc_intervals_audit_matched_uncorrected.csv --output artifacts/forced_alignment/phone_ctc_sentence_targets_audit_matched_uncorrected.npz --report-output reports/results/phone_ctc_target_audit_matched_uncorrected.md
	python3 scripts/45_probe_temporal_phonetics.py --targets artifacts/forced_alignment/phone_ctc_sentence_targets_audit_matched_uncorrected.npz --minimum-confidence 0 --output reports/results/phone_ctc_sentence_probe_results_audit_matched_uncorrected.csv --summary-output reports/results/phone_ctc_sentence_probe_summary_audit_matched_uncorrected.csv
	python3 scripts/49_build_phone_ctc_segment_targets.py --phone-intervals artifacts/forced_alignment/uniform_phone_intervals_audit_matched.csv --output artifacts/forced_alignment/uniform_sentence_targets_audit_matched.npz --report-output reports/results/uniform_sentence_target_audit_manually_matched.md
	python3 scripts/45_probe_temporal_phonetics.py --targets artifacts/forced_alignment/uniform_sentence_targets_audit_matched.npz --minimum-confidence 0 --output reports/results/uniform_sentence_probe_results_audit_matched.csv --summary-output reports/results/uniform_sentence_probe_summary_audit_matched.csv
	python3 scripts/55_generate_phone_boundary_audit_report.py

external-radar-fetch:
	mkdir -p $(EXTERNAL_RADAR_DIR)
	@if [ ! -f $(EXTERNAL_RADAR_ARCHIVE) ] || [ "$$(wc -c < $(EXTERNAL_RADAR_ARCHIVE) | tr -d ' ')" -ne "$(EXTERNAL_RADAR_ARCHIVE_BYTES)" ]; then \
		curl -L -C - --retry 5 -o $(EXTERNAL_RADAR_ARCHIVE) https://www.vocaltractlab.de/supplements/wagner-2022-scientific-reports-supplement.zip; \
	else \
		echo "External radar archive is already complete."; \
	fi
	@test "$$(wc -c < $(EXTERNAL_RADAR_ARCHIVE) | tr -d ' ')" -eq "$(EXTERNAL_RADAR_ARCHIVE_BYTES)"

external-radar-extract: external-radar-fetch
	mkdir -p $(EXTERNAL_RADAR_DIR)/extracted
	@if [ ! -f $(EXTERNAL_RADAR_DIR)/extracted/.complete ] || [ $(EXTERNAL_RADAR_ARCHIVE) -nt $(EXTERNAL_RADAR_DIR)/extracted/.complete ]; then \
		unzip -q -o $(EXTERNAL_RADAR_ARCHIVE) -d $(EXTERNAL_RADAR_DIR)/extracted; \
		touch $(EXTERNAL_RADAR_DIR)/extracted/.complete; \
	else \
		echo "External radar archive is already extracted."; \
	fi

external-radar-audit: external-radar-extract
	python3 scripts/56_audit_external_radar_corpus.py

external-radar-features: external-radar-audit
	python3 scripts/57_extract_external_radar_features.py

external-radar-hubert-pilot: external-radar-audit
	python3 scripts/58_extract_external_hubert_targets.py --local-files-only --limit 20 --output artifacts/external/radar_command_words/hubert_temporal4_targets_pilot20.npz --audit-output artifacts/external/radar_command_words/hubert_target_audit_pilot20.csv

external-radar-replication: external-radar-features
	python3 scripts/58_extract_external_hubert_targets.py --local-files-only
	python3 scripts/59_run_external_radar_student.py
