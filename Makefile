.PHONY: test manifest baseline cv cleanup hubert-student-cv hubert-interpretability hubert-feature-causality hubert-temporal-interpretability hubert-temporal-sensors hubert-temporal-multitask hubert-temporal-attention prompt-manifest phonetic-alignment phonetic-probes audio-phonetic-batch wav2vec2-teacher-comparison

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
