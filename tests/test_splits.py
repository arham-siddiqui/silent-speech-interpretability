from silent_speech_interpretability.data.splits import make_speaker_kfold_splits, validate_speaker_disjoint
from silent_speech_interpretability.data.synthetic import make_synthetic_manifest


def test_fixed_speaker_disjoint_validation():
    manifest = make_synthetic_manifest(num_speakers=6, classes_per_speaker=3)
    train = manifest[manifest.user_id.isin([1, 2])]
    val = manifest[manifest.user_id.isin([3])]
    test = manifest[manifest.user_id.isin([4, 5, 6])]
    validate_speaker_disjoint(train, val, test)


def test_every_speaker_appears_once_in_test_fold():
    manifest = make_synthetic_manifest(num_speakers=20, classes_per_speaker=3)
    folds = make_speaker_kfold_splits(manifest, num_folds=5, seed=42)
    test_speakers = [speaker for fold in folds for speaker in fold["test_speakers"]]
    assert sorted(test_speakers) == list(range(1, 21))
    assert len(test_speakers) == len(set(test_speakers))
