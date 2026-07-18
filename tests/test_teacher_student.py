from pathlib import Path
import sys
import types

import numpy as np
import torch

from silent_speech_interpretability.models.teachers.ssl_teacher import SSLTeacher, relative_segment_pool
from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent
from silent_speech_interpretability.models.teachers.teacher_targets import (
    common_teacher_pairs,
    load_teacher_targets,
    make_class_structured_targets,
    save_teacher_targets,
    teacher_arrays,
)
from silent_speech_interpretability.training.train_articulatory_student import target_alignment_loss


def test_teacher_target_round_trip(tmp_path: Path):
    labels = np.array([0, 1, 0])
    targets = make_class_structured_targets(labels, target_dim=8, seed=1)
    path = save_teacher_targets(
        tmp_path / "targets.npz",
        targets,
        labels,
        np.array(["1", "1", "2"]),
        np.array(["a", "b", "c"]),
    )
    loaded = load_teacher_targets(path)
    assert loaded["target_dim"] == 8
    assert loaded["pairs"] == {("1", "a"), ("1", "b"), ("2", "c")}
    x, y = teacher_arrays(loaded, [("1", "a"), ("2", "c")])
    assert x.shape == (2, 8)
    assert y.tolist() == [0, 0]


def test_temporal_teacher_target_shape_round_trip(tmp_path: Path):
    path = save_teacher_targets(
        tmp_path / "temporal.npz",
        np.zeros((2, 12), dtype=np.float32),
        np.array([0, 1]),
        np.array(["1", "2"]),
        np.array(["a", "b"]),
        target_shape=(3, 4),
    )
    assert load_teacher_targets(path)["target_shape"] == (3, 4)


def test_relative_segment_pool_preserves_order():
    hidden = np.arange(24, dtype=np.float32).reshape(6, 4)
    pooled = relative_segment_pool(hidden, 3)
    assert pooled.shape == (3, 4)
    np.testing.assert_allclose(pooled[0], hidden[:2].mean(axis=0))
    np.testing.assert_allclose(pooled[-1], hidden[-2:].mean(axis=0))


def test_common_teacher_pairs_filters_speakers(tmp_path: Path):
    labels = np.array([0, 1])
    path = save_teacher_targets(
        tmp_path / "targets.npz",
        make_class_structured_targets(labels, target_dim=4),
        labels,
        np.array(["1", "2"]),
        np.array(["a", "b"]),
    )
    teacher = load_teacher_targets(path)
    payloads = {
        "lip": {"pairs": {("1", "a"), ("2", "b")}},
        "laser": {"pairs": {("1", "a"), ("2", "b")}},
    }
    assert common_teacher_pairs(payloads, teacher, speakers=[2]) == [("2", "b")]


def test_articulatory_student_shapes():
    model = ArticulatoryStudent(["lip", "laser"], embedding_dim=8, hidden_dim=16, bottleneck_dim=6, target_dim=5, num_classes=3)
    output = model({"lip": torch.randn(4, 8), "laser": torch.randn(4, 8)})
    assert output["target"].shape == (4, 5)
    assert output["logits"].shape == (4, 3)
    assert output["bottleneck"].shape == (4, 6)


def test_ssl_teacher_reports_dependency_availability():
    teacher = SSLTeacher("facebook/hubert-base-ls960", local_files_only=True)
    assert isinstance(teacher.available(), bool)


def test_ssl_teacher_loads_feature_extractor_without_tokenizer(monkeypatch):
    class FakeFeatureExtractor:
        @staticmethod
        def from_pretrained(model_name, local_files_only=False):
            return (model_name, local_files_only)

    class FakeModel:
        @staticmethod
        def from_pretrained(model_name, local_files_only=False):
            return FakeModel()

        def to(self, device):
            return self

        def eval(self):
            return self

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoFeatureExtractor = FakeFeatureExtractor
    fake_transformers.AutoModel = FakeModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "librosa", types.ModuleType("librosa"))

    teacher = SSLTeacher("facebook/hubert-base-ls960", local_files_only=True).load()

    assert teacher.processor == ("facebook/hubert-base-ls960", True)
    assert isinstance(teacher.model, FakeModel)


def test_target_alignment_loss_is_per_sample_squared_distance():
    predicted = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    # First sample distance is 0; second is 2; mean per-sample distance is 1.
    assert torch.isclose(target_alignment_loss(predicted, teacher), torch.tensor(1.0))
