import numpy as np

from silent_speech_interpretability.models.fusion import PrototypeClassifier


def test_prototype_classifier_predicts_nearest_class():
    embeddings = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    clf = PrototypeClassifier(temperature=0.1).fit(embeddings, labels)
    preds = clf.predict(np.array([[1.0, 0.0], [0.0, 1.0]]))
    assert preds.tolist() == [0, 1]
