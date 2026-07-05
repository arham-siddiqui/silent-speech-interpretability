"""Prototype and learned late-fusion methods."""

from .consistency_weighted import borda_count_fusion, consistency_weighted_fusion, equal_weight_fusion
from .prototype import PrototypeClassifier

__all__ = [
    "PrototypeClassifier",
    "equal_weight_fusion",
    "borda_count_fusion",
    "consistency_weighted_fusion",
]
