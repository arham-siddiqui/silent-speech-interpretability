"""Student models for sensor-to-articulatory representation learning."""

from .articulatory_student import ArticulatoryStudent
from .bottleneck import BottleneckMLP

__all__ = ["ArticulatoryStudent", "BottleneckMLP"]
