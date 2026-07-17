"""Optional audio-teacher wrappers and target storage helpers."""

from .teacher_targets import (
    common_teacher_pairs,
    load_teacher_targets,
    make_class_structured_targets,
    save_teacher_targets,
    teacher_arrays,
)

__all__ = [
    "save_teacher_targets",
    "load_teacher_targets",
    "common_teacher_pairs",
    "teacher_arrays",
    "make_class_structured_targets",
]
