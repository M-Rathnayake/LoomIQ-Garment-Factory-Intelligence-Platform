"""Run the defect model on fabric images and compute DHU.

DHU (Defects per Hundred Units) = total defects / units inspected * 100.
A lower DHU means better quality.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Defect:
    """One defect found in an image."""

    label: str  # e.g. "hole", "stain", "broken_thread"
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


def load_defect_model(path: Path | None = None) -> Any:
    """Load the trained defect model (YOLO or ResNet, based on config)."""
    # TODO: Default to models/<vision.model_file>; ultralytics.YOLO(path) for yolo.
    raise NotImplementedError("load_defect_model is not implemented yet")


def detect_defects(model: Any, image: Any, confidence_threshold: float = 0.25) -> list[Defect]:
    """Find defects in a single image.

    Args:
        model: Model from :func:`load_defect_model`.
        image: File path, PIL image or numpy array.
        confidence_threshold: Ignore detections below this score.

    Returns:
        A list of :class:`Defect` (empty if the fabric is clean).
    """
    # TODO: results = model.predict(image, conf=confidence_threshold)
    # TODO: Convert result boxes to Defect objects with class names.
    raise NotImplementedError("detect_defects is not implemented yet")


def compute_dhu(total_defects: int, units_inspected: int) -> float:
    """Compute DHU (Defects per Hundred Units).

    Args:
        total_defects: Number of defects found.
        units_inspected: Number of units (garments or fabric panels) checked.

    Returns:
        Defects per 100 units.

    Raises:
        ValueError: If ``units_inspected`` is not positive or defects is negative.
    """
    if units_inspected <= 0:
        raise ValueError("units_inspected must be greater than 0")
    if total_defects < 0:
        raise ValueError("total_defects cannot be negative")
    return total_defects / units_inspected * 100
