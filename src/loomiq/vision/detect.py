"""Find fabric defects in an image and compute DHU.

Images of any size are handled the same way as in training: wide fabric strips
are cut into square tiles, each tile is checked, and the boxes are mapped back
onto the full image. Boxes that touch across a tile border are merged, so one
defect is counted once.

DHU (Defects per Hundred Units) = total defects / units inspected x 100.
A lower DHU means better quality.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Defect:
    """One defect found in an image."""

    label: str
    confidence: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels of the full image


def get_model_path() -> Path:
    """Where the trained defect model is saved (``models/defect_model.pt``)."""
    from loomiq.config import get_path, load_config

    return get_path("models") / load_config()["vision"]["model_file"]


def load_defect_model(path: Path | None = None) -> Any:
    """Load the trained YOLO defect model.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    from ultralytics import YOLO

    path = Path(path or get_model_path())
    if not path.exists():
        raise FileNotFoundError(f"No defect model at {path}. Train one first: python -m loomiq.vision.train_defect")
    return YOLO(str(path))


def to_gray_array(image: Any) -> np.ndarray:
    """Accept a file path, an uploaded file, a PIL image or an array; return a 2-D uint8 array."""
    from PIL import Image

    if isinstance(image, np.ndarray):
        array = image
    else:
        array = np.asarray(Image.open(image) if not isinstance(image, Image.Image) else image)
    if array.ndim == 3:  # colour -> gray, the same as the AITEX training images
        array = np.asarray(Image.fromarray(array[..., :3].astype(np.uint8)).convert("L"))
    return array.astype(np.uint8)


def merge_boxes(defects: list[Defect], gap: float = 4.0) -> list[Defect]:
    """Merge boxes that overlap or touch (within ``gap`` pixels) into one defect."""
    merged = sorted(defects, key=lambda d: d.box[0])
    changed = True
    while changed:
        changed = False
        result: list[Defect] = []
        for defect in merged:
            for i, other in enumerate(result):
                a, b = defect.box, other.box
                if a[0] <= b[2] + gap and b[0] <= a[2] + gap and a[1] <= b[3] + gap and b[1] <= a[3] + gap:
                    result[i] = Defect(
                        other.label,
                        max(other.confidence, defect.confidence),
                        (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])),
                    )
                    changed = True
                    break
            else:
                result.append(defect)
        merged = result
    return merged


def detect_defects(model: Any, image: Any, confidence_threshold: float | None = None) -> list[Defect]:
    """Find defects in one image of any size.

    Args:
        model: Model from :func:`load_defect_model` (anything with an Ultralytics-style ``predict``).
        image: File path, uploaded file, PIL image or numpy array.
        confidence_threshold: Ignore detections below this score (defaults to config).

    Returns:
        A list of :class:`Defect` in full-image pixel coordinates (empty if the fabric is clean).
    """
    from loomiq.vision.prepare_dataset import get_settings, make_tiles

    settings = get_settings()
    confidence = settings["confidence_threshold"] if confidence_threshold is None else confidence_threshold
    pixels = to_gray_array(image)
    tiles = make_tiles(pixels, settings["tile_size"], settings["tile_stride"])

    # YOLO expects 3-channel images
    batch = [np.repeat(tile[..., None], 3, axis=2) for _, _, tile in tiles]
    results = model.predict(batch, conf=confidence, imgsz=settings["tile_size"], verbose=False)

    found = []
    for (x, y, _), result in zip(tiles, results):
        names = result.names
        for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            x1, y1, x2, y2 = box
            # Clip to the real image (tiles may be padded)
            full = (
                min(x1 + x, pixels.shape[1]), min(y1 + y, pixels.shape[0]),
                min(x2 + x, pixels.shape[1]), min(y2 + y, pixels.shape[0]),
            )
            found.append(Defect(names[int(cls)], float(score), full))
    return merge_boxes(found)


def draw_defects(image: Any, defects: list[Defect], padding: int = 10) -> Any:
    """Return a colour PIL image with a red box and score around each defect.

    Boxes are padded and drawn thick, so tiny defects stay visible when a wide
    fabric strip is shrunk to fit the screen.
    """
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(to_gray_array(image)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width = max(3, canvas.width // 400)
    for defect in defects:
        x1, y1, x2, y2 = defect.box
        box = (max(0, x1 - padding), max(0, y1 - padding), min(canvas.width - 1, x2 + padding), min(canvas.height - 1, y2 + padding))
        draw.rectangle(box, outline=(208, 59, 59), width=width)
        draw.text((box[0] + 4, max(0, box[1] - 14)), f"{defect.confidence:.2f}", fill=(208, 59, 59))
    return canvas


def defect_closeups(image: Any, defects: list[Defect], size: int = 96, zoom: int = 3) -> list[Any]:
    """Enlarged square crops centred on each defect (for a close-up view)."""
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(to_gray_array(image)).convert("RGB")
    closeups = []
    for defect in defects:
        x1, y1, x2, y2 = defect.box
        half = max(size // 2, int(max(x2 - x1, y2 - y1) / 2) + 8)
        left = int(min(max(0, (x1 + x2) / 2 - half), max(0, canvas.width - 2 * half)))
        top = int(min(max(0, (y1 + y2) / 2 - half), max(0, canvas.height - 2 * half)))
        crop = canvas.crop((left, top, left + 2 * half, top + 2 * half))
        crop = crop.resize((2 * half * zoom, 2 * half * zoom), Image.NEAREST)
        # Thin outline drawn after zooming, a few pixels outside the defect so it stays visible
        pad = 3 * zoom
        ImageDraw.Draw(crop).rectangle(
            ((x1 - left) * zoom - pad, (y1 - top) * zoom - pad, (x2 - left) * zoom + pad, (y2 - top) * zoom + pad),
            outline=(208, 59, 59), width=2,
        )
        closeups.append(crop)
    return closeups


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
