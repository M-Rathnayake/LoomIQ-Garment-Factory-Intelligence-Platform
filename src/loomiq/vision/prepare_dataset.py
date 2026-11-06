"""Turn the AITEX fabric images into a YOLO training dataset.

AITEX layout (see docs/DATASETS.md):
    data/images/aitex/
        Defect_images/     e.g. 0001_002_00.png   (4096 x 256, grayscale)
        Mask_images/       e.g. 0001_002_00_mask.png  (white = defect pixels)
        NODefect_images/   defect-free images, in sub-folders per fabric

Steps:
    1. Find every image and its mask(s).
    2. **Split by source image** (train / val / test), so tiles from one photo
       never appear in two splits. Otherwise test scores would be inflated.
    3. Cut each image into square tiles and turn mask blobs into YOLO boxes.
    4. Keep every defect tile, and a limited number of defect-free tiles
       (they would otherwise outnumber defects about 30 to 1).
    5. Write ``data.yaml`` for Ultralytics, plus a manifest of the source images.

Output (``vision.prepared_dir``, default data/processed/fabric/):
    images/{train,val,test}/*.png
    labels/{train,val,test}/*.txt   (one "class x_center y_center width height" line per box)
    data.yaml
    manifest.csv                    (source image -> split, number of true defects)

Run from the command line:
    python -m loomiq.vision.prepare_dataset
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


@dataclass
class SourceImage:
    """One original fabric image and its defect masks (none if defect-free)."""

    path: Path
    masks: list[Path] = field(default_factory=list)

    @property
    def has_defect(self) -> bool:
        return bool(self.masks)


def get_settings() -> dict[str, Any]:
    """The ``vision`` config section, with folders as absolute paths."""
    from loomiq.config import PROJECT_ROOT, load_config

    settings = dict(load_config()["vision"])
    settings["raw_dir"] = PROJECT_ROOT / settings["raw_dir"]
    settings["prepared_dir"] = PROJECT_ROOT / settings["prepared_dir"]
    return settings


def _images_in(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def find_aitex_images(raw_dir: Path) -> list[SourceImage]:
    """Find AITEX images and pair each defect image with its mask file(s).

    Masks are matched by name: ``0001_002_00_mask.png`` (and ``..._mask1.png``,
    ``..._mask2.png`` when one image has several masks) belong to ``0001_002_00.png``.

    Raises:
        FileNotFoundError: If the expected AITEX folders are missing.
    """
    defect_dir, mask_dir, normal_dir = raw_dir / "Defect_images", raw_dir / "Mask_images", raw_dir / "NODefect_images"
    if not defect_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"AITEX not found in {raw_dir}. Expected Defect_images/ and Mask_images/ (see docs/DATASETS.md)."
        )

    masks_by_stem: dict[str, list[Path]] = {}
    for mask in _images_in(mask_dir):
        stem = re.sub(r"_mask\d*$", "", mask.stem)
        masks_by_stem.setdefault(stem, []).append(mask)

    sources = []
    for image in _images_in(defect_dir):
        masks = masks_by_stem.get(image.stem, [])
        if not masks:
            logger.warning("No mask for %s; skipped", image.name)
            continue
        sources.append(SourceImage(image, masks))
    if normal_dir.exists():
        sources += [SourceImage(image) for image in _images_in(normal_dir)]

    logger.info("Found %s defect images and %s defect-free images",
                sum(s.has_defect for s in sources), sum(not s.has_defect for s in sources))
    return sources


def load_gray(path: Path) -> np.ndarray:
    """Load an image as a 2-D uint8 array."""
    from PIL import Image

    return np.asarray(Image.open(path).convert("L"))


def combined_mask(source: SourceImage, shape: tuple[int, int]) -> np.ndarray:
    """All of an image's masks merged into one boolean array (all False if no defect)."""
    mask = np.zeros(shape, dtype=bool)
    for mask_path in source.masks:
        mask |= load_gray(mask_path) > 127
    return mask


def mask_to_boxes(mask: np.ndarray, min_area: int = 20) -> list[tuple[int, int, int, int]]:
    """Turn defect blobs in a boolean mask into pixel boxes ``(x1, y1, x2, y2)``.

    Each connected blob becomes one box; specks smaller than ``min_area`` pixels are ignored.
    """
    from scipy import ndimage

    labels, count = ndimage.label(mask)
    boxes = []
    for index, region in enumerate(ndimage.find_objects(labels), start=1):
        if region is None or (labels[region] == index).sum() < min_area:
            continue
        rows, cols = region
        boxes.append((cols.start, rows.start, cols.stop, rows.stop))
    return boxes


def expand_box(box: tuple[int, int, int, int], min_size: int, width: int, height: int) -> tuple[int, int, int, int]:
    """Grow a box around its centre to at least ``min_size`` pixels per side, staying inside the image."""
    x1, y1, x2, y2 = box
    if x2 - x1 < min_size:
        x1 = max(0, min((x1 + x2 - min_size) // 2, width - min_size))
        x2 = min(width, x1 + min_size)
    if y2 - y1 < min_size:
        y1 = max(0, min((y1 + y2 - min_size) // 2, height - min_size))
        y2 = min(height, y1 + min_size)
    return x1, y1, x2, y2


def count_true_defects(mask: np.ndarray, min_area: int) -> int:
    """Separate defects in a full mask, merging blobs that touch (as the detector does)."""
    from loomiq.vision.detect import Defect, merge_boxes

    return len(merge_boxes([Defect("defect", 1.0, box) for box in mask_to_boxes(mask, min_area)]))


def to_yolo_line(box: tuple[int, int, int, int], width: int, height: int, class_id: int = 0) -> str:
    """Pixel box -> YOLO label line (centre x, centre y, width, height, all 0 to 1)."""
    x1, y1, x2, y2 = box
    return (
        f"{class_id} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"
    )


def tile_positions(length: int, tile: int, stride: int) -> list[int]:
    """Start positions that cover ``length`` with tiles, the last one flush with the end."""
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] + tile < length:
        positions.append(length - tile)
    return positions


def make_tiles(image: np.ndarray, tile: int, stride: int) -> list[tuple[int, int, np.ndarray]]:
    """Cut an image into ``(x, y, tile_array)`` squares (padded if the image is smaller)."""
    height, width = image.shape[:2]
    if height < tile or width < tile:
        padded = np.zeros((max(height, tile), max(width, tile)), dtype=image.dtype)
        padded[:height, :width] = image
        image, height, width = padded, *padded.shape[:2]
    return [
        (x, y, image[y:y + tile, x:x + tile])
        for y in tile_positions(height, tile, stride)
        for x in tile_positions(width, tile, stride)
    ]


def split_sources(
    sources: list[SourceImage], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> dict[str, list[SourceImage]]:
    """Split source images into train / val / test, keeping the defect share similar in each."""
    rng = np.random.default_rng(seed)
    splits: dict[str, list[SourceImage]] = {name: [] for name in SPLITS}
    for group in ([s for s in sources if s.has_defect], [s for s in sources if not s.has_defect]):
        order = rng.permutation(len(group))
        n_train = round(len(group) * ratios[0])
        n_val = round(len(group) * ratios[1])
        for position, index in enumerate(order):
            name = "train" if position < n_train else "val" if position < n_train + n_val else "test"
            splits[name].append(group[index])
    return splits


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """Write the ``data.yaml`` file that Ultralytics YOLO expects."""
    import yaml

    path = output_dir / "data.yaml"
    content = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(class_names)),
    }
    path.write_text(yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return path


def prepare_fabric_dataset(
    raw_dir: Path | None = None, output_dir: Path | None = None, seed: int = 42, **overrides: Any
) -> Path:
    """Run the full preparation and return the path to ``data.yaml``.

    Args:
        raw_dir: AITEX folder (defaults to ``vision.raw_dir``).
        output_dir: Where to write the dataset (defaults to ``vision.prepared_dir``).
        seed: Random seed for the split and background sampling.
        **overrides: Any ``vision`` setting, e.g. ``tile_size=128``.
    """
    from PIL import Image

    settings = {**get_settings(), **overrides}
    raw_dir = Path(raw_dir or settings["raw_dir"])
    output_dir = Path(output_dir or settings["prepared_dir"])
    tile, stride, min_area = settings["tile_size"], settings["tile_stride"], settings["min_defect_area"]
    rng = np.random.default_rng(seed)

    splits = split_sources(find_aitex_images(raw_dir), tuple(settings["split"]), seed)
    manifest = []
    for split, sources in splits.items():
        image_dir, label_dir = output_dir / "images" / split, output_dir / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        defect_tiles, background_tiles = [], []
        for source in sources:
            image = load_gray(source.path)
            mask = combined_mask(source, image.shape)
            if source.has_defect and not mask_to_boxes(mask, min_area):
                logger.warning("Mask of %s has no usable defect pixels; skipped", source.path.name)
                continue
            image_tiles = make_tiles(image, tile, stride)
            mask_tiles = make_tiles(mask.astype(np.uint8), tile, stride)
            for (x, y, pixels), (_, _, mask_tile) in zip(image_tiles, mask_tiles):
                boxes = mask_to_boxes(mask_tile.astype(bool), min_area)
                name = f"{source.path.stem}_x{x}_y{y}"
                (defect_tiles if boxes else background_tiles).append((name, pixels, boxes))
            manifest.append({
                "source": source.path.name,
                "path": str(source.path.resolve()),
                "split": split,
                "has_defect": source.has_defect,
                "true_defects": count_true_defects(mask, min_area),
            })

        # Keep all defect tiles, and a limited sample of defect-free ones
        n_background = min(len(background_tiles), max(1, round(len(defect_tiles) * settings["background_ratio"])))
        keep = rng.choice(len(background_tiles), n_background, replace=False) if background_tiles else []
        for name, pixels, boxes in defect_tiles + [background_tiles[i] for i in keep]:
            Image.fromarray(pixels).save(image_dir / f"{name}.png")
            lines = [to_yolo_line(expand_box(box, settings["min_box_size"], tile, tile), tile, tile) for box in boxes]
            (label_dir / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        logger.info("%-5s: %s source images -> %s defect tiles + %s defect-free tiles",
                    split, len(sources), len(defect_tiles), n_background)

    pd.DataFrame(manifest).to_csv(output_dir / "manifest.csv", index=False)
    return write_data_yaml(output_dir, settings["classes"])


if __name__ == "__main__":
    prepare_fabric_dataset()
