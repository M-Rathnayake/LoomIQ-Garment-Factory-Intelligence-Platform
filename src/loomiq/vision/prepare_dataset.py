"""Organize and augment fabric defect images for training.

Two supported datasets (see docs/DATASETS.md):
    * AITEX: large fabric images with pixel masks. Masks are converted into
      YOLO bounding boxes, and the wide images are tiled into squares.
    * TILDA: small labelled texture patches, used for a classification
      approach (pretrained ResNet).

Output layout (YOLO format):
    data/processed/fabric/
        images/{train,val,test}/*.png
        labels/{train,val,test}/*.txt
        data.yaml
"""

from pathlib import Path

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


def tile_image(image_path: Path, tile_size: int, output_dir: Path) -> list[Path]:
    """Cut a wide fabric image into square tiles.

    Returns:
        Paths of the saved tiles.
    """
    # TODO: AITEX images are 4096x256; slide a window of tile_size with some overlap.
    raise NotImplementedError("tile_image is not implemented yet")


def mask_to_yolo_boxes(mask_path: Path, class_id: int) -> list[tuple[int, float, float, float, float]]:
    """Convert a binary defect mask into YOLO boxes.

    Returns:
        List of ``(class_id, x_center, y_center, width, height)``, normalized 0-1.
    """
    # TODO: Find connected components in the mask (e.g. with OpenCV or numpy),
    #       take each bounding rectangle, normalize by image size.
    raise NotImplementedError("mask_to_yolo_boxes is not implemented yet")


def split_dataset(
    source_dir: Path, output_dir: Path, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> None:
    """Split images (and labels) into train / val / test folders."""
    # TODO: Shuffle with seed, copy files into images/<split> and labels/<split>.
    # TODO: Keep defect-free images too, so the model learns "no defect".
    raise NotImplementedError("split_dataset is not implemented yet")


def write_data_yaml(output_dir: Path, class_names: list[str]) -> Path:
    """Write the ``data.yaml`` file that Ultralytics YOLO expects."""
    # TODO: Write path, train, val, test, and names keys.
    raise NotImplementedError("write_data_yaml is not implemented yet")


def prepare_fabric_dataset() -> Path:
    """Run the full preparation pipeline and return the path to ``data.yaml``."""
    # TODO: Read paths and classes from config, then tile -> label -> split -> yaml.
    # TODO: Optional offline augmentation (flips, brightness) for rare defect classes.
    raise NotImplementedError("prepare_fabric_dataset is not implemented yet")
