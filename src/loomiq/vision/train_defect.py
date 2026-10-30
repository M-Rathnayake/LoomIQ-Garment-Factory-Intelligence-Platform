"""Train a fabric defect model with transfer learning.

Two options, selected by ``vision.approach`` in config.yaml:
    * ``yolo``: fine-tune a pretrained YOLOv8 detector (Ultralytics) on AITEX.
    * ``resnet``: fine-tune a pretrained ResNet classifier (torchvision) on TILDA.

``ultralytics`` and ``torch`` are imported inside functions so
``import loomiq`` stays fast.
"""

from pathlib import Path
from typing import Any

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


def train_yolo(data_yaml: Path, config: dict[str, Any]) -> Path:
    """Fine-tune YOLOv8 on the prepared fabric dataset.

    Returns:
        Path to the best weights file.
    """
    # TODO: model = ultralytics.YOLO(config["vision"]["yolo_weights"])
    # TODO: model.train(data=data_yaml, epochs=..., imgsz=..., batch=..., seed=...)
    # TODO: Copy runs/detect/train/weights/best.pt to models/<model_file>.
    raise NotImplementedError("train_yolo is not implemented yet")


def train_resnet(data_dir: Path, config: dict[str, Any]) -> Path:
    """Fine-tune a pretrained ResNet as a defect / no-defect classifier.

    Returns:
        Path to the saved model weights.
    """
    # TODO: torchvision.models.resnet18(weights="DEFAULT"); replace final fc layer.
    # TODO: Freeze early layers first, train head, then unfreeze for fine-tuning.
    # TODO: Use ImageFolder + transforms; track val accuracy; save best state_dict.
    raise NotImplementedError("train_resnet is not implemented yet")


def train_defect_model() -> Path:
    """Train the defect model chosen in config and return the weights path."""
    # TODO: Read config, call prepare_dataset if needed, then train_yolo or train_resnet.
    # TODO: Save precision / recall / mAP to reports/ for the README.
    raise NotImplementedError("train_defect_model is not implemented yet")
