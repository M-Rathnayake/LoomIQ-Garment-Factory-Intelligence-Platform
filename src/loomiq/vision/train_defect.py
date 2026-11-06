"""Train and evaluate the fabric defect detector (YOLOv8, transfer learning).

Transfer learning: start from YOLOv8 nano weights already trained on everyday
photos (COCO), and fine-tune them on fabric tiles. This needs far fewer images
than training from scratch, which matters with only ~105 AITEX defect images.

Evaluation on the **test images** (never used in training), at two levels:
    * **Tiles** (YOLO's own metrics): precision, recall, mAP@50.
    * **Panels** (what a factory cares about), on the full original images:
        - defective panels caught (recall) and clean panels wrongly flagged,
        - DHU estimate error: predicted vs true defects per 100 panels.

Run from the command line (after downloading AITEX, see docs/DATASETS.md):
    python -m loomiq.vision.train_defect
"""

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from loomiq.utils.logger import get_logger

logger = get_logger(__name__)


def train_yolo(data_yaml: Path, settings: dict[str, Any], weights: str | None = None, **overrides: Any) -> Path:
    """Fine-tune YOLOv8 on the prepared fabric tiles.

    Args:
        data_yaml: Output of :func:`loomiq.vision.prepare_dataset.prepare_fabric_dataset`.
        settings: The ``vision`` settings.
        weights: Starting weights; defaults to the pretrained ``yolov8n.pt``.
        **overrides: Extra Ultralytics training arguments (e.g. ``epochs=1``).

    Returns:
        Path to the best weights from this run.
    """
    from ultralytics import YOLO

    from loomiq.config import PROJECT_ROOT, load_config

    model = YOLO(weights or settings["yolo_weights"])
    arguments = {
        "data": str(data_yaml),
        "epochs": settings["epochs"],
        "batch": settings["batch_size"],
        "imgsz": settings["tile_size"],
        "patience": settings["patience"],
        "seed": load_config()["project"]["random_seed"],
        "project": str(PROJECT_ROOT / "runs" / "defect"),
        "name": "train",
        "exist_ok": True,
        "verbose": False,
        **overrides,
    }
    model.train(**arguments)
    return Path(model.trainer.best)


def evaluate_tiles(weights: Path, data_yaml: Path, settings: dict[str, Any]) -> dict[str, float]:
    """YOLO's precision, recall and mAP on the test tiles."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(
        data=str(data_yaml), split="test", imgsz=settings["tile_size"], verbose=False, plots=False
    )
    return {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }


def evaluate_panels(model: Any, manifest: pd.DataFrame) -> dict[str, float]:
    """Check the detector on whole test images, the way the scanner page uses it.

    Args:
        model: Loaded defect model.
        manifest: ``manifest.csv`` from dataset preparation.

    Returns:
        Panel-level metrics and the DHU estimate error.
    """
    from loomiq.vision.detect import compute_dhu, detect_defects

    test = manifest[manifest["split"] == "test"].copy()
    test["predicted_defects"] = [len(detect_defects(model, path)) for path in test["path"]]
    flagged = test["predicted_defects"] > 0
    defective = test["has_defect"].astype(bool)

    true_dhu = compute_dhu(int(test["true_defects"].sum()), len(test))
    predicted_dhu = compute_dhu(int(test["predicted_defects"].sum()), len(test))
    return {
        "test_panels": int(len(test)),
        "defective_panels": int(defective.sum()),
        "defective_panels_caught": float((flagged & defective).sum() / max(defective.sum(), 1)),
        "clean_panels_flagged": float((flagged & ~defective).sum() / max((~defective).sum(), 1)),
        "panel_accuracy": float((flagged == defective).mean()),
        "true_dhu": true_dhu,
        "predicted_dhu": predicted_dhu,
        "dhu_error": predicted_dhu - true_dhu,
    }


def train_defect_model(prepare: bool = True, **overrides: Any) -> dict[str, Any]:
    """Full pipeline: prepare tiles -> fine-tune YOLO -> evaluate -> save model and metrics.

    Args:
        prepare: Rebuild the tile dataset first (set False to reuse an existing one).
        **overrides: Extra Ultralytics training arguments (e.g. ``epochs=10``).

    Returns:
        The metrics that were saved to ``reports/defect_metrics.json``.
    """
    from loomiq.config import get_path
    from loomiq.vision.detect import get_model_path, load_defect_model
    from loomiq.vision.prepare_dataset import get_settings, prepare_fabric_dataset

    settings = get_settings()
    data_yaml = prepare_fabric_dataset() if prepare else settings["prepared_dir"] / "data.yaml"

    best = train_yolo(data_yaml, settings, **overrides)
    model_path = get_model_path()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, model_path)
    logger.info("Saved defect model to %s", model_path)

    manifest = pd.read_csv(settings["prepared_dir"] / "manifest.csv")
    metrics = {
        "tiles": evaluate_tiles(model_path, data_yaml, settings),
        "panels": evaluate_panels(load_defect_model(model_path), manifest),
    }
    metrics_path = get_path("reports") / settings["metrics_file"]
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Defect metrics: %s", metrics)
    return metrics


if __name__ == "__main__":
    train_defect_model()
