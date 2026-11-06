"""Tests for loomiq.vision: dataset preparation, detection and DHU.

A small synthetic dataset with the same folder layout as AITEX is generated,
so no download is needed. The training smoke test runs only if Ultralytics is installed.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from loomiq.vision import detect, prepare_dataset as prep
from loomiq.vision.detect import Defect, compute_dhu

TILE = 128


def fabric(rng: np.random.Generator, width: int = 512, height: int = 128) -> np.ndarray:
    """Grey woven-looking texture."""
    stripes = (np.sin(np.arange(width) / 2.0) * 20 + 128).astype(np.uint8)
    return np.clip(stripes[None, :] + rng.normal(0, 8, (height, width)), 0, 255).astype(np.uint8)


@pytest.fixture
def aitex_dir(tmp_path: Path) -> Path:
    """AITEX-style folders: 4 defect images with masks, 6 defect-free images."""
    rng = np.random.default_rng(0)
    root = tmp_path / "aitex"
    for folder in ["Defect_images", "Mask_images", "NODefect_images/fabric_a"]:
        (root / folder).mkdir(parents=True)

    for i in range(4):
        image, mask = fabric(rng), np.zeros((128, 512), dtype=np.uint8)
        x = 40 + 100 * i
        image[50:70, x:x + 30] = 20  # a dark stain
        mask[50:70, x:x + 30] = 255
        stem = f"000{i}_002_00"
        Image.fromarray(image).save(root / "Defect_images" / f"{stem}.png")
        if i == 3:  # one image with its defect split over two mask files
            half = mask.copy()
            half[:, x + 15:] = 0
            Image.fromarray(half).save(root / "Mask_images" / f"{stem}_mask1.png")
            Image.fromarray(mask - half).save(root / "Mask_images" / f"{stem}_mask2.png")
        else:
            Image.fromarray(mask).save(root / "Mask_images" / f"{stem}_mask.png")

    for i in range(6):
        Image.fromarray(fabric(rng)).save(root / "NODefect_images" / "fabric_a" / f"clean_{i}.png")
    return root


def test_find_images_pairs_masks(aitex_dir: Path) -> None:
    sources = prep.find_aitex_images(aitex_dir)
    defects = [s for s in sources if s.has_defect]
    assert len(defects) == 4 and len(sources) == 10
    assert len(next(s for s in defects if s.path.stem == "0003_002_00").masks) == 2


def test_missing_dataset_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prep.find_aitex_images(tmp_path / "nothing")


def test_mask_to_boxes_and_yolo_format() -> None:
    mask = np.zeros((100, 200), dtype=bool)
    mask[10:20, 30:60] = True  # real defect
    mask[80, 150] = True  # a 1-pixel speck, ignored
    assert prep.mask_to_boxes(mask, min_area=20) == [(30, 10, 60, 20)]
    assert prep.to_yolo_line((30, 10, 60, 20), 200, 100) == "0 0.225000 0.150000 0.150000 0.100000"


def test_tiles_cover_the_image() -> None:
    assert prep.tile_positions(512, 128, 128) == [0, 128, 256, 384]
    assert prep.tile_positions(500, 128, 128) == [0, 128, 256, 372]  # last tile flush with the end
    tiles = prep.make_tiles(np.ones((100, 300), dtype=np.uint8), 128, 128)
    assert all(t.shape == (128, 128) for _, _, t in tiles)  # small images are padded


def test_split_keeps_each_source_in_one_split(aitex_dir: Path) -> None:
    sources = prep.find_aitex_images(aitex_dir)
    splits = prep.split_sources(sources, (0.5, 0.25, 0.25))
    names = [s.path.name for group in splits.values() for s in group]
    assert sorted(names) == sorted(s.path.name for s in sources)
    assert all(any(s.has_defect for s in group) for group in splits.values())


def test_prepare_dataset_writes_yolo_files(aitex_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "prepared"
    data_yaml = prep.prepare_fabric_dataset(aitex_dir, out, tile_size=TILE, tile_stride=TILE, split=[0.5, 0.25, 0.25])

    config = yaml.safe_load(data_yaml.read_text())
    assert config["names"] == {0: "defect"} and config["train"] == "images/train"

    labels = list((out / "labels").rglob("*.txt"))
    images = list((out / "images").rglob("*.png"))
    assert len(labels) == len(images) > 0
    with_boxes = [p for p in labels if p.read_text().strip()]
    for path in with_boxes:
        cls, xc, yc, w, h = path.read_text().split()[:5]
        assert cls == "0" and all(0 < float(v) <= 1 for v in (xc, yc, w, h))
    # Defect-free tiles are capped at background_ratio (1.0) x defect tiles per split
    assert len(labels) - len(with_boxes) <= len(with_boxes) + 3

    manifest = pd.read_csv(out / "manifest.csv")
    assert len(manifest) == 10 and manifest.groupby("source")["split"].nunique().max() == 1
    assert manifest.loc[manifest["has_defect"], "true_defects"].eq(1).all()


class FakeModel:
    """Stands in for YOLO: reports one box in the tile at x=128, crossing into the next tile."""

    def predict(self, batch, **kwargs):
        results = []
        for i, _ in enumerate(batch):
            boxes = [[100.0, 40.0, 128.0, 60.0]] if i == 0 else [[0.0, 42.0, 20.0, 58.0]] if i == 1 else []
            results.append(SimpleNamespace(
                names={0: "defect"},
                boxes=SimpleNamespace(xyxy=np.array(boxes).reshape(-1, 4), conf=np.full(len(boxes), 0.8), cls=np.zeros(len(boxes))),
            ))
        return results


def test_detect_maps_tiles_back_and_merges(monkeypatch) -> None:
    settings = {**prep.get_settings(), "tile_size": TILE, "tile_stride": TILE}
    monkeypatch.setattr(prep, "get_settings", lambda: settings)
    image = np.full((128, 512), 128, dtype=np.uint8)

    defects = detect.detect_defects(FakeModel(), image)
    # Two halves of one defect, split by the tile border at x=128, count once
    assert len(defects) == 1
    assert defects[0].box == (100.0, 40.0, 148.0, 60.0)
    assert detect.draw_defects(image, defects).size == (512, 128)


def test_merge_boxes_keeps_separate_defects() -> None:
    far_apart = [Defect("defect", 0.9, (0, 0, 10, 10)), Defect("defect", 0.7, (50, 50, 60, 60))]
    assert len(detect.merge_boxes(far_apart)) == 2


def test_to_gray_array_accepts_colour() -> None:
    colour = np.zeros((10, 20, 3), dtype=np.uint8)
    assert detect.to_gray_array(colour).shape == (10, 20)


def test_compute_dhu() -> None:
    assert compute_dhu(total_defects=5, units_inspected=200) == pytest.approx(2.5)


def test_compute_dhu_rejects_zero_units() -> None:
    with pytest.raises(ValueError):
        compute_dhu(total_defects=1, units_inspected=0)


@pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None, reason="ultralytics not installed")
def test_yolo_training_smoke(aitex_dir: Path, tmp_path: Path, monkeypatch) -> None:
    """Train for one epoch from an untrained YOLO config (no download), then detect."""
    from loomiq.config import PROJECT_ROOT
    from loomiq.vision import train_defect

    data_yaml = prep.prepare_fabric_dataset(aitex_dir, tmp_path / "prepared", tile_size=TILE, tile_stride=TILE)
    settings = {**prep.get_settings(), "tile_size": TILE, "epochs": 1, "batch_size": 4, "patience": 1}
    monkeypatch.setattr(prep, "get_settings", lambda: settings)
    monkeypatch.chdir(tmp_path)

    best = train_defect.train_yolo(
        data_yaml, settings, weights="yolov8n.yaml", plots=False, workers=0,
        project=str(tmp_path / "runs"), device="cpu", amp=False,
    )
    assert best.exists() and not (PROJECT_ROOT / "runs" / "defect" / "smoke").exists()

    model = detect.load_defect_model(best)
    manifest = pd.read_csv(tmp_path / "prepared" / "manifest.csv")
    panels = train_defect.evaluate_panels(model, manifest)
    assert panels["test_panels"] > 0 and 0 <= panels["panel_accuracy"] <= 1


def test_defect_closeups_are_square_and_zoomed() -> None:
    image = np.full((256, 1024), 128, dtype=np.uint8)
    defects = [Defect("defect", 0.9, (500.0, 120.0, 506.0, 126.0)), Defect("defect", 0.5, (1015.0, 0.0, 1024.0, 8.0))]
    crops = detect.defect_closeups(image, defects, size=96, zoom=3)
    assert [c.size for c in crops] == [(288, 288), (288, 288)]  # edge defects are shifted inside the image
