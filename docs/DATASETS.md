# Datasets

LoomIQ uses two **real** public datasets for training and one **simulated** data stream for the live dashboard. None of the data is stored in this repository; download it yourself using the steps below.

| Dataset | Real or simulated | Used by | Place it in |
|---|---|---|---|
| UCI Productivity Prediction of Garment Employees | Real | Predict + Recommend | `data/raw/` |
| AITEX Fabric Image Database | Real | Detect (YOLO approach) | `data/images/aitex/` |
| TILDA Textile Texture Database | Real | Detect (ResNet approach) | `data/images/tilda/` |
| Factory simulator stream | Simulated | Live Dashboard | `data/interim/live_stream.csv` (auto-created) |

> Dataset links sometimes move. If a link is broken, search the dataset name on the UCI repository, Kaggle or Papers with Code. Always check each dataset's licence before using it in public work.

---

## 1. UCI: Productivity Prediction of Garment Employees

- **Page:** https://archive.ics.uci.edu/dataset/597/productivity+prediction+of+garment+employees
- **What it is:** 1,197 daily records of sewing and finishing teams from a garment factory in Bangladesh (Jan to Mar 2015).
- **Licence:** CC BY 4.0.

**Steps**
1. Open the page above and click **Download**.
2. Unzip it and copy `garments_worker_productivity.csv` to:
   ```
   data/raw/garments_worker_productivity.csv
   ```
3. The file name must match `data.productivity_file` in `configs/config.yaml`.

**Check it loaded**
```bash
python -c "from loomiq.data.load import load_raw_productivity; print(load_raw_productivity().shape)"
# expected: (1197, 15)
```

**Known data issues** (handled in `src/loomiq/data/clean.py`): `department` typo `"sweing"` and trailing spaces, missing `wip` for finishing rows, `no_of_workers` stored as float, a rare `Quarter5` value.

---

## 2. AITEX Fabric Image Database

- **Page:** https://www.aitex.es/afid/
- **What it is:** 245 grayscale fabric images (4096 x 256 px) from 7 fabric types: 140 defect-free and 105 with 12 defect types. Each defect image comes with a pixel **mask**.
- **Use in LoomIQ:** masks are converted into bounding boxes and the wide images are tiled into squares for YOLOv8 (see `src/loomiq/vision/prepare_dataset.py`).

**Steps**
1. Download the archive from the page above.
2. Extract it so the folders look like this:
   ```
   data/images/aitex/
   ├── Defect_images/
   ├── Mask_images/
   └── NODefect_images/
   ```

---

## 3. TILDA Textile Texture Database

- **Page:** https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html
  (also mirrored on Kaggle; search "TILDA textile")
- **What it is:** textile texture images in several classes: good fabric plus defect classes such as holes, stains, threads and knots.
- **Use in LoomIQ:** a classification approach (pretrained ResNet) when you want "defect / no defect" or defect type per image patch, rather than boxes.

**Steps**
1. Download the dataset.
2. Arrange one folder per class (the layout `torchvision.datasets.ImageFolder` expects):
   ```
   data/images/tilda/
   ├── good/
   ├── hole/
   ├── stain/
   └── thread/
   ```
3. Set `vision.approach: resnet` in `configs/config.yaml` to train on TILDA.

---

## 4. Simulated factory stream

The UCI data is daily, but the dashboard needs **hourly, live** data from several lines. `src/loomiq/simulation/factory_simulator.py` generates this, using value ranges learned from the cleaned UCI data.

```bash
make simulate
# or
python scripts/run_simulator.py --lines 6 --hours 8
```

**This data is simulated.** It shows how the system would work in a real factory; it is not real factory output. All reported model metrics come from the real datasets above.
