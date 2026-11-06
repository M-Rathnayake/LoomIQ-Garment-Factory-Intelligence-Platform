# Datasets

LoomIQ uses two **real** public datasets for training and one **simulated** data stream for the live dashboard. None of the data is stored in this repository; download it yourself using the steps below.

| Dataset | Real or simulated | Used by | Place it in |
|---|---|---|---|
| UCI Productivity Prediction of Garment Employees | Real | Predict + Recommend | `data/raw/` |
| AITEX Fabric Image Database | Real | Detect | `data/images/aitex/` |
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

- **Licence:** no open licence is stated. The authors ask that research using the images cites: J. Silvestre-Blanes, T. Albero-Albero, I. Miralles, R. Pérez-Llorens, J. Moreno, *"AFID: a public fabric image database for defect detection"*, AUTEX Research Journal, No. 4, 2019.
- **Known quirks** (handled by the code): one defect image (`0100_025_08`) has no mask and one mask (`0106_010_03`) is empty, so both are skipped; one image is 3796 px wide instead of 4096.

**Steps**
1. Download the three archives linked on the page (the links say `.zip`, but the files are **7z**):
   - `https://www.aitex.es/wp-content/uploads/2019/07/Defect_images.7z` (about 78 MB)
   - `https://www.aitex.es/wp-content/uploads/2019/07/Mask_images.7z` (under 1 MB)
   - `https://www.aitex.es/wp-content/uploads/2019/07/NODefect_images.7z` (about 98 MB)
2. Extract them (e.g. with 7-Zip, or `pip install py7zr`) so the folders look like this:
   ```
   data/images/aitex/
   ├── Defect_images/      e.g. 0001_002_00.png
   ├── Mask_images/        e.g. 0001_002_00_mask.png (white = defect pixels)
   └── NODefect_images/    defect-free images, in sub-folders per fabric
   ```
3. Prepare the tiles and train the detector:
   ```bash
   python -m loomiq.vision.train_defect
   ```
   This also downloads the pretrained YOLOv8 nano weights (about 6 MB) on first use.

> **Why one "defect" class?** AITEX labels 12 defect types, but with about 105 defect images most types have only a handful of examples, too few to learn each type reliably, and they don't map cleanly to "hole / stain / broken thread". DHU only needs a defect count, so LoomIQ detects a single "defect" class.

> **Alternative dataset:** the TILDA textile texture database (classification patches: good, hole, stain, thread...) could be used for a defect-type classifier. It is not used by the current code.

---

## 3. Simulated factory stream

The UCI data is daily, but the dashboard needs **hourly, live** data from several lines. `src/loomiq/simulation/factory_simulator.py` **replays real days from the UCI test weeks** (real teams, SMV, workers and targets, never seen by the model during training), revealing output hour by hour with random noise, machine breakdowns and style changes. The dashboard runs its own simulator; the script below only saves one day to CSV.

```bash
make simulate
# or
python scripts/run_simulator.py --lines 12 --seed 3
```

**This data is simulated.** It shows how the system would work in a real factory; it is not real factory output. All reported model metrics come from the real datasets above.
