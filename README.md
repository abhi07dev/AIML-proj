# SPECTRA — Deepfake Detection Website

A website that runs the exact model architecture from `Deep_Fake_detection_final.ipynb`
(dual-stream EfficientNet-B4 + FFT frequency branch) against **your own dataset**, with
a Flask backend for training/inference and a browser UI for uploading images or videos.

```
deep_fake/
├── model.py         # the network architecture (shared by train.py and app.py)
├── train.py         # trains on your dataset, saves checkpoints/best_model.pt
├── evaluate.py      # scores a checkpoint on a held-out test/ split
├── app.py           # Flask server: serves the site + prediction API
├── index.html       # the website (upload UI, results, spectrum readout)
├── style.css        # external stylesheet
├── script.js        # external JavaScript
└── requirements.txt
```

> Note: there are two virtualenvs in this repo. **`venv/` is the CUDA one**
> (torch 2.6.0+cu124, GPU available) — always activate `venv` before running
> anything. `.venv/` is CPU-only and the system `python` has no torch at all.
> The system Python (3.12.10) on PATH has no torch; `python app.py` will only
> work inside an activated venv.

---

## Step 1 — Install prerequisites

You need Python 3.10+ and, ideally, a CUDA GPU (training on CPU works but is slow).

```bash
cd deep_fake
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch instead of the default one
(check your CUDA version and get the right command from https://pytorch.org/get-started/locally/).

---

## Step 2 — Arrange your dataset

You said your dataset is already Kaggle-style. `train.py` expects exactly this layout:

```
your_dataset/
├── train/
│   ├── real/     ← real face images
│   └── fake/     ← fake/manipulated face images
└── val/
    ├── real/
    └── fake/
```

Notes:
- Folder names must be `real` and `fake` (case-insensitive). `original`/`manipulated`
  and `0`/`1` also work, in case your dataset uses those instead.
- If you only have `train/` and no `val/`, split off ~15% of each class into `val/`
  before training — a model evaluated on its own training data will look artificially
  good and won't generalize.
- Supported image formats: `.jpg .jpeg .png .webp .bmp`.
- If your dataset is actually **videos** rather than pre-extracted face images, extract
  frames first (e.g. with OpenCV, sampling every N frames) and sort them into the same
  `real/fake` folders before training — the model is an image classifier under the hood,
  and `app.py`'s video endpoint already handles frame sampling at inference time.

---

## Step 3 — Train the model on your dataset

From the project directory:

```bash
python train.py --data_root /path/to/your_dataset --epochs 10 --batch_size 32
```

Useful flags:
| Flag | Default | Purpose |
|---|---|---|
| `--epochs` | 10 | **total** target epoch; when resuming, training continues from checkpoint epoch + 1 up to this value |
| `--resume` | none | path to `checkpoints/checkpoint_epochNNN.pt` to continue training from (restores model, optimizer, AMP scaler, history, best AUC, early stopping) |
| `--batch_size` | 32 | lower this if you hit out-of-memory errors |
| `--lr` | 1e-4 | learning rate for the classifier head (backbone uses 10% of this) |
| `--warmup_epochs` | 2 | epochs before the backbone unfreezes |
| `--patience` | 5 | early-stopping patience on validation AUC |
| `--max_train` | none | cap training samples — use a small number first for a smoke test |
| `--save_dir` | `checkpoints` | where checkpoints are written |

Run a quick smoke test first to make sure everything loads correctly:

```bash
python train.py --data_root /path/to/your_dataset --epochs 1 --max_train 200
```

Once that finishes without errors, launch the real run. This writes:
- `checkpoints/best_model.pt` — the checkpoint with the best validation AUC (this is what the website uses)
- `checkpoints/checkpoint_epochNNN.pt` — one per epoch
- `checkpoints/history.json` — loss/accuracy/AUC per epoch, if you want to plot it later

Training time depends entirely on dataset size and GPU. As a reference point, a few
thousand images per class on a single mid-range GPU is typically minutes-to-an-hour per epoch.

### Training across multiple sessions (pause and resume)

If one sitting isn't enough, stop after any epoch (Ctrl+C between epochs is fine — a
checkpoint is written after every epoch) and pick up later with `--resume`. `--epochs` is
always the **total** target epoch:

```bash
# Session 1 — trains epochs 1..10
python train.py --data_root /path/to/your_dataset --epochs 10

# Session 2 (later) — continues epochs 11..20
python train.py --data_root /path/to/your_dataset --epochs 20 --resume checkpoints/checkpoint_epoch010.pt

# Session 3 (even later) — continues epochs 21..30
python train.py --data_root /path/to/your_dataset --epochs 30 --resume checkpoints/checkpoint_epoch020.pt
```

Notes:
- Use the **same `--data_root`, `--batch_size`, `--img_size`** every session so the data
  pipeline matches the checkpoint.
- `--lr` and the backbone-freeze schedule are restored from the checkpoint; command-line
  values are ignored for the resumed state.
- Early stopping and `best_model.pt` tracking continue seamlessly across sessions.

---

## Step 3b — Evaluate on a held-out test set

Validation numbers can flatter a model; the real test is scoring images it never saw.
If your dataset has a `test/` split (same `real/` / `fake/` layout, or falls back to
`val/` / `valid/`), run:

```bash
python evaluate.py --data_root /path/to/your_dataset --checkpoint checkpoints/best_model.pt
```

Reports test accuracy, ROC-AUC, F1, per-class precision/recall, plus example and
worst-misclassified predictions so you can eyeball sanity. Useful flags:

| Flag | Default | Purpose |
|---|---|---|
| `--max_samples` | none | evaluate on a subset (quick speed check first) |
| `--batch_size` | 32 | lower if you hit out-of-memory |
| `--examples` | 10 | how many example predictions to print |


---

## Step 4 — Start the website

From the project directory, point the server at your checkpoint:

```bash
python app.py --checkpoint checkpoints/best_model.pt
```

You'll see:
```
Loaded checkpoint: checkpoints/best_model.pt
Checkpoint val metrics: {...}
 * Running on http://0.0.0.0:5000
```

Open **http://127.0.0.1:5000** in your browser. The server binds to
`0.0.0.0` (all interfaces) so it is reachable from other machines and cloud
hosts; it also honors the `PORT` environment variable used by platforms like
Render (run `python app.py --checkpoint checkpoints/best_model.pt` with no
`--host`/`--port` and it will pick up Render's `$PORT` automatically).

---

## Step 5 — Use it

- **Image tab**: drop or browse a JPG/PNG/WEBP. Click *Run analysis*. You'll get a
  verdict (e.g. "LIKELY FAKE"), a confidence score, and a frequency-branch readout
  showing the fake-probability signal.
- **Video tab**: drop an MP4/MOV/AVI. The server samples 24 frames evenly across the
  clip, scores each one, and reports the average fake probability plus how many
  individual frames looked suspicious (≥70% fake).

Other useful endpoints:
- `GET /api/health` — confirms the model is loaded and which device (CPU/GPU) it's running on
- `POST /api/predict-image` — `multipart/form-data` with a `file` field, returns JSON
- `POST /api/predict-video` — same, for video files

---

## Optional — Deploying beyond your own machine

By default the server binds to `0.0.0.0` and honors the `PORT` environment
variable (falling back to 5000), so a plain

```bash
python app.py --checkpoint checkpoints/best_model.pt
```

works unchanged on hosts like Render. To override either value explicitly:

```bash
python app.py --checkpoint checkpoints/best_model.pt --host 0.0.0.0 --port 5000
```

For a high-traffic public deployment, put this behind a production WSGI server
(e.g. `gunicorn`) and a reverse proxy (e.g. nginx), and don't run Flask's
built-in dev server directly — happy to help set that up if/when you get there.

---

## Troubleshooting

- **"Expected folder not found" during training** — double-check `--data_root` points
  to the folder that *contains* `train/` and `val/`, not to `train/` itself.
- **CUDA out of memory** — lower `--batch_size` (try 16 or 8).
- **`model_loaded: false` on `/api/health`** — the checkpoint path passed to `app.py`
  is wrong or the file is corrupted; re-check the `--checkpoint` argument.
- **Render: "No open ports detected" then `Exited with status 137`** — the process was
  killed, almost always out-of-memory on the free/starter 512MB tier. `app.py` already
  caps torch's thread pools, loads the checkpoint with `mmap=True`, and binds to
  `0.0.0.0` + `$PORT`; make sure your Render start command is exactly
  `python app.py --checkpoint checkpoints/best_model.pt`. If it still OOMs, upgrade the
  instance to Standard (2GB) — a 235MB checkpoint + torch on 512MB is a tight squeeze.
- **Video upload fails / "Could not read any frames"** — make sure `opencv-python-headless`
  installed correctly and the video isn't corrupted; try a standard MP4 (H.264) first.
- **Training accuracy stays near 50%** — check your `real`/`fake` folders aren't
  accidentally swapped, and that both classes have a reasonable number of samples.
