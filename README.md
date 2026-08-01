# SPECTRA — Deepfake Detection Website

A website that runs the exact model architecture from `Deep_Fake_detection_final.ipynb`
(dual-stream EfficientNet-B4 + FFT frequency branch) against **your own dataset**, with
a Flask backend for training/inference and a browser UI for uploading images or videos.

```
deep_fake/
├── model.py         # the network architecture (shared by train.py and app.py)
├── train.py         # trains on your dataset, saves checkpoints/best_model.pt
├── app.py           # Flask server: serves the site + prediction API
├── index.html       # the website (upload UI, results, spectrum readout)
├── style.css        # external stylesheet
├── script.js        # external JavaScript
└── requirements.txt
```

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
| `--epochs` | 10 | training epochs |
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
 * Running on http://127.0.0.1:5000
```

Open **http://127.0.0.1:5000** in your browser.

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

By default the server only listens on `127.0.0.1` (your machine only). To make it
reachable on your local network:

```bash
python app.py --checkpoint checkpoints/best_model.pt --host 0.0.0.0 --port 5000
```

For a real public deployment, put this behind a production WSGI server (e.g. `gunicorn`)
and a reverse proxy (e.g. nginx), and don't run Flask's built-in dev server directly —
happy to help set that up if/when you get there.

---

## Troubleshooting

- **"Expected folder not found" during training** — double-check `--data_root` points
  to the folder that *contains* `train/` and `val/`, not to `train/` itself.
- **CUDA out of memory** — lower `--batch_size` (try 16 or 8).
- **`model_loaded: false` on `/api/health`** — the checkpoint path passed to `app.py`
  is wrong or the file is corrupted; re-check the `--checkpoint` argument.
- **Video upload fails / "Could not read any frames"** — make sure `opencv-python-headless`
  installed correctly and the video isn't corrupted; try a standard MP4 (H.264) first.
- **Training accuracy stays near 50%** — check your `real`/`fake` folders aren't
  accidentally swapped, and that both classes have a reasonable number of samples.
