"""
Flask backend for the deepfake detection website.

Serves the frontend (index.html in the same directory as app.py) and two JSON endpoints:
    POST /api/predict-image   - single image upload -> verdict
    POST /api/predict-video   - video upload -> per-frame + aggregate verdict

Run:
    python app.py --checkpoint checkpoints/best_model.pt

Model hosting (for Render.com / cloud deployments):
    Render.com does NOT support Git LFS, so the model is downloaded from
    Hugging Face Hub at startup when the checkpoint file is missing or is
    just an LFS pointer.  Set these environment variables in your Render
    service dashboard:
        HF_MODEL_REPO   - e.g.  abhi07dev/deepfake-detector
        HF_MODEL_FILE   - e.g.  best_model.pt  (default)
        HF_TOKEN        - only needed if the repo is private
"""
import argparse
import gc
import io
import os
import tempfile

# Cap BLAS/OpenMP thread pools BEFORE importing torch/cv2: torch defaults to one
# thread per core, which is a huge memory overhead on small RAM instances (e.g.
# Render free tier, 512MB) and provides no benefit for single-request inference.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')

import cv2
import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
from torchvision import transforms

from model import DeepfakeDetector

# ───────────────────────────── Config ──────────────────────────────────────
# Frontend files live in the same directory as app.py
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_UPLOAD_MB = 200
LABELS = {0: 'REAL', 1: 'FAKE'}

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None  # loaded in main()

val_tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _is_lfs_pointer(path: str) -> bool:
    """Return True if the file is a Git LFS pointer (tiny text stub, not the real binary)."""
    try:
        if os.path.getsize(path) > 1024:          # real .pt files are >> 1 KB
            return False
        with open(path, 'rb') as f:
            header = f.read(12)
        return header == b'version http'           # LFS pointer signature
    except OSError:
        return False


def _download_from_hf(checkpoint_path: str) -> None:
    """Download the model from Hugging Face Hub into checkpoint_path."""
    repo_id = os.environ.get('HF_MODEL_REPO', '').strip()
    if not repo_id:
        raise RuntimeError(
            "Model checkpoint is an LFS pointer or missing, and HF_MODEL_REPO "
            "environment variable is not set.  Please upload best_model.pt to a "
            "Hugging Face model repository and set HF_MODEL_REPO=<owner>/<repo> "
            "in your Render service environment variables."
        )

    filename = os.environ.get('HF_MODEL_FILE', os.path.basename(checkpoint_path)).strip()
    token    = os.environ.get('HF_TOKEN', None)
    if token:
        token = token.strip()

    print(f"Downloading model from Hugging Face Hub: {repo_id}/{filename} ...")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "huggingface_hub is not installed.  Add 'huggingface_hub' to requirements.txt."
        )

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=token,
        local_dir=os.path.dirname(checkpoint_path),
    )
    # hf_hub_download may save under a cache subdir; move to expected path.
    if os.path.abspath(local_path) != os.path.abspath(checkpoint_path):
        import shutil
        shutil.move(local_path, checkpoint_path)
    print(f"Model downloaded to {checkpoint_path}")


def load_model(checkpoint_path: str) -> None:
    global model
    if not os.path.isabs(checkpoint_path):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(base_dir, checkpoint_path)

    # Render.com clones repos without resolving LFS pointers — detect and fix.
    if not os.path.exists(checkpoint_path) or _is_lfs_pointer(checkpoint_path):
        print(f"Checkpoint missing or is an LFS pointer at {checkpoint_path}.")
        _download_from_hf(checkpoint_path)

    m = DeepfakeDetector(pretrained=False).to(device)
    # mmap=True loads tensors lazily from disk instead of reading the whole
    # checkpoint into RAM (checkpoints contain ~2x optimizer state that is
    # useless for inference). Critical on low-memory hosts.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False, mmap=True)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    m.load_state_dict(state_dict)
    m.eval()
    model = m
    print(f"Loaded checkpoint: {checkpoint_path}")
    if isinstance(ckpt, dict) and 'metrics' in ckpt:
        print(f"Checkpoint val metrics: {ckpt['metrics']}")
    del ckpt
    gc.collect()


def predict_image(pil_img, threshold=0.5):
    """Run inference on a PIL image. Returns a verdict dict."""
    tensor = val_tfm(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), 1).squeeze().cpu().numpy()
    fake_p = float(probs[1])
    real_p = float(probs[0])
    label = 'FAKE' if fake_p >= threshold else 'REAL'
    conf = fake_p if label == 'FAKE' else real_p
    if fake_p >= 0.85:
        verdict = 'ALMOST CERTAINLY FAKE'
    elif fake_p >= 0.65:
        verdict = 'LIKELY FAKE'
    elif fake_p >= 0.5:
        verdict = 'POSSIBLY FAKE'
    elif fake_p >= 0.35:
        verdict = 'POSSIBLY REAL'
    elif fake_p >= 0.15:
        verdict = 'LIKELY REAL'
    else:
        verdict = 'ALMOST CERTAINLY REAL'
    return {
        'label': label,
        'verdict': verdict,
        'confidence': round(conf * 100, 1),
        'fake_prob': round(fake_p, 4),
        'real_prob': round(real_p, 4),
    }


# ───────────────────────────── Routes ──────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'device': str(device), 'model_loaded': model is not None})


@app.route('/api/predict-image', methods=['POST'])
def predict_image_route():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception as e:
        return jsonify({'error': f'Could not read image: {e}'}), 400

    result = predict_image(img)
    result['filename'] = file.filename
    return jsonify(result)


@app.route('/api/predict-video', methods=['POST'])
def predict_video_route():
    if model is None:
        return jsonify({'error': 'Model not loaded on server'}), 503
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    # Use tempfile for cross-platform compatibility (avoids /tmp hardcode on Windows)
    suffix = os.path.splitext(file.filename)[1] or '.mp4'
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    file.save(tmp_path)

    cap = None
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total <= 0:
            return jsonify({'error': 'Could not read any frames from this video'}), 400
        duration = total / fps if fps > 0 else 0
        n_sample = min(24, total)
        indices = np.linspace(0, total - 1, n_sample, dtype=int)

        frame_results = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = predict_image(pil)
            res['timestamp'] = round(idx / fps, 2) if fps > 0 else int(idx)
            frame_results.append(res)
    finally:
        # Always release the video capture before attempting to delete on Windows
        if cap is not None:
            cap.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not frame_results:
        return jsonify({'error': 'Could not extract any usable frames'}), 400

    fake_probs = [r['fake_prob'] for r in frame_results]
    mean_fp = float(np.mean(fake_probs))
    label = 'FAKE' if mean_fp >= 0.5 else 'REAL'
    conf = mean_fp if label == 'FAKE' else 1 - mean_fp
    suspicious = [r for r in frame_results if r['fake_prob'] >= 0.7]

    return jsonify({
        'filename': file.filename,
        'label': label,
        'confidence': round(conf * 100, 1),
        'mean_fake_prob': round(mean_fp, 4),
        'total_frames': total,
        'fps': round(fps, 2),
        'duration_sec': round(duration, 2),
        'frames_analyzed': len(frame_results),
        'suspicious_frame_count': len(suspicious),
        'frame_results': frame_results,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Path to a trained .pt checkpoint (e.g. checkpoints/best_model.pt)')
    ap.add_argument('--host', default='0.0.0.0', help='Bind address (default 0.0.0.0 so cloud hosts can reach it)')
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)),
                    help='Port to listen on (defaults to $PORT env var, else 5000)')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    load_model(args.checkpoint)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
