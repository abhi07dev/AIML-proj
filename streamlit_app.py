# SPECTRA — Streamlit front-end for Streamlit Community Cloud.
"""Streamlit UI for the SPECTRA deepfake detector.

Reuses the model loading and prediction logic from app.py (same checkpoint
architecture), so the Flask site and this UI are always in sync.

Deployment notes (Streamlit Community Cloud):
  - Set the main file path to `streamlit_app.py`.
  - The checkpoint is NOT in the repo (checkpoints/*.pt is gitignored), so the
    app downloads it from Hugging Face Hub at first startup. Put the model at
    a HF model repo (e.g. abhi07dev/deepfake-detector) and add to Settings ->
    Secrets:
        HF_MODEL_REPO = "abhi07dev/deepfake-detector"
        # HF_TOKEN = "hf_..."   # only if the repo is private
"""
import io
import os
import tempfile

import numpy as np
import streamlit as st

# Pull Hugging Face config from Streamlit Secrets into the environment so
# app.py's HF download path (os.environ) works unchanged. st.secrets raises if
# no secrets.toml exists, so look up defensively.
def _secret(name):
    try:
        return st.secrets.get(name)
    except Exception:
        return None


_hf_repo = _secret("HF_MODEL_REPO")
if _hf_repo:
    os.environ.setdefault("HF_MODEL_REPO", str(_hf_repo))
_hf_token = _secret("HF_TOKEN")
if _hf_token:
    os.environ.setdefault("HF_TOKEN", str(_hf_token))

CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/best_model.pt")

# Import after env setup: app.py caps threads / disables CUDA before it imports
# torch, and exposes the shared load_model + predict_image used below.
from app import device, load_model, predict_image


@st.cache_resource(show_spinner="Loading model...")
def ensure_model():
    """Load the checkpoint once per process (cached across reruns)."""
    load_model(CHECKPOINT)
    return True


def analyze_image_bytes(data: bytes, filename: str):
    """Decode + run inference on a single uploaded image."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    result = predict_image(img)
    result["filename"] = filename
    return result, img


def analyze_video_bytes(data: bytes, filename: str):
    """Sample up to 24 frames evenly across the clip and score each one."""
    import cv2

    suffix = os.path.splitext(filename)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            f.write(data)
    except OSError:
        os.remove(tmp_path)
        raise

    cap = None
    try:
        cap = cv2.VideoCapture(tmp_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total <= 0:
            raise ValueError("Could not read any frames from this video")
        duration = total / fps if fps > 0 else 0
        n_sample = min(24, total)
        indices = np.linspace(0, total - 1, n_sample, dtype=int)

        from PIL import Image

        frame_results = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            res = predict_image(pil)
            res["timestamp"] = round(idx / fps, 2) if fps > 0 else int(idx)
            frame_results.append(res)
    finally:
        if cap is not None:
            cap.release()
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not frame_results:
        raise ValueError("Could not extract any usable frames")
    return frame_results, total, fps, duration


def render_verdict(res, header="Result"):
    """Render a single prediction's verdict + confidence."""
    color = "#d32f2f" if res["label"] == "FAKE" else "#2e7d32"
    st.markdown(
        f"<h2 style='color:{color};margin-bottom:0'>"
        f"{res['label']} — {res['verdict']}</h2>",
        unsafe_allow_html=True,
    )
    st.progress(int(round(res["confidence"])))
    st.write(
        f"Confidence: **{res['confidence']}%** &nbsp;|&nbsp; "
        f"Fake: {res['fake_prob']:.3f} &nbsp;|&nbsp; Real: {res['real_prob']:.3f}"
    )


def render_image_tab():
    st.subheader("Image analysis")
    up = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
    if up is None:
        return
    result, img = analyze_image_bytes(up.getvalue(), up.name)
    st.image(img, width=280)
    render_verdict(result)


def render_video_tab():
    st.subheader("Video analysis")
    up = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if up is None:
        return
    frame_results, total, fps, duration = analyze_video_bytes(up.getvalue(), up.name)

    fake_probs = [r["fake_prob"] for r in frame_results]
    mean_fp = float(np.mean(fake_probs))
    label = "FAKE" if mean_fp >= 0.5 else "REAL"
    conf = mean_fp if label == "FAKE" else 1 - mean_fp
    color = "#d32f2f" if label == "FAKE" else "#2e7d32"
    st.markdown(
        f"<h2 style='color:{color};margin-bottom:0'>{label} — {conf * 100:.1f}%</h2>",
        unsafe_allow_html=True,
    )
    st.write(
        f"Frames analyzed: **{len(frame_results)} / {total}** | "
        f"FPS: {fps:.1f} | Duration: {duration:.1f}s | "
        f"Mean fake probability: **{mean_fp:.3f}**"
    )
    suspicious = sum(1 for r in frame_results if r["fake_prob"] >= 0.7)
    st.write(f"Suspicious frames (fake probability >= 70%): **{suspicious}**")

    import pandas as pd

    st.line_chart(
        pd.DataFrame(
            {
                "time (s)": [r["timestamp"] for r in frame_results],
                "fake probability": fake_probs,
            }
        ).set_index("time (s)")
    )

    with st.expander(f"Per-frame results ({len(frame_results)} frames)"):
        for r in frame_results:
            st.write(f"t={r['timestamp']}s — **{r['label']}** — fake {r['fake_prob']:.3f}")


def main():
    st.set_page_config(page_title="SPECTRA — Deepfake Detector", page_icon="\U0001F6E2", layout="centered")

    st.title("SPECTRA — Deepfake Detector")
    st.caption("Dual-stream EfficientNet-B4 (spatial + FFT frequency) deepfake detector.")

    with st.sidebar:
        st.header("System")
        try:
            ensure_model()
            st.success(f"Model loaded ({device})")
        except Exception as e:
            st.error(
                f"Could not load the model: {e}\n\n"
                "Make sure the checkpoint is on Hugging Face Hub and set "
                "`HF_MODEL_REPO` in Streamlit Secrets."
            )
            st.stop()

    tab_img, tab_vid = st.tabs(["Image", "Video"])
    with tab_img:
        render_image_tab()
    with tab_vid:
        render_video_tab()


if __name__ == "__main__":
    main()
