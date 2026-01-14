import io
import os
import tempfile
import hashlib
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

import librosa
import librosa.display
import soundfile as sf

from lime import lime_image
from skimage.segmentation import mark_boundaries, slic

import shap
from shap.plots import colors
try:
    import cv2
except ImportError:
    cv2 = None
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget, ClassifierOutputSoftmaxTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from matplotlib.colors import LinearSegmentedColormap

from ollama_llm import OllamaConfig, OllamaError, ollama_generate, build_llm_explanation_prompt
from xai_config import get_xai_methods



# =========================
# Config
# =========================


st.set_page_config(page_title="Unified XAI (Audio + X-ray)", layout="wide")

# Custom CSS for Streamlit components: metrics, XAI loading, hover overlay
st.markdown(
        """
        <style>
            /* Ensure HTML-in-markdown blocks can fill the column width */
            div[data-testid="stMarkdownContainer"] {
                width: 100% !important;
                max-width: none !important;
            }
            div[data-testid="stMarkdownContainer"] > div {
                width: 100% !important;
                max-width: none !important;
            }
            div[data-testid="stMarkdownContainer"] p {
                width: 100% !important;
                max-width: none !important;
            }

            /* Metric card */
            div[data-testid="stMetric"] {
                background: var(--secondary-background-color);
                padding: 14px 16px;
                border-radius: 14px;
            }
            div[data-testid="stMetric"] [data-testid="stMetricValue"] {
                font-size: 1.6rem;
            }

            /* Loading placeholder for XAI panels */
            .xai-loading {
                height: 320px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: var(--secondary-background-color);
                border-radius: 14px;
            }
            .xai-spinner {
                width: 84px;
                height: 84px;
                border-radius: 50%;
                border: 10px solid rgba(0,0,0,0);
                border-top-color: var(--text-color);
                animation: xai-spin 0.9s linear infinite;
            }
            @keyframes xai-spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            /* Hover overlay: show XAI, hover -> show original */
            .xai-hover {
                position: relative;
                display: block;
                width: 100% !important;
                max-width: 100% !important;
                box-sizing: border-box;
                border-radius: 14px;
                overflow: hidden;
                background: var(--secondary-background-color);
            }
            .xai-hover img {
                width: 100% !important;
                max-width: 100% !important;
                height: auto !important;
                display: block;
            }
            .xai-hover .xai-img {
                position: relative;
                z-index: 1;
                opacity: 1;
                transition: opacity 120ms ease-in-out;
            }
            .xai-hover .orig-img {
                position: absolute;
                inset: 0;
                z-index: 2;
                opacity: 0;
                transition: opacity 120ms ease-in-out;
            }
            .xai-hover:hover .orig-img { opacity: 1; }
            .xai-hover:hover .xai-img { opacity: 0; }
        </style>
        """,
        unsafe_allow_html=True,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RED_GREEN = LinearSegmentedColormap.from_list(
    "red_green",
    [
        (0.0, (0.0, 0.75, 0.0)),  # negative -> green
        (0.5, (1.0, 1.0, 1.0)),   # zero -> white
        (1.0, (1.0, 0.0, 0.0)),   # positive -> red
    ],
)

AUDIO_SR = 16000
AUDIO_SECONDS = 2.0
IMG_SIZE = 224

# classes
AUDIO_CLASSES = ["real", "fake"] 
XRAY_CLASSES = ["no_opacity", "opacity"]  # proxy "Lung Opacity"

WEIGHTS = {
    "audio_vgg16": "weights/audio_vgg16.pt",
    "audio_resnet50": "weights/audio_resnet50.pt",
    "audio_mobilenetv2": "weights/audio_mobilenetv2.pt",
    "xray_alexnet": "weights/xray_alexnet.pt",
    "xray_densenet121": "weights/xray_densenet121.pt",
}

# normalization ImageNet (comme training avec backbones)
NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


# =========================
# Utils
# =========================


# == PREPROCESSING ==

def load_wav_bytes(wav_bytes: bytes, target_sr=16000):
    """Load WAV bytes, convert to mono, resample to target_sr."""

    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def pad_or_trim(y: np.ndarray, sr: int, seconds: float):

    """Pad with zeros or trim it"""
    Tt = int(sr * seconds)
    if len(y) < Tt:
        y = np.pad(y, (0, Tt - len(y)), mode="constant")
    else:
        y = y[:Tt]
    return y

# # This one is Deprecated: instead using matplotlib for a better visualization (see below)
# def wav_to_melspec_rgb_uint8(y: np.ndarray, sr: int):
#     """Wav file to Mel-spectrogram image"""

#     # MelSpectrogram -> dB -> normalize 0..255
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=20, fmax=8000, power=2.0)
#     S_db = librosa.power_to_db(S, ref=np.max)
#     S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
#     img = (S_norm * 255.0).astype(np.uint8)  # (H, W)

#     # convert to RGB by stacking, then resize to 224x224
#     pil = Image.fromarray(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
#     return np.array(pil).astype(np.uint8)  # (224,224,3)

def wav_bytes_to_melspec_rgb_uint8_matplotlib(wav_bytes: bytes):
    """Wav file bytes to Mel-spectrogram image with matplotlib"""

    wav_path = None
    png_path = None

    try:
        # write temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        # load and plot
        y, sr = librosa.load(wav_path)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr)

        # save to temp png
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name
        plt.savefig(png_path)
        plt.close(fig)

        img = Image.open(png_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.array(img).astype(np.uint8)
    
    # Regardless of success or failure, make sure to clean up the temp files.
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        if png_path and os.path.exists(png_path):
            os.remove(png_path)


def xray_bytes_to_rgb_uint8(img_bytes: bytes):
    """X-ray image bytes to RGB array for model input"""

    pil = Image.open(io.BytesIO(img_bytes)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    pil = pil.convert("RGB")
    return np.array(pil).astype(np.uint8)


def spectrogram_image_bytes_to_rgb_uint8(img_bytes: bytes):
    """Spectrogram image bytes to RGB array for model input"""

    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(pil).astype(np.uint8)


def _autocrop_spectrogram_region(rgb_uint8: np.ndarray) -> np.ndarray:
    """Best-effort crop to remove borders/axes from a plotted spectrogram image."""

    rgb = _to_uint8(rgb_uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        return rgb

    gray = (0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]).astype(np.float32)

    # Find rows/cols with enough variance (content vs uniform background).
    row_std = gray.std(axis=1)
    col_std = gray.std(axis=0)

    r = np.where(row_std > 2.0)[0]
    c = np.where(col_std > 2.0)[0]
    if r.size == 0 or c.size == 0:
        return rgb

    r0, r1 = int(r[0]), int(r[-1])
    c0, c1 = int(c[0]), int(c[-1])

    # Keep a small safety margin.
    pad = 2
    r0 = max(0, r0 - pad)
    c0 = max(0, c0 - pad)
    r1 = min(rgb.shape[0] - 1, r1 + pad)
    c1 = min(rgb.shape[1] - 1, c1 + pad)

    if (r1 - r0) < 20 or (c1 - c0) < 20:
        return rgb
    return rgb[r0 : r1 + 1, c0 : c1 + 1]


def _rgb_to_colormap_scalar(rgb_uint8: np.ndarray, cmap_name: str = "magma") -> np.ndarray:
    """Invert a Matplotlib colormap to approximate a scalar field in [0, 1]."""

    rgb = _to_uint8(rgb_uint8)
    h, w, _ = rgb.shape

    cmap = plt.get_cmap(cmap_name)
    k = 256
    lut = cmap(np.linspace(0, 1, k))[:, :3].astype(np.float32)  # (k, 3)

    p = (rgb.reshape(-1, 3).astype(np.float32) / 255.0)  # (n, 3)
    p2 = np.sum(p * p, axis=1, keepdims=True)  # (n, 1)
    c2 = np.sum(lut * lut, axis=1, keepdims=True).T  # (1, k)
    pc = p @ lut.T  # (n, k)
    dist2 = p2 + c2 - 2.0 * pc
    idx = np.argmin(dist2, axis=1).astype(np.float32)
    return (idx / float(k - 1)).reshape(h, w).clip(0.0, 1.0)


def _reconstruction_quality_score(y: np.ndarray, sr: int) -> float:
    """Heuristic score to pick between vertical flip options."""

    if y is None or y.size == 0:
        return -1e9
    y = y.astype(np.float32)
    y = y - float(y.mean())
    rms = float(np.sqrt(np.mean(y * y)))
    if not np.isfinite(rms) or rms <= 1e-8:
        return -1e9

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid_hz = float(np.nanmedian(centroid)) if centroid.size else float("nan")
    if not np.isfinite(centroid_hz):
        centroid_hz = sr / 4.0

    # Prefer centroid around ~2.5kHz for speech-like audio; punish extremes.
    target = 2500.0
    centroid_penalty = abs(centroid_hz - target) / target
    return -centroid_penalty + np.log(rms + 1e-8)


@st.cache_data(show_spinner=False)
def reconstruct_wav_from_spectrogram_image_bytes(
    img_bytes: bytes,
    *,
    sr: int = AUDIO_SR,
    seconds: float | None = None,
    cmap_name: str = "magma",
    min_db: float = -80.0,
    max_db: float = 0.0,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_iter: int = 32,
) -> bytes:
    """Reconstruct a WAV preview from a spectrogram image (best-effort).
    NOTE: This cannot recover the original audio perfectly (phase + scaling are lost).
    """

    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    rgb = np.array(pil).astype(np.uint8)
    rgb = _autocrop_spectrogram_region(rgb)

    scalar = _rgb_to_colormap_scalar(rgb, cmap_name=cmap_name)
    db = (min_db + scalar * (max_db - min_db)).astype(np.float32)
    mel_power_a = librosa.db_to_power(db, ref=1.0).astype(np.float32)
    mel_power_b = np.flipud(mel_power_a)

    # If `seconds` is provided, we force the output duration. Otherwise we let librosa
    # infer the appropriate length from the spectrogram's time axis. For many uploaded
    # images, forcing a fixed duration can cause internal frame-count mismatches.
    target_len = None
    if seconds is not None:
        if not np.isfinite(seconds) or seconds <= 0:
            raise ValueError("`seconds` must be a positive number when provided")
        target_len = int(sr * float(seconds))

    mel_kwargs = dict(
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
        n_iter=n_iter,
    )

    y_a = librosa.feature.inverse.mel_to_audio(mel_power_a, **mel_kwargs).astype(np.float32)
    y_b = librosa.feature.inverse.mel_to_audio(mel_power_b, **mel_kwargs).astype(np.float32)

    score_a = _reconstruction_quality_score(y_a, sr)
    score_b = _reconstruction_quality_score(y_b, sr)
    y = y_a if score_a >= score_b else y_b

    if target_len is not None:
        y = pad_or_trim(y, sr, float(seconds))

    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1e-8:
        y = (0.98 * y / peak).astype(np.float32)

    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def rgb_uint8_to_input_tensor(rgb: np.ndarray):
    """RGB image to normalized torch tensor for model input"""

    pil = Image.fromarray(rgb)
    x = T.ToTensor()(pil)  # 0..1
    x = NORM(x)
    return x.unsqueeze(0).to(DEVICE)

def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Transform image to uint8 format if needed."""

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).round()
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def fig_to_rgb_uint8(fig) -> np.ndarray:
    """ Convert a Matplotlib figure to an RGB array."""

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)

def _rgb_uint8_to_png_bytes(rgb_uint8: np.ndarray) -> bytes:
    """Convert RGB array to PNG bytes."""

    pil = Image.fromarray(_to_uint8(rgb_uint8)).convert("RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def _img_to_data_uri_png(image: np.ndarray) -> str:
    """Convert RGB array to data URI PNG string."""

    png = _rgb_uint8_to_png_bytes(image)
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_hover_overlay_component(
    xai_uri: str,
    orig_uri: str,
    *,
    border_radius_px: int = 14,
):
    """Render a hover overlay component using HTML/CSS in Streamlit.
    Shows the XAI image by default, and on hover shows the original image."""
        
    # Use components.html to avoid Streamlit Markdown max-width constraints.
    # This renders in an iframe that can expand to the full column width.
    html = f"""
        <style>
            html, body {{ margin: 0; padding: 0; }}
            .xai-hover {{
                position: relative;
                width: 100%;
                aspect-ratio: 1 / 1;
                border-radius: {border_radius_px}px;
                overflow: hidden;
                background: transparent;
            }}
            .xai-hover img {{
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                display: block;
                transition: opacity 120ms ease-in-out;
            }}
            .xai-hover img.xai-img {{ opacity: 1; z-index: 1; }}
            .xai-hover img.orig-img {{ opacity: 0; z-index: 2; }}
            .xai-hover:hover img.orig-img {{ opacity: 1; }}
            .xai-hover:hover img.xai-img {{ opacity: 0; }}
        </style>
        <div class="xai-hover">
            <img class="xai-img" src="{xai_uri}" alt="XAI" />
            <img class="orig-img" src="{orig_uri}" alt="Original" />
        </div>
        <script>
            (function() {{
                function resizeFrameToSquare() {{
                    const el = document.querySelector('.xai-hover');
                    if (!el) return;
                    const w = el.getBoundingClientRect().width;
                    if (!w || w < 10) return;

                    const h = Math.ceil(w);
                    if (window.frameElement) {{
                        window.frameElement.style.height = h + 'px';
                        window.frameElement.style.width = '100%';
                    }}
                }}

                // Run soon after render and on resize.
                window.addEventListener('load', resizeFrameToSquare);
                window.addEventListener('resize', resizeFrameToSquare);
                setTimeout(resizeFrameToSquare, 0);
                setTimeout(resizeFrameToSquare, 50);
                setTimeout(resizeFrameToSquare, 200);
            }})();
        </script>
    """
    # Height auto-adjusted by the JS above.
    components.html(html, height=10, scrolling=False)


def _resize_rgb_uint8_to_match(image: np.ndarray, target_rgb_uint8: np.ndarray) -> np.ndarray:
    """Resize image to match target shape if needed."""

    image = _to_uint8(image)
    target_rgb_uint8 = _to_uint8(target_rgb_uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        image = np.array(Image.fromarray(image).convert("RGB"))
    if image.shape[:2] == target_rgb_uint8.shape[:2]:
        return image
    pil = Image.fromarray(image).convert("RGB")
    pil = pil.resize((target_rgb_uint8.shape[1], target_rgb_uint8.shape[0]))
    return np.array(pil).astype(np.uint8)

# == MODELS LOADING ==

def build_model_audio(name: str) -> nn.Module:
    """
    Build the audio model based on the name.
    Possible names are "audio_vgg16", "audio_resnet50", "audio_mobilenetv2".
    The training of the model are in the training_models.ipynb notebook.

    Change the final layer to output 1 logit for binary classification.
    """

    if name == "audio_vgg16":
        m = models.vgg16(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)  # 1 logit
        return m
    if name == "audio_resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)  # 1 logit
        return m
    if name == "audio_mobilenetv2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)  # 1 logit
        return m
    
    raise ValueError("Unknown audio model")


def build_model_xray(name: str) -> nn.Module:
    """
    Build the xray model based on the name.
    Possible names are "xray_alexnet", "xray_densenet121".
    The training of the model are in the training_models.ipynb notebook.

    Change the final layer to output 1 logit for binary classification.
    """

    if name == "xray_alexnet":
        m = models.alexnet(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 1)  # 1 logit
        return m
    if name == "xray_densenet121":
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)  # 1 logit
        return m
    
    raise ValueError("Unknown xray model")


@st.cache_resource
def load_model(model_key: str) -> nn.Module:
    """Function that load a model"""

    ckpt = WEIGHTS[model_key]
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing weights: {ckpt}")

    if model_key.startswith("audio_"):
        model = build_model_audio(model_key)
    else:
        model = build_model_xray(model_key)

    sd = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(sd, strict=True)
    model.to(DEVICE).eval()
    return model


# == Prediction helpers ==

@torch.inference_mode()
def predict_proba(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Predict probability for binary classification model outputting 1 logit."""

    logit = model(x)  # (1,1)
    p1 = torch.sigmoid(logit)[0, 0].detach().cpu().item()
    p0 = 1.0 - p1
    return np.array([p0, p1], dtype=np.float32)  # [class0, class1]


# =========================
# XAI: Grad-CAM
# =========================


def _summarize_grayscale_cam(grayscale_cam, *, topk: float = 0.10) -> dict:
    """Compute lightweight numeric stats from a Grad-CAM map in [0, 1]."""

    cam = np.asarray(grayscale_cam, dtype=np.float32)
    if cam.ndim != 2 or cam.size == 0:
        return {"available": False}

    cam = np.nan_to_num(cam, nan=0.0, posinf=0.0, neginf=0.0)
    cam = np.clip(cam, 0.0, 1.0)

    mean_intensity = float(cam.mean())
    max_intensity = float(cam.max())

    total = float(cam.sum())
    if total <= 1e-12:
        com_x, com_y = 0.5, 0.5
    else:
        yy, xx = np.indices(cam.shape)
        com_y = float((yy * cam).sum() / total) / float(cam.shape[0] - 1 if cam.shape[0] > 1 else 1)
        com_x = float((xx * cam).sum() / total) / float(cam.shape[1] - 1 if cam.shape[1] > 1 else 1)

    k = float(np.clip(topk, 1e-3, 0.5))
    thr = float(np.quantile(cam.reshape(-1), 1.0 - k))
    topk_coverage = float((cam >= thr).mean())

    return {
        "available": True,
        "mean_intensity": mean_intensity,
        "max_intensity": max_intensity,
        "topk": k,
        "topk_coverage": topk_coverage,
        "center_of_mass_xy": [com_x, com_y],
    }


def find_last_conv_layer(model: nn.Module):
    """Find the last Conv2d layer in the model. Used for Grad-CAM."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last

def find_last_relu_layer(model: nn.Module):
    """FFind the last ReLU layer in the model. Used for Grad-CAM."""
    last = None
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            last = m
    if last is None:
        raise ValueError("No ReLU layer found for Grad-CAM.")
    return last

class TwoClassWrapper(nn.Module):
    """Small class to wrap a binary classifier outputting 1 logit 
    into a 2-class classifier outputting 2 logits (for Grad-CAM)."""

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        logit = self.base(x)
        if logit.ndim == 1:
            logit = logit.unsqueeze(1)
        return torch.cat([-logit, logit], dim=1)

def get_gradcam_target_layers(model: nn.Module, model_key: str):
    """Get the grad-CAM target layers based on the model type.
    Fallback to last conv layer if no specific rule for this model."""

    if model_key == "xray_alexnet":
        if hasattr(model, "features") and len(model.features) > 11:
            return [model.features[11]]  # last ReLU after final conv
    if model_key == "xray_densenet121":
        if hasattr(model, "features"):
            return [find_last_relu_layer(model.features)]
    if hasattr(model, "features"):
        try:
            return [find_last_conv_layer(model.features)]
        except ValueError:
            pass
    return [find_last_conv_layer(model)]

def gradcam_explain(
    model: nn.Module,
    x: torch.Tensor,
    rgb_uint8: np.ndarray,
    class_idx: int,
    model_key: str,
    *,
    return_cam: bool = False,
):
    """Grad-CAM explanation function."""

    # setup Grad-CAM
    target_layers = get_gradcam_target_layers(model, model_key)
    # xray models needs to wrap to 2-class output
    if model_key.startswith("xray_"):
        wrapped = TwoClassWrapper(model).eval()
        cam = GradCAM(model=wrapped, target_layers=target_layers)
        cam.compute_input_gradient = True
        targets = [ClassifierOutputSoftmaxTarget(class_idx)]
    # for audio models, keep as is
    else:
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [BinaryClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=x, targets=targets)[0]  # (H,W) in [0,1] approx

    # overlay heatmap on image
    heat = np.uint8(255 * grayscale_cam)
    if cv2 is not None:
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb_uint8, 0.6, heat, 0.4, 0)
        return (overlay, grayscale_cam) if return_cam else overlay

    rgb_float = (rgb_uint8.astype(np.float32) / 255.0)

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    return (overlay, grayscale_cam) if return_cam else overlay


# =========================
# XAI: LIME
# =========================


def lime_explain(rgb_uint8: np.ndarray, model: nn.Module, *, return_details: bool = False):
    """LIME explanation function."""

    def classifier_fn(images):
        # images: list/np array of (H,W,3) uint8
        out = []
        for im in images:
            x = rgb_uint8_to_input_tensor(_to_uint8(im))
            p = predict_proba(model, x)
            out.append(p)
        return np.array(out, dtype=np.float32)

    # setup LIME explainer
    explainer = lime_image.LimeImageExplainer()

    exp = explainer.explain_instance(
        rgb_uint8.astype(np.uint8),
        classifier_fn=classifier_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    # get explanation for the predicted label
    label = int(np.argmax(predict_proba(model, rgb_uint8_to_input_tensor(rgb_uint8.astype(np.uint8)))))
    temp, mask = exp.get_image_and_mask(label, positive_only=False, num_features=8, hide_rest=True)
    vis = mark_boundaries(temp / 255.0, mask)
    out_img = (vis * 255).astype(np.uint8)

    if not return_details:
        return out_img

    mask_arr = np.asarray(mask)
    selected = float((mask_arr != 0).mean())
    pos = float((mask_arr > 0).mean())
    neg = float((mask_arr < 0).mean())

    seg_weights = exp.local_exp.get(label, [])
    seg_weights_sorted = sorted(seg_weights, key=lambda t: abs(float(t[1])), reverse=True)
    top = seg_weights_sorted[:8]
    top_pos = [(int(i), float(w)) for i, w in top if float(w) > 0][:4]
    top_neg = [(int(i), float(w)) for i, w in top if float(w) < 0][:4]

    details = {
        "available": True,
        "selected_pixel_fraction": selected,
        "positive_pixel_fraction": pos,
        "negative_pixel_fraction": neg,
        "top_segments_pos": top_pos,
        "top_segments_neg": top_neg,
    }
    return out_img, details


# =========================
# XAI: SHAP
# =========================


def shap_explain(rgb_uint8: np.ndarray, model: nn.Module, class_names, *, return_details: bool = False):
    """SHAP explanation function"""

    def f(images):
        # images is array of (H,W,3). return (N,C) array of probabilities
        out = []
        for im in images:
            x = rgb_uint8_to_input_tensor(_to_uint8(im))
            p = predict_proba(model, x)
            out.append(p)
        return np.array(out, dtype=np.float32)

    img_uint8 = rgb_uint8.astype(np.uint8)
    img_float = img_uint8.astype(np.float32) / 255.0
    masker = shap.maskers.Image("blur(64,64)", img_float.shape)
    explainer = shap.Explainer(f, masker, output_names=class_names)

    # compute SHAP values
    sv = explainer(img_float[None, ...], max_evals=1000, batch_size=8)
    probs = f(img_float[None, ...])[0]
    cls = int(np.argmax(probs))

    vals = sv.values[0, :, :, :, cls]  # (H,W,3)
    sv2d = vals.sum(axis=-1)

    # segment + average within segments
    segments = slic(img_float, n_segments=120, compactness=10, sigma=1, start_label=0)
    sv_seg = np.zeros_like(sv2d)
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        sv_seg[mask] = sv2d[mask].mean()

    # separate positive and negative impacts, scale by 99th percentile for visualization
    pos = sv_seg[sv_seg > 0]
    neg = -sv_seg[sv_seg < 0]
    pos_scale = np.nanpercentile(pos, 99) if pos.size else 1.0
    neg_scale = np.nanpercentile(neg, 99) if neg.size else 1.0

    # normalize
    sv_norm = np.zeros_like(sv_seg)
    if pos_scale > 0:
        sv_norm[sv_seg > 0] = sv_seg[sv_seg > 0] / pos_scale
    if neg_scale > 0:
        sv_norm[sv_seg < 0] = sv_seg[sv_seg < 0] / neg_scale
    sv_norm = np.clip(sv_norm, -1, 1)

    # overlay on original image (gray scale)
    x_gray = 0.2989 * img_float[:, :, 0] + 0.5870 * img_float[:, :, 1] + 0.1140 * img_float[:, :, 2]
    fig, ax = plt.subplots()
    ax.imshow(x_gray, cmap=plt.get_cmap("gray"), alpha=0.12)
    ax.imshow(sv_norm, cmap=RED_GREEN, vmin=-1, vmax=1, alpha=0.85)
    ax.axis("off")
    out = fig_to_rgb_uint8(fig)
    plt.close(fig)

    if not return_details:
        return out

    # compute numeric details
    strong = 0.20
    pos_frac = float((sv_norm > strong).mean())
    neg_frac = float((sv_norm < -strong).mean())
    mean_abs = float(np.mean(np.abs(sv_norm)))

    # top segments by mean contribution
    seg_means = []
    for seg_id in np.unique(segments):
        m = segments == seg_id
        seg_means.append((int(seg_id), float(sv_seg[m].mean())))
    seg_means_sorted = sorted(seg_means, key=lambda t: abs(t[1]), reverse=True)[:12]
    top_pos = [(i, v) for i, v in seg_means_sorted if v > 0][:4]
    top_neg = [(i, v) for i, v in seg_means_sorted if v < 0][:4]

    # return full details
    details = {
        "available": True,
        "explained_class": str(class_names[cls]) if cls < len(class_names) else str(cls),
        "positive_fraction_strong": pos_frac,
        "negative_fraction_strong": neg_frac,
        "mean_abs_contribution": mean_abs,
        "top_segments_pos": top_pos,
        "top_segments_neg": top_neg,
    }
    return out, details


# =========================
# UI
# =========================


def available_xai(file_ext: str):
    return get_xai_methods(file_ext)


def _infer_upload_extension(uploaded_file) -> str:
    """Infer the uploaded file extension in lowercase."""

    if uploaded_file is None:
        return ""

    name = getattr(uploaded_file, "name", "") or ""
    ext = os.path.splitext(name)[1].lower()
    return ext


def run_xai(
    method: str,
    model: nn.Module,
    x: torch.Tensor,
    rgb_uint8: np.ndarray,
    class_names,
    model_key: str,
):
    """Utils that runs the XAI method selected."""

    probs = predict_proba(model, x)
    cls = int(np.argmax(probs))
    if method == "Grad-CAM":
        return gradcam_explain(model, x, rgb_uint8, cls, model_key)
    if method == "LIME":
        return lime_explain(rgb_uint8, model)
    if method == "SHAP":
        return shap_explain(rgb_uint8, model, class_names)
    return None


def main():
    """Main Streamlit app function."""
    
    # Title
    st.title("Unified Explainable AI Interface")
    st.caption(f"Device: {DEVICE}")

    # Sidebar
    with st.sidebar:

        st.header("Configuration", divider="red")

        # Select the type of data/task
        section = st.radio(
            "Section",
            ["Deepfake audio detection", "Lung cancer detection"],
            horizontal=False,
        )

        st.divider()

        # Depending of the task chosen
        if section == "Deepfake audio detection":
            # Audio uploader
            modality = "Audio"
            up = st.file_uploader(
                "Upload audio (.wav) or spectrogram image",
                type=["wav", "png", "jpg", "jpeg"],
            )
            st.divider()
            # model selection
            model_key = st.selectbox("Audio model", ["audio_vgg16", "audio_resnet50", "audio_mobilenetv2"])
            class_names = AUDIO_CLASSES
        else:
            # X-ray uploader
            modality = "X-ray (image)"
            up = st.file_uploader("Upload chest X-ray image", type=["png", "jpg", "jpeg"])
            st.divider()
            # model selection
            model_key = st.selectbox("X-ray model", ["xray_alexnet", "xray_densenet121"])
            class_names = XRAY_CLASSES

        # Select if LLM explanation is enabled or not
        st.divider()
        st.subheader("LLM explanation (local)")
        enable_llm = st.checkbox("Enable Ollama explanation", value=False)
        # default values
        ollama_host = "http://localhost:11434"
        ollama_model = "gpt-oss:20b"
        # inputs for ollama configuration
        if enable_llm:
            ollama_host = st.text_input("Ollama host", value=ollama_host)
            ollama_model = st.text_input("Ollama model", value=ollama_model)

        if up is None:
            st.stop()

        upload_ext = _infer_upload_extension(up)
        upload_label = upload_ext if upload_ext else "file"

        # deprecated variables. Before xai methods were variables but both task are on image input.
        xai_single = "Grad-CAM"
        xai_multi = available_xai(upload_ext)

    st.subheader(section)

    tab1, tab2 = st.tabs(["Result", "Compare"])

    model = load_model(model_key)
    file_bytes = up.getvalue()

    # Clear cached XAI + image URIs + llm explanations whenever a new file is uploaded.
    # Images are cached in session_state for performance and because the overlay 
    # refresh the page and would restart the xai computation.
    # Deleted otherwise if same file is uploaded again won't recompute xai even 
    # tho not the same output or even model and free spaces.
    # (Model weights remain cached via st.cache_resource.)
    input_sig = hashlib.sha1(file_bytes).hexdigest()
    last_sig_key = f"last_input_sig_{section}"
    if st.session_state.get(last_sig_key) != input_sig:
        st.session_state[last_sig_key] = input_sig
        st.session_state["xai_cache"] = {}
        st.session_state["uri_cache"] = {}
        st.session_state["xai_stats_cache"] = {}
        st.session_state["llm_cache"] = {}

    # Determine if the uploaded file is a WAV audio file to support both audio and spectrogram image inputs.
    is_wav_audio = (
        section == "Deepfake audio detection"
        and hasattr(up, "name")
        and isinstance(up.name, str)
        and up.name.lower().endswith(".wav")
    )

    # Prepare input as RGB + torch tensor
    if section == "Deepfake audio detection":
        # support of .wav file
        if is_wav_audio:
            rgb_uint8 = wav_bytes_to_melspec_rgb_uint8_matplotlib(file_bytes)
        else:
            rgb_uint8 = spectrogram_image_bytes_to_rgb_uint8(file_bytes)
        x = rgb_uint8_to_input_tensor(rgb_uint8)
    else:
        rgb_uint8 = xray_bytes_to_rgb_uint8(file_bytes)
        x = rgb_uint8_to_input_tensor(rgb_uint8)

    # Prediction
    probs = predict_proba(model, x)
    cls = int(np.argmax(probs))
    pred_label = class_names[cls]
    pred_score = float(probs[cls])

    # Tab Results
    with tab1:
        c1, c2 = st.columns([1, 1])

        # show image input + audio if applicable
        with c1:
            st.subheader("Input")
            if section == "Deepfake audio detection":
                if is_wav_audio:
                    st.image(rgb_uint8, caption="Mel-spectrogram (as image)", use_container_width=True)
                    st.audio(file_bytes, format="audio/wav")
                else:
                    st.image(rgb_uint8, caption="Spectrogram image", use_container_width=True)

                    # Automatically make the image-upload case listenable.
                    with st.spinner("Reconstructing audio from spectrogram image…"):
                        try:
                            wav_preview = reconstruct_wav_from_spectrogram_image_bytes(file_bytes)
                            st.audio(wav_preview, format="audio/wav")
                        except Exception as e:
                            st.warning(f"Could not reconstruct audio from this image: {e}")
            else:
                st.image(rgb_uint8, caption="X-ray (RGB view)", use_container_width=True)

        # Show metrics and results
        with c2:
            st.subheader("Prediction")
            m1, m2 = st.columns(2)
            with m1:
                st.metric(
                    "Predicted class", 
                    pred_label, 
                    f"{pred_score*100:.2f}%",
                    border=True,
                    height=140,
                )
            with m2:
                margin = float(abs(probs[1] - probs[0]))
                st.metric(
                    "Confidence margin", 
                    f"{margin*100:.2f}%", 
                    border=True, 
                    height=140,
                )

            st.json({class_names[i]: float(probs[i]) for i in range(len(class_names))})

            # if llm enables show a section for it
            if enable_llm:
                st.divider()
                with st.expander("LLM explanation (Ollama)", expanded=False):
                    st.caption(
                        "Requires a local Ollama server served (see README)."
                        "No images are being passed to the LLM and no tools support is needed." 
                        "Therefore, most of ollama's model will work. Write in the sidebar the one you prefer."
                        "NOTE: for better interpretability wait that the xai methods are computed in the Compare tab first."
                    )

                    cfg = OllamaConfig(host=ollama_host, model=ollama_model, timeout_s=60.0)
                    llm_cache = st.session_state.setdefault("llm_cache", {})
                    llm_key = ("llm", section, model_key, input_sig, cfg.host, cfg.model)

                    b1, b2 = st.columns([1, 2])
                    with b1:
                        generate = st.button("Generate", key=f"llm_gen_{section}_{model_key}_{input_sig}")
                    with b2:
                        st.write("")

                    if generate:
                        # Build optional numeric summaries from XAI.
                        # if any, used the cached stats from gradcam/lime/shap to generate the prompt.
                        xai_summaries = {}
                        stats_cache = st.session_state.get("xai_stats_cache", {})
                        if "Grad-CAM" in xai_multi:
                            try:
                                _, cam_map = gradcam_explain(
                                    model,
                                    x,
                                    rgb_uint8,
                                    cls,
                                    model_key,
                                    return_cam=True,
                                )
                                xai_summaries["Grad-CAM"] = _summarize_grayscale_cam(cam_map)
                            except Exception:
                                xai_summaries["Grad-CAM"] = {"available": False}

                        if "LIME" in xai_multi:
                            lime_key = (section, model_key, input_sig, "LIME")
                            xai_summaries["LIME"] = stats_cache.get(
                                lime_key, {"available": False, "note": "Run Compare tab to compute."}
                            )

                        if "SHAP" in xai_multi:
                            shap_key = (section, model_key, input_sig, "SHAP")
                            xai_summaries["SHAP"] = stats_cache.get(
                                shap_key, {"available": False, "note": "Run Compare tab to compute."}
                            )

                        prompt = build_llm_explanation_prompt(
                            task_name=section,
                            model_key=model_key,
                            class_names=list(class_names),
                            probs={class_names[i]: float(probs[i]) for i in range(len(class_names))},
                            predicted_class=pred_label,
                            confidence_margin=float(abs(probs[1] - probs[0])),
                            input_kind=("wav audio" if is_wav_audio else ("spectrogram image" if section == "Deepfake audio detection" else "x-ray image")),
                            xai_methods=list(xai_multi),
                            xai_summaries=xai_summaries,
                        )
                        with st.spinner("Calling Ollama…"):
                            try:
                                llm_text = ollama_generate(cfg=cfg, prompt=prompt)
                                llm_cache[llm_key] = llm_text
                            except OllamaError as e:
                                st.error(str(e))
                    if llm_key in llm_cache:
                        st.markdown(llm_cache[llm_key])

        # # XAI single output deprecated for Compare tab
        # st.divider()
        # st.subheader(f"XAI: {xai_single}")
        # out = run_xai(xai_single, model, x, rgb_uint8, class_names, model_key)
        # st.image(out, use_container_width=True)

    # Tab Compare
    with tab2:
        st.subheader("Compare XAI methods")
        st.caption("Note: SHAP can be slow; we cap max_evals to keep it usable.")
        if xai_multi:
            method_labels = " ".join(f"`{m}`" for m in xai_multi)
            st.markdown(f"XAI for {upload_label} type of entry: {method_labels}")
        else:
            st.info(f"No XAI methods configured for {upload_label} entries.")
            return
        # input_sig computed above

        overlay_mode = st.toggle(
            "Overlay mode (hover to view original)",
            value=False,
            key=f"overlay_mode_{section}_{input_sig}",
        )

        n_cols = min(3, len(xai_multi))
        cols = st.columns(n_cols)
        xai_cache = st.session_state.setdefault("xai_cache", {})
        uri_cache = st.session_state.setdefault("uri_cache", {})
        xai_stats_cache = st.session_state.setdefault("xai_stats_cache", {})

        orig_uri_key = ("orig", section, input_sig)
        if orig_uri_key not in uri_cache:
            uri_cache[orig_uri_key] = _img_to_data_uri_png(_to_uint8(rgb_uint8))
        orig_uri = uri_cache[orig_uri_key]
        
        # For each XAI method, compute and display
        for i, method in enumerate(xai_multi):

            with cols[i % len(cols)]:
                st.markdown(f"### {method}")

                # Spinner placeholder
                placeholder = st.empty()
                placeholder.markdown(
                    '<div class="xai-loading"><div class="xai-spinner"></div></div>',
                    unsafe_allow_html=True,
                )
                
                # check cache to display xai result if already computed
                cache_key = (section, model_key, input_sig, method)
                if cache_key in xai_cache:
                    out = xai_cache[cache_key]
                else:
                    # Compute XAI and store lightweight stats for later reuse (e.g., LLM prompt).
                    if method == "Grad-CAM":
                        out, cam_map = gradcam_explain(
                            model,
                            x,
                            rgb_uint8,
                            cls,
                            model_key,
                            return_cam=True,
                        )
                        xai_stats_cache[cache_key] = _summarize_grayscale_cam(cam_map)
                    elif method == "LIME":
                        out, details = lime_explain(rgb_uint8, model, return_details=True)
                        if isinstance(details, dict):
                            xai_stats_cache[cache_key] = details
                    elif method == "SHAP":
                        out, details = shap_explain(rgb_uint8, model, class_names, return_details=True)
                        if isinstance(details, dict):
                            xai_stats_cache[cache_key] = details
                    else:
                        out = run_xai(method, model, x, rgb_uint8, class_names, model_key)
                    out = _resize_rgb_uint8_to_match(out, rgb_uint8)
                    xai_cache[cache_key] = out

                # Ensure cached outputs are aligned
                out = _resize_rgb_uint8_to_match(out, rgb_uint8)

                # overlay mode to display the original on hover
                if overlay_mode:
                    xai_uri_key = ("xai", section, model_key, input_sig, method)
                    if xai_uri_key not in uri_cache:
                        uri_cache[xai_uri_key] = _img_to_data_uri_png(_to_uint8(out))
                    xai_uri = uri_cache[xai_uri_key]

                    placeholder.empty()
                    _render_hover_overlay_component(
                        xai_uri,
                        orig_uri,
                    )
                else:
                    placeholder.empty()
                    st.image(out, use_container_width=True)


if __name__ == "__main__":
    main()
