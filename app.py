import io
import os
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
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



# =========================
# Config
# =========================


st.set_page_config(page_title="Unified XAI (Audio + X-ray)", layout="wide")

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
AUDIO_CLASSES = ["real", "fake"]  # adapte si tu as l'ordre inverse
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
    y, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return y.astype(np.float32), sr


def pad_or_trim(y: np.ndarray, sr: int, seconds: float):
    Tt = int(sr * seconds)
    if len(y) < Tt:
        y = np.pad(y, (0, Tt - len(y)), mode="constant")
    else:
        y = y[:Tt]
    return y


def wav_to_melspec_rgb_uint8(y: np.ndarray, sr: int):
    # MelSpectrogram -> dB -> normalize 0..255
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=20, fmax=8000, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-9)
    img = (S_norm * 255.0).astype(np.uint8)  # (H, W)

    # convert to RGB by stacking, then resize to 224x224
    pil = Image.fromarray(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.array(pil).astype(np.uint8)  # (224,224,3)

def wav_bytes_to_melspec_rgb_uint8_matplotlib(wav_bytes: bytes):
    wav_path = None
    png_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            wav_path = f.name

        y, sr = librosa.load(wav_path)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

        ms = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(ms, ref=np.max)
        librosa.display.specshow(log_ms, sr=sr)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            png_path = f.name
        plt.savefig(png_path)
        plt.close(fig)

        img = Image.open(png_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.array(img).astype(np.uint8)
    finally:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
        if png_path and os.path.exists(png_path):
            os.remove(png_path)


def xray_bytes_to_rgb_uint8(img_bytes: bytes):
    pil = Image.open(io.BytesIO(img_bytes)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    pil = pil.convert("RGB")
    return np.array(pil).astype(np.uint8)


def rgb_uint8_to_input_tensor(rgb: np.ndarray):
    pil = Image.fromarray(rgb)
    x = T.ToTensor()(pil)  # 0..1
    x = NORM(x)
    return x.unsqueeze(0).to(DEVICE)

def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255.0).round()
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def fig_to_rgb_uint8(fig) -> np.ndarray:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)


# == MODELS LOADING ==


def build_model_audio(name: str) -> nn.Module:
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
    logit = model(x)  # (1,1)
    p1 = torch.sigmoid(logit)[0, 0].detach().cpu().item()
    p0 = 1.0 - p1
    return np.array([p0, p1], dtype=np.float32)  # [class0, class1]


# =========================
# XAI: Grad-CAM
# =========================


def find_last_conv_layer(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    if last is None:
        raise ValueError("No Conv2d layer found for Grad-CAM.")
    return last

def find_last_relu_layer(model: nn.Module):
    last = None
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            last = m
    if last is None:
        raise ValueError("No ReLU layer found for Grad-CAM.")
    return last

class TwoClassWrapper(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base

    def forward(self, x):
        logit = self.base(x)
        if logit.ndim == 1:
            logit = logit.unsqueeze(1)
        return torch.cat([-logit, logit], dim=1)

def get_gradcam_target_layers(model: nn.Module, model_key: str):
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
):
    target_layers = get_gradcam_target_layers(model, model_key)
    if model_key.startswith("xray_"):
        wrapped = TwoClassWrapper(model).eval()
        cam = GradCAM(model=wrapped, target_layers=target_layers)
        cam.compute_input_gradient = True
        targets = [ClassifierOutputSoftmaxTarget(class_idx)]
    else:
        cam = GradCAM(model=model, target_layers=target_layers)
        targets = [BinaryClassifierOutputTarget(class_idx)]
    grayscale_cam = cam(input_tensor=x, targets=targets)[0]  # (H,W) in [0,1] approx

    heat = np.uint8(255 * grayscale_cam)
    if cv2 is not None:
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(rgb_uint8, 0.6, heat, 0.4, 0)
        return overlay

    rgb_float = (rgb_uint8.astype(np.float32) / 255.0)
    return show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)


# =========================
# XAI: LIME
# =========================


def lime_explain(rgb_uint8: np.ndarray, model: nn.Module):
    def classifier_fn(images):
        # images: list/np array of (H,W,3) uint8
        out = []
        for im in images:
            x = rgb_uint8_to_input_tensor(_to_uint8(im))
            p = predict_proba(model, x)
            out.append(p)
        return np.array(out, dtype=np.float32)

    explainer = lime_image.LimeImageExplainer()
    exp = explainer.explain_instance(
        rgb_uint8.astype(np.uint8),
        classifier_fn=classifier_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    label = int(np.argmax(predict_proba(model, rgb_uint8_to_input_tensor(rgb_uint8.astype(np.uint8)))))
    temp, mask = exp.get_image_and_mask(label, positive_only=False, num_features=8, hide_rest=True)
    vis = mark_boundaries(temp / 255.0, mask)
    return (vis * 255).astype(np.uint8)


# =========================
# XAI: SHAP
# =========================


def shap_explain(rgb_uint8: np.ndarray, model: nn.Module, class_names):
    # SHAP image explainer (slow): keep it bounded
    def f(images):
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

    sv = explainer(img_float[None, ...], max_evals=1000, batch_size=8)
    probs = f(img_float[None, ...])[0]
    cls = int(np.argmax(probs))

    vals = sv.values[0, :, :, :, cls]  # (H,W,3)
    sv2d = vals.sum(axis=-1)

    segments = slic(img_float, n_segments=120, compactness=10, sigma=1, start_label=0)
    sv_seg = np.zeros_like(sv2d)
    for seg_id in np.unique(segments):
        mask = segments == seg_id
        sv_seg[mask] = sv2d[mask].mean()

    pos = sv_seg[sv_seg > 0]
    neg = -sv_seg[sv_seg < 0]
    pos_scale = np.nanpercentile(pos, 99) if pos.size else 1.0
    neg_scale = np.nanpercentile(neg, 99) if neg.size else 1.0

    sv_norm = np.zeros_like(sv_seg)
    if pos_scale > 0:
        sv_norm[sv_seg > 0] = sv_seg[sv_seg > 0] / pos_scale
    if neg_scale > 0:
        sv_norm[sv_seg < 0] = sv_seg[sv_seg < 0] / neg_scale
    sv_norm = np.clip(sv_norm, -1, 1)

    x_gray = 0.2989 * img_float[:, :, 0] + 0.5870 * img_float[:, :, 1] + 0.1140 * img_float[:, :, 2]
    fig, ax = plt.subplots()
    ax.imshow(x_gray, cmap=plt.get_cmap("gray"), alpha=0.12)
    ax.imshow(sv_norm, cmap=RED_GREEN, vmin=-1, vmax=1, alpha=0.85)
    ax.axis("off")
    out = fig_to_rgb_uint8(fig)
    plt.close(fig)
    return out


# =========================
# UI
# =========================


def available_xai(modality: str):
    # pour ton projet de base: LIME/SHAP/Grad-CAM partout
    # (si tu ajoutes une XAI audio-only plus tard, tu la filtreras ici)
    return ["Grad-CAM", "LIME", "SHAP"]


def run_xai(
    method: str,
    model: nn.Module,
    x: torch.Tensor,
    rgb_uint8: np.ndarray,
    class_names,
    model_key: str,
):
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
    st.title("Unified Explainable AI Interface (Audio + X-ray)")
    st.caption(f"Device: {DEVICE}")

    tab1, tab2 = st.tabs(["Single run", "Compare"])

    with st.sidebar:
        modality = st.radio("Modality", ["Audio (.wav)", "X-ray (image)"], horizontal=False)

        if modality.startswith("Audio"):
            up = st.file_uploader("Upload FoR audio (.wav)", type=["wav"])
            model_key = st.selectbox("Audio model", ["audio_vgg16", "audio_resnet50", "audio_mobilenetv2"])
            class_names = AUDIO_CLASSES
        else:
            up = st.file_uploader("Upload X-ray image", type=["png", "jpg", "jpeg"])
            model_key = st.selectbox("X-ray model", ["xray_alexnet", "xray_densenet121"])
            class_names = XRAY_CLASSES

        if up is None:
            st.stop()

        xai_single = st.selectbox("XAI method (single)", available_xai(modality))
        xai_multi = st.multiselect("XAI methods (compare)", available_xai(modality), default=["Grad-CAM", "LIME", "SHAP"])

    model = load_model(model_key)
    file_bytes = up.getvalue()

    # Prepare input as RGB uint8 + torch tensor
    if modality.startswith("Audio"):
        rgb_uint8 = wav_bytes_to_melspec_rgb_uint8_matplotlib(file_bytes)
        x = rgb_uint8_to_input_tensor(rgb_uint8)
    else:
        rgb_uint8 = xray_bytes_to_rgb_uint8(file_bytes)
        x = rgb_uint8_to_input_tensor(rgb_uint8)

    probs = predict_proba(model, x)
    cls = int(np.argmax(probs))
    pred_label = class_names[cls]
    pred_score = float(probs[cls])

    # Tab Single
    with tab1:
        c1, c2 = st.columns([1, 1])

        with c1:
            st.subheader("Input")
            if modality.startswith("Audio"):
                st.image(rgb_uint8, caption="Mel-spectrogram (as image)", use_container_width=True)
                st.audio(file_bytes, format="audio/wav")
            else:
                st.image(rgb_uint8, caption="X-ray (RGB view)", use_container_width=True)

        with c2:
            st.subheader("Prediction")
            st.metric("Predicted class", pred_label, f"{pred_score:.3f}")
            st.json({class_names[i]: float(probs[i]) for i in range(len(class_names))})

        st.divider()
        st.subheader(f"XAI: {xai_single}")
        out = run_xai(xai_single, model, x, rgb_uint8, class_names, model_key)
        st.image(out, use_container_width=True)

    # Tab Compare
    with tab2:
        st.subheader("Compare XAI methods")
        if len(xai_multi) == 0:
            st.info("Select at least one method.")
            st.stop()

        cols = st.columns(min(3, len(xai_multi)))
        for i, method in enumerate(xai_multi):
            with cols[i % len(cols)]:
                st.markdown(f"### {method}")
                out = run_xai(method, model, x, rgb_uint8, class_names, model_key)
                st.image(out, use_container_width=True)

        st.caption("Note: SHAP can be slow; we cap max_evals to keep it usable.")



if __name__ == "__main__":
    main()