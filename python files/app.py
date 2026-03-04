"""
DeepShield Enterprise — Deepfake & Spoof Attack Detection System
Professional Edition | AI & Data Science Final Year Project
"""

import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import timm
from PIL import Image
from torchvision import transforms

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeepShield Enterprise",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — ENTERPRISE FORENSIC INTELLIGENCE PLATFORM
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600;700&display=swap');

/* ── Reset & Root ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --ink-1:     #f0f2f5;
    --ink-2:     #9aa3b2;
    --ink-3:     #5c6478;
    --ink-4:     #363d50;
    --bg-deep:   #080b11;
    --bg-base:   #0d1017;
    --bg-raised: #111620;
    --bg-float:  #161c2a;
    --line:      rgba(255,255,255,0.06);
    --line-mid:  rgba(255,255,255,0.10);
    --azure:     #3d8ef8;
    --azure-dim: rgba(61,142,248,0.15);
    --azure-glow:rgba(61,142,248,0.35);
    --crimson:   #f04b5a;
    --crim-dim:  rgba(240,75,90,0.15);
    --emerald:   #2dd4a0;
    --emer-dim:  rgba(45,212,160,0.15);
    --amber:     #f0a84b;
    --amber-dim: rgba(240,168,75,0.12);
    --radius-sm: 6px;
    --radius-md: 10px;
    --radius-lg: 16px;
}

/* ── Base ── */
html, body, [class*="css"], .stApp {
    background-color: var(--bg-base) !important;
    color: var(--ink-1);
    font-family: 'Geist', sans-serif;
    font-size: 14px;
    line-height: 1.6;
}

/* ── Scanline texture overlay ── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255,255,255,0.012) 2px,
        rgba(255,255,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Navigation Bar ── */
.nav-bar {
    background: linear-gradient(180deg, rgba(13,16,23,0.98) 0%, rgba(13,16,23,0.92) 100%);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--line-mid);
    padding: 0 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 60px;
    margin: -1rem -1rem 0 -1rem;
    position: sticky;
    top: 0;
    z-index: 100;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.nav-logo-mark {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--azure) 0%, #6b5cf6 100%);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 20px var(--azure-glow);
}
.nav-logo-text {
    font-family: 'Geist', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--ink-1);
    letter-spacing: -0.3px;
}
.nav-logo-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--ink-3);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: -2px;
}
.nav-status {
    display: flex;
    align-items: center;
    gap: 20px;
}
.status-pill {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-float);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--ink-2);
}
.pulse-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}
.nav-version {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink-4);
}

/* ── Hero Section ── */
.hero {
    padding: 2.8rem 0 2rem;
    border-bottom: 1px solid var(--line);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -80px; right: -100px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(61,142,248,0.06) 0%, transparent 65%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--azure);
    margin-bottom: 0.7rem;
}
.hero-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.6rem;
    font-weight: 400;
    line-height: 1.15;
    color: var(--ink-1);
    letter-spacing: -0.5px;
    margin-bottom: 0.6rem;
}
.hero-title em {
    font-style: italic;
    color: var(--azure);
}
.hero-desc {
    font-size: 0.88rem;
    color: var(--ink-3);
    max-width: 520px;
    line-height: 1.7;
}
.hero-tags {
    display: flex;
    gap: 8px;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    color: var(--ink-3);
    background: var(--bg-raised);
    border: 1px solid var(--line-mid);
    border-radius: 4px;
    padding: 3px 8px;
}

/* ── Panel / Card ── */
.panel {
    background: var(--bg-raised);
    border: 1px solid var(--line);
    border-radius: var(--radius-lg);
    overflow: hidden;
}
.panel-header {
    padding: 1rem 1.4rem;
    border-bottom: 1px solid var(--line);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-3);
}
.panel-body { padding: 1.4rem; }

/* ── Upload area ── */
[data-testid="stFileUploader"] > div {
    background: var(--bg-float) !important;
    border: 1.5px dashed var(--line-mid) !important;
    border-radius: var(--radius-md) !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--azure-dim) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

/* ── Verdict Display ── */
.verdict-wrapper {
    border-radius: var(--radius-md);
    padding: 1.6rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1.2rem;
    animation: fadeSlideIn 0.4s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.verdict-threat {
    background: var(--crim-dim);
    border: 1px solid rgba(240,75,90,0.3);
}
.verdict-safe {
    background: var(--emer-dim);
    border: 1px solid rgba(45,212,160,0.3);
}
.verdict-warn {
    background: var(--amber-dim);
    border: 1px solid rgba(240,168,75,0.3);
}
.verdict-icon {
    width: 44px; height: 44px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    flex-shrink: 0;
}
.verdict-threat .verdict-icon { background: rgba(240,75,90,0.2); }
.verdict-safe   .verdict-icon { background: rgba(45,212,160,0.2); }
.verdict-warn   .verdict-icon { background: rgba(240,168,75,0.2); }
.verdict-content { flex: 1; }
.verdict-class {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.verdict-threat .verdict-class { color: var(--crimson); }
.verdict-safe   .verdict-class { color: var(--emerald); }
.verdict-warn   .verdict-class { color: var(--amber); }
.verdict-headline {
    font-family: 'Instrument Serif', serif;
    font-size: 1.4rem;
    font-weight: 400;
    color: var(--ink-1);
    line-height: 1.2;
}
.verdict-note {
    font-size: 0.78rem;
    color: var(--ink-3);
    margin-top: 4px;
}

/* ── Metric Tiles ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 1.2rem;
}
.metric-tile {
    background: var(--bg-float);
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    padding: 0.9rem 1rem;
}
.metric-tile-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--ink-4);
    margin-bottom: 5px;
}
.metric-tile-value {
    font-family: 'DM Mono', monospace;
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--ink-1);
}
.metric-tile-sub {
    font-size: 0.7rem;
    color: var(--ink-3);
    margin-top: 2px;
}

/* ── Probability Bars ── */
.prob-section { margin-bottom: 1.2rem; }
.prob-item { margin-bottom: 10px; }
.prob-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.prob-name {
    font-size: 0.78rem;
    color: var(--ink-2);
    display: flex;
    align-items: center;
    gap: 6px;
}
.prob-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
}
.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--ink-1);
}
.prob-track {
    height: 4px;
    background: var(--bg-deep);
    border-radius: 2px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Evidence section ── */
.evidence-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-3);
    margin-bottom: 0.7rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.evidence-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--line);
}

/* ── Image info strip ── */
.img-meta {
    display: flex;
    gap: 1px;
    background: var(--line);
    border-radius: var(--radius-sm);
    overflow: hidden;
    margin-bottom: 1rem;
}
.img-meta-cell {
    flex: 1;
    background: var(--bg-float);
    padding: 8px 12px;
    text-align: center;
}
.img-meta-cell-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--ink-4);
}
.img-meta-cell-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: var(--ink-2);
    margin-top: 2px;
}

/* ── Risk Indicator ── */
.risk-bar-wrap {
    background: var(--bg-float);
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    padding: 1rem 1.2rem;
    margin-bottom: 1.2rem;
}
.risk-bar-header {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
}
.risk-bar-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--ink-3);
}
.risk-bar-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
}
.risk-track {
    height: 8px;
    background: var(--bg-deep);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
.risk-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Pipeline Steps ── */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 1.4rem 0;
}
.pipe-step {
    flex: 1;
    text-align: center;
    position: relative;
}
.pipe-step:not(:last-child)::after {
    content: '→';
    position: absolute;
    right: -8px;
    top: 50%;
    transform: translateY(-50%);
    color: var(--ink-4);
    font-size: 0.8rem;
}
.pipe-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    color: var(--ink-4);
    letter-spacing: 1px;
    margin-bottom: 4px;
}
.pipe-icon {
    font-size: 1.3rem;
    margin-bottom: 4px;
}
.pipe-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.5px;
    color: var(--azure);
    text-transform: uppercase;
    margin-bottom: 3px;
}
.pipe-desc {
    font-size: 0.68rem;
    color: var(--ink-4);
    line-height: 1.4;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-raised) !important;
    border-right: 1px solid var(--line) !important;
}
[data-testid="stSidebar"] > div { padding-top: 1rem !important; }
[data-testid="stSidebar"] * { color: var(--ink-2) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--ink-1) !important; }

.sidebar-section {
    padding: 0.8rem 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 0.6rem;
}
.sidebar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--ink-4);
    margin-bottom: 0.6rem;
}
.model-badge {
    background: var(--azure-dim);
    border: 1px solid rgba(61,142,248,0.25);
    border-radius: var(--radius-sm);
    padding: 8px 12px;
    margin-bottom: 8px;
}
.model-badge-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    font-weight: 500;
    color: var(--azure);
}
.model-badge-detail {
    font-size: 0.7rem;
    color: var(--ink-3);
    margin-top: 2px;
}

/* ── Buttons ── */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--azure) 0%, #5b6cf6 100%) !important;
    color: #fff !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 20px rgba(61,142,248,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(61,142,248,0.4) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Slider ── */
.stSlider [data-testid="stThumbValue"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: var(--bg-float) !important;
    color: var(--ink-2) !important;
    border: 1px solid var(--line-mid) !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 1px !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: none !important;
}

/* ── Divider ── */
hr { border-color: var(--line) !important; margin: 1.2rem 0 !important; }

/* ── Streamlit overrides ── */
[data-testid="stImage"] img { border-radius: var(--radius-md) !important; }
.stSpinner > div { border-top-color: var(--azure) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--ink-2); }
.stCheckbox label span { color: var(--ink-2) !important; }

/* ── Footer ── */
.footer-bar {
    margin-top: 3rem;
    padding: 1.2rem 0;
    border-top: 1px solid var(--line);
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-left {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink-4);
}
.footer-right {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink-4);
    text-align: right;
}

/* ── Warning box ── */
.warn-box {
    background: var(--amber-dim);
    border: 1px solid rgba(240,168,75,0.25);
    border-radius: var(--radius-sm);
    padding: 0.7rem 1rem;
    font-size: 0.78rem;
    color: var(--amber);
    margin-bottom: 1rem;
    font-family: 'DM Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_NAME  = "efficientnet_b0"
MODEL_PATH  = Path("outputs/best_model.pth")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 224
CLASS_NAMES = ["ai", "real"]

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    if MODEL_PATH.exists():
        state = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE).eval()
        return model, True
    model.to(DEVICE).eval()
    return model, False

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE & GRADCAM
# ═══════════════════════════════════════════════════════════════════════════════
def predict(model, pil_img):
    x = TFM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs  = F.softmax(logits, dim=1)[0].cpu()
    pred_idx = int(torch.argmax(probs))
    return CLASS_NAMES[pred_idx], float(probs[pred_idx]), probs

def run_gradcam(model, pil_img):
    if not GRADCAM_AVAILABLE:
        return None
    pil_r   = pil_img.resize((IMG_SIZE, IMG_SIZE))
    rgb_arr = np.array(pil_r).astype(np.float32) / 255.0
    inp     = TFM(pil_r).unsqueeze(0).to(DEVICE)
    cam     = GradCAM(model=model, target_layers=[model.conv_head])
    g_cam   = cam(input_tensor=inp)[0]
    overlay = show_cam_on_image(rgb_arr, g_cam, use_rgb=True)
    return Image.fromarray(overlay)

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION BAR
# ═══════════════════════════════════════════════════════════════════════════════
model, model_loaded = load_model()
model_status = "MODEL LOADED" if model_loaded else "DEMO MODE"

st.markdown(f"""
<div class="nav-bar">
    <div class="nav-logo">
        <div class="nav-logo-mark">🛡</div>
        <div>
            <div class="nav-logo-text">DeepShield</div>
            <div class="nav-logo-sub">Forensic Intelligence Platform</div>
        </div>
    </div>
    <div class="nav-status">
        <div class="status-pill">
            <div class="pulse-dot"></div>
            SYSTEM ONLINE
        </div>
        <div class="status-pill">{DEVICE.upper()}</div>
        <div class="status-pill">{"✓ " if model_loaded else "⚠ "}{model_status}</div>
        <div class="nav-version">v2.0.0</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1rem;">
        <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:2px;
                    text-transform:uppercase;color:#363d50;margin-bottom:0.8rem;">
            Analysis Configuration
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Detection Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="model-badge">
        <div class="model-badge-name">EfficientNet-B0</div>
        <div class="model-badge-detail">Transfer Learning · ImageNet Pretrained</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label" style="margin-top:1rem;">Explainability</div>', unsafe_allow_html=True)
    show_gradcam = st.checkbox("Grad-CAM Heatmap", value=True)
    show_raw     = st.checkbox("Show Raw Probabilities", value=False)

    st.markdown('<div class="sidebar-label" style="margin-top:1rem;">Confidence Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("", 0.50, 0.99, 0.72, 0.01, label_visibility="collapsed")
    st.markdown(f"""
    <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#363d50;margin-top:4px;">
        Predictions below {threshold:.0%} flagged as uncertain
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="sidebar-label">System Information</div>', unsafe_allow_html=True)
    info_items = [
        ("Architecture", "EfficientNet-B0"),
        ("Framework",    "PyTorch + timm"),
        ("Input Size",   "224 × 224 px"),
        ("Classes",      "AI / Real"),
        ("XAI Method",   "Grad-CAM"),
        ("Device",       DEVICE.upper()),
    ]
    for k, v in info_items:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;
                    border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#363d50;">{k}</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#9aa3b2;">{v}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#363d50;line-height:1.8;">
        Final Year B.Tech Project<br>
        AI & Data Science<br>
        Deepfake & Spoof Detection
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
if not model_loaded:
    st.markdown("""
    <div class="warn-box">
        ⚠ &nbsp; No trained model found at <code>outputs/best_model.pth</code> —
        running in demo mode with random predictions. Train your model first.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Threat Intelligence · Image Forensics</div>
    <div class="hero-title">Detect <em>Deepfakes</em> &<br>Spoof Attacks</div>
    <div class="hero-desc">
        Upload any image to run it through our EfficientNet-B0 forensic classifier.
        The system returns a verdict, confidence score, and Grad-CAM explainability heatmap
        highlighting manipulated regions.
    </div>
    <div class="hero-tags">
        <span class="tag">EfficientNet-B0</span>
        <span class="tag">Grad-CAM XAI</span>
        <span class="tag">Binary Classification</span>
        <span class="tag">Cybersecurity</span>
        <span class="tag">Digital Forensics</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════════════
col_upload, col_results = st.columns([1, 1.3], gap="large")

# ── UPLOAD PANEL ──────────────────────────────────────────────────────────────
with col_upload:
    st.markdown("""
    <div class="panel-header" style="background:var(--bg-raised);border:1px solid var(--line);
         border-radius:10px 10px 0 0;padding:0.9rem 1.2rem;">
        <span class="panel-label">01 — Input Image</span>
        <span style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#363d50;">
            JPG · PNG · WEBP · BMP
        </span>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "upload",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        pil_img  = Image.open(uploaded).convert("RGB")
        w, h     = pil_img.size
        size_kb  = len(uploaded.getvalue()) / 1024
        fmt      = uploaded.type.split("/")[-1].upper()

        st.markdown(f"""
        <div class="img-meta" style="margin-top:1rem;">
            <div class="img-meta-cell">
                <div class="img-meta-cell-label">Width</div>
                <div class="img-meta-cell-val">{w}px</div>
            </div>
            <div class="img-meta-cell">
                <div class="img-meta-cell-label">Height</div>
                <div class="img-meta-cell-val">{h}px</div>
            </div>
            <div class="img-meta-cell">
                <div class="img-meta-cell-label">Size</div>
                <div class="img-meta-cell-val">{size_kb:.1f}KB</div>
            </div>
            <div class="img-meta-cell">
                <div class="img-meta-cell-label">Format</div>
                <div class="img-meta-cell-val">{fmt}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.image(pil_img, use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        analyze = st.button("⬡  Run Forensic Analysis", use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem 1rem;
                    border:1.5px dashed rgba(255,255,255,0.07);
                    border-radius:0 0 10px 10px;
                    background:var(--bg-raised);">
            <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.3;">⬡</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                        color:#363d50;letter-spacing:1px;">
                DROP IMAGE HERE<br>OR CLICK TO BROWSE
            </div>
        </div>
        """, unsafe_allow_html=True)
        analyze = False

# ── RESULTS PANEL ─────────────────────────────────────────────────────────────
with col_results:
    st.markdown("""
    <div class="panel-header" style="background:var(--bg-raised);border:1px solid var(--line);
         border-radius:10px 10px 0 0;padding:0.9rem 1.2rem;">
        <span class="panel-label">02 — Forensic Analysis Report</span>
    </div>
    """, unsafe_allow_html=True)

    if uploaded and analyze:
        with st.spinner(""):
            time.sleep(0.25)
            pred_label, confidence, probs = predict(model, pil_img)

        ai_prob   = float(probs[0])
        real_prob = float(probs[1])
        uncertain = confidence < threshold

        # ── Verdict ──
        if uncertain:
            v_class, v_icon, v_head = "verdict-warn",   "⚡", "Uncertain Result"
            v_note = f"Confidence {confidence:.1%} is below the {threshold:.0%} threshold. Manual review recommended."
        elif pred_label == "ai":
            v_class, v_icon, v_head = "verdict-threat", "⚠", "Deepfake Detected"
            v_note = f"This image has been classified as AI-generated or manipulated with {confidence:.1%} confidence."
        else:
            v_class, v_icon, v_head = "verdict-safe",   "✓", "Authentic Image"
            v_note = f"This image appears to be a genuine photograph with {confidence:.1%} confidence."

        st.markdown(f"""
        <div class="verdict-wrapper {v_class}">
            <div class="verdict-icon">{v_icon}</div>
            <div class="verdict-content">
                <div class="verdict-class">
                    {'Threat Detected' if pred_label=='ai' else 'Clear' if not uncertain else 'Low Confidence'}
                </div>
                <div class="verdict-headline">{v_head}</div>
                <div class="verdict-note">{v_note}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Risk bar ──
        risk_score = ai_prob
        if risk_score < 0.35:
            risk_color, risk_label = "#2dd4a0", "LOW RISK"
        elif risk_score < 0.65:
            risk_color, risk_label = "#f0a84b", "MEDIUM RISK"
        else:
            risk_color, risk_label = "#f04b5a", "HIGH RISK"

        st.markdown(f"""
        <div class="risk-bar-wrap">
            <div class="risk-bar-header">
                <span class="risk-bar-title">Manipulation Risk Score</span>
                <span class="risk-bar-score" style="color:{risk_color};">
                    {risk_score:.1%} — {risk_label}
                </span>
            </div>
            <div class="risk-track">
                <div class="risk-fill" style="width:{risk_score*100:.1f}%;
                     background:linear-gradient(90deg, #2dd4a0, {risk_color});"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metric tiles ──
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="metric-tile-label">Confidence</div>
                <div class="metric-tile-value" style="color:{'#f04b5a' if pred_label=='ai' else '#2dd4a0'};">
                    {confidence:.1%}
                </div>
                <div class="metric-tile-sub">Model certainty</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Verdict</div>
                <div class="metric-tile-value" style="font-size:0.9rem;color:{'#f04b5a' if pred_label=='ai' else '#2dd4a0'};">
                    {'AI-GEN' if pred_label=='ai' else 'REAL'}
                </div>
                <div class="metric-tile-sub">Classification</div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Threshold</div>
                <div class="metric-tile-value" style="font-size:0.9rem;
                     color:{'#f0a84b' if uncertain else '#3d8ef8'};">
                    {'WARN' if uncertain else 'PASS'}
                </div>
                <div class="metric-tile-sub">@ {threshold:.0%} cutoff</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Probability bars ──
        st.markdown('<div class="evidence-label">Probability Distribution</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="prob-section">
            <div class="prob-item">
                <div class="prob-header">
                    <span class="prob-name">
                        <span class="prob-dot" style="background:#f04b5a;"></span>
                        AI-Generated / Deepfake
                    </span>
                    <span class="prob-pct">{ai_prob:.4f}</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{ai_prob*100:.2f}%;background:#f04b5a;"></div>
                </div>
            </div>
            <div class="prob-item">
                <div class="prob-header">
                    <span class="prob-name">
                        <span class="prob-dot" style="background:#2dd4a0;"></span>
                        Real / Authentic
                    </span>
                    <span class="prob-pct">{real_prob:.4f}</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{real_prob*100:.2f}%;background:#2dd4a0;"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if show_raw:
            st.markdown(f"""
            <div style="background:var(--bg-deep);border-radius:6px;padding:0.8rem 1rem;
                        font-family:'DM Mono',monospace;font-size:0.7rem;color:#5c6478;
                        margin-bottom:1rem;">
                <span style="color:#363d50;">raw logits →</span>
                ai: {ai_prob:.6f} &nbsp;·&nbsp; real: {real_prob:.6f}
            </div>
            """, unsafe_allow_html=True)

        # ── Grad-CAM ──
        if show_gradcam:
            st.markdown('<div class="evidence-label">Grad-CAM Explainability</div>', unsafe_allow_html=True)

            if GRADCAM_AVAILABLE:
                with st.spinner("Generating attention heatmap..."):
                    cam_img = run_gradcam(model, pil_img)

                if cam_img:
                    st.image(cam_img, use_column_width=True,
                             caption="Red = high model attention · Blue = low attention")

                    buf = io.BytesIO()
                    cam_img.save(buf, format="PNG")
                    st.download_button(
                        "↓  Export Heatmap (PNG)",
                        buf.getvalue(),
                        "deepshield_gradcam.png",
                        "image/png",
                        use_container_width=True
                    )
            else:
                st.markdown("""
                <div style="background:var(--bg-float);border:1px solid var(--line);
                            border-radius:6px;padding:1rem;text-align:center;
                            font-family:'DM Mono',monospace;font-size:0.7rem;color:#5c6478;">
                    Install grad-cam to enable heatmaps:<br>
                    <code style="color:#3d8ef8;">pip install grad-cam</code>
                </div>
                """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="background:var(--bg-raised);border:1px solid var(--line);
                    border-radius:0 0 10px 10px;text-align:center;
                    padding:5rem 2rem;">
            <div style="font-size:2.5rem;margin-bottom:1rem;opacity:0.15;">◈</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                        color:#363d50;letter-spacing:1px;line-height:2;">
                AWAITING INPUT<br>
                Upload an image and run analysis<br>to view the forensic report
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:var(--bg-raised);border:1px solid var(--line);
                    border-radius:0 0 10px 10px;text-align:center;
                    padding:5rem 2rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                        color:#363d50;letter-spacing:1px;line-height:2;">
                Click <span style="color:#3d8ef8;">Run Forensic Analysis</span><br>
                to generate the report
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="panel-label" style="margin-bottom:0.5rem;">Detection Pipeline</div>',
            unsafe_allow_html=True)

st.markdown("""
<div class="pipeline">
    <div class="pipe-step">
        <div class="pipe-num">01</div>
        <div class="pipe-icon">📥</div>
        <div class="pipe-name">Ingest</div>
        <div class="pipe-desc">Image upload<br>& validation</div>
    </div>
    <div class="pipe-step">
        <div class="pipe-num">02</div>
        <div class="pipe-icon">⚙</div>
        <div class="pipe-name">Preprocess</div>
        <div class="pipe-desc">Resize 224×224<br>ImageNet norm</div>
    </div>
    <div class="pipe-step">
        <div class="pipe-num">03</div>
        <div class="pipe-icon">🧠</div>
        <div class="pipe-name">Inference</div>
        <div class="pipe-desc">EfficientNet-B0<br>feature extraction</div>
    </div>
    <div class="pipe-step">
        <div class="pipe-num">04</div>
        <div class="pipe-icon">◉</div>
        <div class="pipe-name">Classify</div>
        <div class="pipe-desc">Softmax · binary<br>AI vs Real</div>
    </div>
    <div class="pipe-step">
        <div class="pipe-num">05</div>
        <div class="pipe-icon">🔥</div>
        <div class="pipe-name">Grad-CAM</div>
        <div class="pipe-desc">Attention map<br>generation</div>
    </div>
    <div class="pipe-step">
        <div class="pipe-num">06</div>
        <div class="pipe-icon">📋</div>
        <div class="pipe-name">Report</div>
        <div class="pipe-desc">Verdict · risk<br>score · export</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer-bar">
    <div class="footer-left">
        DeepShield Enterprise &nbsp;·&nbsp; Deepfake & Spoof Attack Detection in Cyber Threats<br>
        Final Year B.Tech — Artificial Intelligence & Data Science
    </div>
    <div class="footer-right">
        EfficientNet-B0 · PyTorch · Grad-CAM<br>
        Device: {DEVICE.upper()} &nbsp;·&nbsp; v2.0.0
    </div>
</div>
""", unsafe_allow_html=True)