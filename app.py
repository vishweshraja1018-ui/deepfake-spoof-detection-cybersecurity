"""
DeepShield Enterprise v3.0
3-Stage Pipeline: Spoof Gate → Deepfake Detection → Grad-CAM
Final Year B.Tech — Artificial Intelligence & Data Science
"""

import io
import time
import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import timm
from pathlib import Path
from PIL import Image
from torchvision import transforms

# ── Optional imports ──────────────────────────────────────────────────────────
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
# DESIGN SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Instrument+Serif:ital@0;1&family=Geist:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --ink-1:      #f0f2f5;
    --ink-2:      #9aa3b2;
    --ink-3:      #5c6478;
    --ink-4:      #363d50;
    --bg-deep:    #080b11;
    --bg-base:    #0d1017;
    --bg-raised:  #111620;
    --bg-float:   #161c2a;
    --line:       rgba(255,255,255,0.06);
    --line-mid:   rgba(255,255,255,0.10);
    --azure:      #3d8ef8;
    --azure-dim:  rgba(61,142,248,0.15);
    --azure-glow: rgba(61,142,248,0.35);
    --crimson:    #f04b5a;
    --crim-dim:   rgba(240,75,90,0.15);
    --emerald:    #2dd4a0;
    --emer-dim:   rgba(45,212,160,0.15);
    --amber:      #f0a84b;
    --amber-dim:  rgba(240,168,75,0.12);
    --violet:     #a78bfa;
    --violet-dim: rgba(167,139,250,0.15);
    --radius-sm:  6px;
    --radius-md:  10px;
    --radius-lg:  16px;
}

html, body, [class*="css"], .stApp {
    background-color: var(--bg-base) !important;
    color: var(--ink-1);
    font-family: 'Geist', sans-serif;
    font-size: 14px;
    line-height: 1.6;
}

/* Scanline overlay */
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(255,255,255,0.011) 2px, rgba(255,255,255,0.011) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Nav Bar ── */
.nav-bar {
    background: rgba(13,16,23,0.97);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--line-mid);
    padding: 0 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
    height: 60px;
    margin: -1rem -1rem 0 -1rem;
}
.nav-logo { display: flex; align-items: center; gap: 10px; }
.nav-logo-mark {
    width: 32px; height: 32px;
    background: linear-gradient(135deg, var(--azure) 0%, #6b5cf6 100%);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    box-shadow: 0 0 20px var(--azure-glow);
}
.nav-logo-text { font-weight: 700; font-size: 1rem; color: var(--ink-1); letter-spacing: -0.3px; }
.nav-logo-sub  { font-family: 'DM Mono', monospace; font-size: 0.58rem; color: var(--ink-4); letter-spacing: 1.5px; }
.nav-right     { display: flex; align-items: center; gap: 12px; }
.status-pill {
    display: flex; align-items: center; gap: 6px;
    background: var(--bg-float); border: 1px solid var(--line);
    border-radius: 20px; padding: 4px 12px;
    font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--ink-3);
}
.pulse { width:7px; height:7px; border-radius:50%; background: var(--emerald);
         box-shadow: 0 0 8px var(--emerald); animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

/* ── Stage Tracker ── */
.stage-tracker {
    display: flex; align-items: center; gap: 0;
    background: var(--bg-raised); border: 1px solid var(--line);
    border-radius: var(--radius-lg); padding: 1.2rem 1.6rem;
    margin: 1.5rem 0;
    overflow: hidden;
    position: relative;
}
.stage-tracker::before {
    content: '';
    position: absolute; bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--azure), transparent);
    opacity: 0.3;
}
.stage-item {
    flex: 1; display: flex; align-items: center; gap: 10px;
    position: relative;
}
.stage-item:not(:last-child)::after {
    content: ''; position: absolute; right: 0;
    top: 50%; transform: translateY(-50%);
    width: 1px; height: 30px; background: var(--line-mid);
}
.stage-num-wrap {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; position: relative;
}
.stage-num {
    font-family: 'DM Mono', monospace; font-size: 0.75rem; font-weight: 500;
}
.stage-text { flex: 1; padding: 0 12px; }
.stage-label {
    font-family: 'DM Mono', monospace; font-size: 0.58rem;
    letter-spacing: 1.5px; text-transform: uppercase; color: var(--ink-4);
    margin-bottom: 2px;
}
.stage-title { font-size: 0.82rem; font-weight: 600; }

/* Stage states */
.stage-pending .stage-num-wrap { background: var(--bg-float); border: 1px solid var(--line); }
.stage-pending .stage-num  { color: var(--ink-4); }
.stage-pending .stage-title { color: var(--ink-4); }

.stage-active  .stage-num-wrap { background: var(--azure-dim); border: 1px solid rgba(61,142,248,0.4);
    box-shadow: 0 0 15px var(--azure-glow); animation: stagePulse 1.5s infinite; }
.stage-active  .stage-num  { color: var(--azure); }
.stage-active  .stage-title { color: var(--azure); }
@keyframes stagePulse { 0%,100%{box-shadow:0 0 12px var(--azure-glow)} 50%{box-shadow:0 0 24px var(--azure-glow)} }

.stage-pass    .stage-num-wrap { background: var(--emer-dim); border: 1px solid rgba(45,212,160,0.4); }
.stage-pass    .stage-num  { color: var(--emerald); }
.stage-pass    .stage-title { color: var(--emerald); }

.stage-block   .stage-num-wrap { background: var(--crim-dim); border: 1px solid rgba(240,75,90,0.4); }
.stage-block   .stage-num  { color: var(--crimson); }
.stage-block   .stage-title { color: var(--crimson); }

.stage-warn    .stage-num-wrap { background: var(--amber-dim); border: 1px solid rgba(240,168,75,0.4); }
.stage-warn    .stage-num  { color: var(--amber); }
.stage-warn    .stage-title { color: var(--amber); }

/* ── Spoof Signals Grid ── */
.signal-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 1rem 0; }
.signal-cell {
    background: var(--bg-float); border: 1px solid var(--line);
    border-radius: var(--radius-sm); padding: 0.9rem;
    position: relative; overflow: hidden;
}
.signal-cell::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.signal-cell.ok::after    { background: var(--emerald); }
.signal-cell.warn::after  { background: var(--amber); }
.signal-cell.alert::after { background: var(--crimson); }

.signal-name {
    font-family: 'DM Mono', monospace; font-size: 0.58rem;
    letter-spacing: 1px; text-transform: uppercase; color: var(--ink-4); margin-bottom: 6px;
}
.signal-val {
    font-family: 'DM Mono', monospace; font-size: 1.1rem; font-weight: 500;
}
.signal-cell.ok    .signal-val { color: var(--emerald); }
.signal-cell.warn  .signal-val { color: var(--amber); }
.signal-cell.alert .signal-val { color: var(--crimson); }
.signal-note { font-size: 0.68rem; color: var(--ink-4); margin-top: 3px; }

/* ── Verdict Banner ── */
.verdict-banner {
    border-radius: var(--radius-md); padding: 1.4rem 1.6rem;
    display: flex; align-items: flex-start; gap: 1rem;
    animation: fadeUp 0.4s ease;
}
@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.verdict-banner.blocked  { background: var(--crim-dim);   border: 1px solid rgba(240,75,90,0.35); }
.verdict-banner.clear    { background: var(--emer-dim);   border: 1px solid rgba(45,212,160,0.35); }
.verdict-banner.warn     { background: var(--amber-dim);  border: 1px solid rgba(240,168,75,0.35); }
.verdict-banner.deepfake { background: var(--crim-dim);   border: 1px solid rgba(240,75,90,0.35); }
.verdict-banner.real     { background: var(--emer-dim);   border: 1px solid rgba(45,212,160,0.35); }

.vb-icon {
    width: 48px; height: 48px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; flex-shrink: 0;
}
.blocked  .vb-icon, .deepfake .vb-icon { background: rgba(240,75,90,0.2); }
.clear    .vb-icon, .real     .vb-icon { background: rgba(45,212,160,0.2); }
.warn     .vb-icon                     { background: rgba(240,168,75,0.2); }
.vb-tag {
    font-family: 'DM Mono', monospace; font-size: 0.58rem;
    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 3px;
}
.blocked  .vb-tag, .deepfake .vb-tag { color: var(--crimson); }
.clear    .vb-tag, .real     .vb-tag { color: var(--emerald); }
.warn     .vb-tag                    { color: var(--amber); }
.vb-title {
    font-family: 'Instrument Serif', serif; font-size: 1.5rem;
    font-weight: 400; color: var(--ink-1); margin-bottom: 4px;
}
.vb-note { font-size: 0.78rem; color: var(--ink-3); line-height: 1.5; }

/* ── Probability Bars ── */
.prob-row-wrap { margin-bottom: 10px; }
.prob-row-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px; }
.prob-row-name { font-size: 0.78rem; color: var(--ink-2); display: flex; align-items: center; gap: 6px; }
.prob-dot      { width: 6px; height: 6px; border-radius: 50%; }
.prob-row-pct  { font-family: 'DM Mono', monospace; font-size: 0.73rem; color: var(--ink-1); }
.prob-track    { height: 4px; background: var(--bg-deep); border-radius: 2px; overflow: hidden; }
.prob-fill     { height: 100%; border-radius: 2px; transition: width 0.8s cubic-bezier(0.4,0,0.2,1); }

/* ── Metric Tiles ── */
.metric-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 1rem 0; }
.metric-tile {
    background: var(--bg-float); border: 1px solid var(--line);
    border-radius: var(--radius-sm); padding: 0.9rem 1rem;
}
.mt-label { font-family: 'DM Mono', monospace; font-size: 0.56rem; letter-spacing: 1.5px;
            text-transform: uppercase; color: var(--ink-4); margin-bottom: 5px; }
.mt-val   { font-family: 'DM Mono', monospace; font-size: 1.25rem; font-weight: 500; }
.mt-sub   { font-size: 0.68rem; color: var(--ink-4); margin-top: 2px; }

/* ── Risk bar ── */
.risk-wrap {
    background: var(--bg-float); border: 1px solid var(--line);
    border-radius: var(--radius-sm); padding: 0.9rem 1.1rem; margin: 1rem 0;
}
.risk-head { display: flex; justify-content: space-between; margin-bottom: 8px; }
.risk-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 1.5px;
              text-transform: uppercase; color: var(--ink-3); }
.risk-score { font-family: 'DM Mono', monospace; font-size: 0.73rem; font-weight: 500; }
.risk-track { height: 8px; background: var(--bg-deep); border-radius: 4px; overflow: hidden; }
.risk-fill  { height: 100%; border-radius: 4px; }

/* ── Section label ── */
.sec-label {
    font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 2px;
    text-transform: uppercase; color: var(--ink-4); margin-bottom: 0.7rem;
    display: flex; align-items: center; gap: 8px;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: var(--line); }

/* ── Divider / Separator ── */
hr { border-color: var(--line) !important; margin: 1.2rem 0 !important; }

/* ── Blocked gate screen ── */
.gate-blocked {
    background: linear-gradient(135deg, rgba(240,75,90,0.08), rgba(240,75,90,0.03));
    border: 1px solid rgba(240,75,90,0.3);
    border-radius: var(--radius-lg);
    padding: 3rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.gate-blocked-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.gate-blocked-title {
    font-family: 'Instrument Serif', serif; font-size: 2rem;
    color: var(--crimson); margin-bottom: 0.5rem;
}
.gate-blocked-sub { font-size: 0.85rem; color: var(--ink-3); max-width: 420px; margin: 0 auto 1.5rem; }
.gate-blocked-reasons {
    display: inline-flex; flex-direction: column; gap: 6px;
    text-align: left; background: rgba(240,75,90,0.08);
    border: 1px solid rgba(240,75,90,0.2);
    border-radius: var(--radius-sm); padding: 0.8rem 1.2rem;
    margin-top: 0.5rem;
}
.gate-reason {
    font-family: 'DM Mono', monospace; font-size: 0.68rem; color: var(--crimson);
    display: flex; align-items: center; gap: 6px;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] > div {
    background: var(--bg-float) !important;
    border: 1.5px dashed var(--line-mid) !important;
    border-radius: var(--radius-md) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

/* ── Buttons ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, var(--azure) 0%, #5b6cf6 100%) !important;
    color: #fff !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important; font-weight: 500 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
    border: none !important; border-radius: var(--radius-sm) !important;
    padding: 0.65rem 1.6rem !important;
    box-shadow: 0 4px 20px rgba(61,142,248,0.22) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 28px rgba(61,142,248,0.4) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"]          { background: var(--bg-raised) !important; border-right: 1px solid var(--line) !important; }
[data-testid="stSidebar"] *        { color: var(--ink-2) !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3       { color: var(--ink-1) !important; }

/* ── Misc ── */
[data-testid="stImage"] img        { border-radius: var(--radius-md) !important; }
.stSpinner > div                   { border-top-color: var(--azure) !important; }
.stDownloadButton > button {
    background: var(--bg-float) !important; color: var(--ink-2) !important;
    border: 1px solid var(--line-mid) !important;
    font-family: 'DM Mono', monospace !important; font-size: 0.66rem !important;
    letter-spacing: 1px !important; border-radius: var(--radius-sm) !important;
    box-shadow: none !important;
}

/* ── Footer ── */
.footer {
    margin-top: 3rem; padding: 1.2rem 0; border-top: 1px solid var(--line);
    display: flex; justify-content: space-between; align-items: center;
}
.footer-l, .footer-r { font-family: 'DM Mono', monospace; font-size: 0.6rem; color: var(--ink-4); }
.footer-r { text-align: right; }

/* ── img meta strip ── */
.img-meta { display: flex; gap: 1px; background: var(--line); border-radius: var(--radius-sm); overflow: hidden; margin-bottom: 1rem; }
.img-meta-cell { flex: 1; background: var(--bg-float); padding: 7px 10px; text-align: center; }
.imc-l { font-family: 'DM Mono', monospace; font-size: 0.53rem; letter-spacing: 1px; text-transform: uppercase; color: var(--ink-4); }
.imc-v { font-family: 'DM Mono', monospace; font-size: 0.78rem; color: var(--ink-2); margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = Path("outputs/best_model.pth")
MODEL_NAME  = "efficientnet_b0"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 224
CLASS_NAMES = ["ai", "real"]
SPOOF_THRESHOLD    = 0.45   # from your spoof_detector.py

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

# ═══════════════════════════════════════════════════════════════════════════════
# SPOOF DETECTOR  (your spoof_detector.py logic, embedded)
# ═══════════════════════════════════════════════════════════════════════════════
def _blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _glare_score(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v   = hsv[:, :, 2]
    return float((v > 245).astype(np.uint8).mean())

def _moire_score(gray):
    f      = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag    = np.log(np.abs(fshift) + 1.0)
    h, w   = mag.shape
    cy, cx = h // 2, w // 2
    r      = int(min(h, w) * 0.08)
    low    = mag[cy-r:cy+r, cx-r:cx+r]
    total  = float(mag.mean())
    low_e  = float(low.mean())
    return float(max(total - low_e, 0.0) / (total + 1e-6))

def _border_score(gray):
    edges = cv2.Canny(gray, 80, 160)
    edges = (edges > 0).astype(np.float32)
    h, w  = edges.shape
    m     = int(min(h, w) * 0.06)
    top    = float(edges[:m, :].mean())
    bottom = float(edges[h-m:, :].mean())
    left   = float(edges[:, :m].mean())
    right  = float(edges[:, w-m:].mean())
    return float((top + bottom + left + right) / 4.0)

def detect_spoof(pil_img):
    """Run spoof detection. Returns full result dict."""
    img_bgr = cv2.cvtColor(np.array(pil_img.resize((512, 512))), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur   = _blur_score(gray)
    glare  = _glare_score(img_bgr)
    moire  = _moire_score(gray)
    border = _border_score(gray)

    reasons     = []
    spoof_score = 0.0

    if blur < 60:
        reasons.append(f"High blur detected (score={blur:.1f})")
        spoof_score += 0.30
    elif blur < 120:
        reasons.append(f"Medium blur detected (score={blur:.1f})")
        spoof_score += 0.15

    if glare > 0.015:
        reasons.append(f"Screen glare detected (ratio={glare:.3f})")
        spoof_score += 0.20

    if moire > 0.45:
        reasons.append(f"Moiré patterns detected (score={moire:.2f})")
        spoof_score += 0.25

    if border > 0.10:
        reasons.append(f"Strong border edges (score={border:.3f})")
        spoof_score += 0.25

    spoof_score = float(min(max(spoof_score, 0.0), 1.0))
    is_spoof    = spoof_score >= SPOOF_THRESHOLD

    return {
        "is_spoof":    bool(is_spoof),
        "spoof_score": spoof_score,
        "reasons":     reasons if reasons else ["No strong spoof signals detected"],
        "signals": {
            "Blur Score":   (blur,   "blur"),
            "Glare Ratio":  (glare,  "glare"),
            "Moiré Score":  (moire,  "moire"),
            "Border Score": (border, "border"),
        },
    }

# ═══════════════════════════════════════════════════════════════════════════════
# DEEPFAKE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_deepfake_model():
    m = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    if MODEL_PATH.exists():
        m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        m.to(DEVICE).eval()
        return m, True
    m.to(DEVICE).eval()
    return m, False

def run_deepfake(model, pil_img):
    x = TFM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu()
    idx = int(torch.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def run_gradcam(model, pil_img):
    if not GRADCAM_AVAILABLE:
        return None
    pil_r   = pil_img.resize((IMG_SIZE, IMG_SIZE))
    rgb_arr = np.array(pil_r).astype(np.float32) / 255.0
    inp     = TFM(pil_r).unsqueeze(0).to(DEVICE)
    cam     = GradCAM(model=model, target_layers=[model.conv_head])
    overlay = show_cam_on_image(rgb_arr, cam(input_tensor=inp)[0], use_rgb=True)
    return Image.fromarray(overlay)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ═══════════════════════════════════════════════════════════════════════════════
deepfake_model, model_loaded = load_deepfake_model()

# ═══════════════════════════════════════════════════════════════════════════════
# NAVIGATION BAR
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="nav-bar">
    <div class="nav-logo">
        <div class="nav-logo-mark">🛡</div>
        <div>
            <div class="nav-logo-text">DeepShield Enterprise</div>
            <div class="nav-logo-sub">Forensic Intelligence Platform · v3.0</div>
        </div>
    </div>
    <div class="nav-right">
        <div class="status-pill"><div class="pulse"></div>SYSTEM ONLINE</div>
        <div class="status-pill">{DEVICE.upper()}</div>
        <div class="status-pill">{"✓ MODEL READY" if model_loaded else "⚠ DEMO MODE"}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:0.8rem 0 0.5rem;">
        <div style="font-family:'DM Mono',monospace;font-size:1rem;font-weight:600;
                    color:#f0f2f5;letter-spacing:-0.3px;">Configuration</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("**Spoof Detection**")
    spoof_thresh = st.slider("Spoof Threshold", 0.20, 0.90, float(SPOOF_THRESHOLD), 0.05)
    bypass_spoof = st.checkbox("Bypass Spoof Gate (debug)", value=False)

    st.markdown("**Deepfake Detection**")
    df_thresh    = st.slider("Confidence Threshold", 0.50, 0.99, 0.72, 0.01)
    show_gradcam = st.checkbox("Grad-CAM Heatmap", value=True)
    show_raw     = st.checkbox("Raw Probability Values", value=False)

    st.markdown("---")
    st.markdown("**System Info**")
    rows = [("Architecture","EfficientNet-B0"),("Framework","PyTorch + timm"),
            ("Spoof Engine","OpenCV · FFT · Canny"),("XAI","Grad-CAM"),
            ("Device",DEVICE.upper()),("Model",("✓ Loaded" if model_loaded else "⚠ Demo"))]
    for k,v in rows:
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:5px 0;
                    border-bottom:1px solid rgba(255,255,255,0.04);">
            <span style="font-family:'DM Mono',monospace;font-size:0.63rem;color:#363d50;">{k}</span>
            <span style="font-family:'DM Mono',monospace;font-size:0.63rem;color:#9aa3b2;">{v}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                    color:#363d50;line-height:1.9;">
        Final Year B.Tech<br>AI & Data Science<br>
        Deepfake &amp; Spoof Detection<br>in Cyber Threats
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "spoof_result"  not in st.session_state:
    st.session_state.spoof_result  = None
if "df_result"     not in st.session_state:
    st.session_state.df_result     = None

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding:2.5rem 0 1.5rem;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1.5rem;">
    <div style="font-family:'DM Mono',monospace;font-size:0.62rem;letter-spacing:2.5px;
                text-transform:uppercase;color:#3d8ef8;margin-bottom:0.6rem;">
        3-Stage Forensic Pipeline
    </div>
    <div style="font-family:'Instrument Serif',serif;font-size:2.4rem;
                font-weight:400;color:#f0f2f5;letter-spacing:-0.5px;line-height:1.15;margin-bottom:0.5rem;">
        Detect <em style="color:#3d8ef8;">Spoofs</em> &<br>
        <em style="color:#f04b5a;">Deepfakes</em> in One Flow
    </div>
    <div style="font-size:0.85rem;color:#5c6478;max-width:520px;line-height:1.7;">
        Every image first passes through the <strong style="color:#9aa3b2;">Spoof Gate</strong>
        — checking for print attacks, replay attacks, and screen captures.
        Only clean images proceed to <strong style="color:#9aa3b2;">Deepfake Analysis</strong>
        and <strong style="color:#9aa3b2;">Grad-CAM</strong> explainability.
    </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE TRACKER  (dynamic)
# ═══════════════════════════════════════════════════════════════════════════════
def stage_tracker(spoof_done, spoof_blocked, df_done, cam_done):
    def stage_class(cond_active, cond_pass, cond_block=False, cond_warn=False):
        if cond_block: return "stage-block"
        if cond_warn:  return "stage-warn"
        if cond_pass:  return "stage-pass"
        if cond_active:return "stage-active"
        return "stage-pending"

    s1 = stage_class(not spoof_done, spoof_done and not spoof_blocked, cond_block=spoof_done and spoof_blocked)
    s2 = stage_class(spoof_done and not spoof_blocked and not df_done,
                     df_done, cond_block=spoof_blocked)
    s3 = stage_class(df_done and not cam_done, cam_done, cond_block=spoof_blocked)

    icons = {
        "stage-pending":"○","stage-active":"◉","stage-pass":"✓",
        "stage-block":"✗","stage-warn":"⚡"
    }

    stages = [
        (s1, "01", "Spoof Gate",         "Print · Replay · Screen"),
        (s2, "02", "Deepfake Analysis",  "EfficientNet-B0 CNN"),
        (s3, "03", "Grad-CAM Report",    "Explainability · Export"),
    ]

    html = '<div class="stage-tracker">'
    for i,(sc,num,title,desc) in enumerate(stages):
        sep = "" if i == len(stages)-1 else ""
        html += f"""
        <div class="stage-item {sc}">
            <div class="stage-num-wrap">
                <span class="stage-num">{icons.get(sc,'○')}</span>
            </div>
            <div class="stage-text">
                <div class="stage-label">{num}</div>
                <div class="stage-title">{title}</div>
                <div style="font-size:0.67rem;color:var(--ink-4);margin-top:1px;">{desc}</div>
            </div>
        </div>{sep}"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1.35], gap="large")

# ── LEFT: Upload ──────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:2px;
                text-transform:uppercase;color:#363d50;margin-bottom:0.6rem;">
        Input Image
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("img", type=["jpg","jpeg","png","webp","bmp"],
                                label_visibility="collapsed")

    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        w, h    = pil_img.size
        kb      = len(uploaded.getvalue()) / 1024
        fmt     = uploaded.type.split("/")[-1].upper()

        st.markdown(f"""
        <div class="img-meta">
            <div class="img-meta-cell"><div class="imc-l">Width</div><div class="imc-v">{w}px</div></div>
            <div class="img-meta-cell"><div class="imc-l">Height</div><div class="imc-v">{h}px</div></div>
            <div class="img-meta-cell"><div class="imc-l">Size</div><div class="imc-v">{kb:.1f}KB</div></div>
            <div class="img-meta-cell"><div class="imc-l">Format</div><div class="imc-v">{fmt}</div></div>
        </div>""", unsafe_allow_html=True)

        st.image(pil_img, use_column_width=True)
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("⬡  Run Full Pipeline Analysis", use_container_width=True)

        if run_btn:
            st.session_state.analysis_done = False
            st.session_state.spoof_result  = None
            st.session_state.df_result     = None

            # Stage 1 – Spoof
            with st.spinner("Stage 1 — Running Spoof Gate..."):
                time.sleep(0.2)
                sr = detect_spoof(pil_img)
                sr["is_spoof"] = sr["is_spoof"] and not bypass_spoof
                st.session_state.spoof_result = sr

            # Stage 2 – Deepfake (only if not spoof)
            if not sr["is_spoof"]:
                with st.spinner("Stage 2 — Running Deepfake Analysis..."):
                    time.sleep(0.2)
                    label, conf, probs = run_deepfake(deepfake_model, pil_img)
                    st.session_state.df_result = {
                        "label": label, "conf": conf, "probs": probs
                    }

            st.session_state.analysis_done = True

    else:
        st.markdown("""
        <div style="text-align:center;padding:5rem 1rem;
                    border:1.5px dashed rgba(255,255,255,0.07);
                    border-radius:10px;background:var(--bg-raised);">
            <div style="font-size:2.5rem;opacity:0.2;margin-bottom:1rem;">⬡</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                        color:#363d50;letter-spacing:1px;">
                DROP IMAGE HERE<br>OR CLICK TO BROWSE
            </div>
        </div>""", unsafe_allow_html=True)
        run_btn = False

# ── RIGHT: Results ────────────────────────────────────────────────────────────
with col_right:
    sr  = st.session_state.spoof_result
    dfr = st.session_state.df_result

    spoof_done   = sr  is not None
    spoof_blocked= spoof_done and sr["is_spoof"]
    df_done      = dfr is not None
    cam_done     = df_done and show_gradcam

    # Stage tracker
    stage_tracker(spoof_done, spoof_blocked, df_done, cam_done)

    if not uploaded:
        st.markdown("""
        <div style="background:var(--bg-raised);border:1px solid var(--line);
                    border-radius:var(--radius-lg);text-align:center;padding:5rem 2rem;">
            <div style="font-size:2.5rem;opacity:0.12;margin-bottom:1rem;">◈</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                        color:#363d50;letter-spacing:1px;line-height:2;">
                AWAITING INPUT<br>Upload an image to begin
            </div>
        </div>""", unsafe_allow_html=True)

    elif not st.session_state.analysis_done:
        st.markdown("""
        <div style="background:var(--bg-raised);border:1px solid var(--line);
                    border-radius:var(--radius-lg);text-align:center;padding:4rem 2rem;">
            <div style="font-family:'DM Mono',monospace;font-size:0.68rem;
                        color:#363d50;letter-spacing:1px;line-height:2;">
                Click <span style="color:#3d8ef8;">Run Full Pipeline Analysis</span><br>
                to execute all 3 stages
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        # ══ STAGE 1 RESULTS ══════════════════════════════════════════════════
        st.markdown('<div class="sec-label">Stage 01 — Spoof Gate</div>', unsafe_allow_html=True)

        spoof_score = sr["spoof_score"]
        sc_pct      = f"{spoof_score:.1%}"

        if spoof_blocked:
            # BLOCKED
            st.markdown(f"""
            <div class="verdict-banner blocked">
                <div class="vb-icon">🚫</div>
                <div>
                    <div class="vb-tag">Access Blocked · Spoof Detected</div>
                    <div class="vb-title">Spoof Attack Identified</div>
                    <div class="vb-note">
                        This image has been flagged as a potential spoof attack
                        (score {sc_pct} ≥ threshold {spoof_thresh:.0%}).
                        Deepfake analysis blocked to prevent false analysis on manipulated input.
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-banner clear">
                <div class="vb-icon">✓</div>
                <div>
                    <div class="vb-tag">Gate Passed · No Spoof Detected</div>
                    <div class="vb-title">Image Cleared</div>
                    <div class="vb-note">
                        No significant spoof signals found (score {sc_pct} &lt; threshold {spoof_thresh:.0%}).
                        Proceeding to deepfake analysis.
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Spoof signal grid
        signals = sr["signals"]

        def signal_state(key, val):
            if key == "blur":
                if val < 60:   return "alert", f"{val:.0f}"
                if val < 120:  return "warn",  f"{val:.0f}"
                return "ok",   f"{val:.0f}"
            if key == "glare":
                if val > 0.015: return "alert", f"{val:.3f}"
                return "ok",    f"{val:.3f}"
            if key == "moire":
                if val > 0.45: return "alert", f"{val:.2f}"
                return "ok",   f"{val:.2f}"
            if key == "border":
                if val > 0.10: return "alert", f"{val:.3f}"
                return "ok",   f"{val:.3f}"
            return "ok", f"{val:.3f}"

        cells_html = ""
        for name, (val, key) in signals.items():
            state, display = signal_state(key, val)
            note = {"blur":"Sharp=safe·Blurry=risk","glare":"Overexposed=screen",
                    "moire":"HF patterns=recapture","border":"Edges=physical print"}[key]
            cells_html += f"""
            <div class="signal-cell {state}">
                <div class="signal-name">{name}</div>
                <div class="signal-val">{display}</div>
                <div class="signal-note">{note}</div>
            </div>"""

        st.markdown(f'<div class="signal-grid">{cells_html}</div>', unsafe_allow_html=True)

        # Reasons
        if sr["reasons"]:
            reasons_html = "".join(
                f'<div class="gate-reason">{"⚠" if spoof_blocked else "·"} {r}</div>'
                for r in sr["reasons"]
            )
            color = "rgba(240,75,90,0.08)" if spoof_blocked else "rgba(45,212,160,0.06)"
            bcolor= "rgba(240,75,90,0.2)"  if spoof_blocked else "rgba(45,212,160,0.2)"
            tcolor= "var(--crimson)"         if spoof_blocked else "var(--emerald)"
            st.markdown(f"""
            <div style="background:{color};border:1px solid {bcolor};
                        border-radius:var(--radius-sm);padding:0.8rem 1rem;margin-top:0.5rem;">
                <div style="font-family:'DM Mono',monospace;font-size:0.58rem;
                            letter-spacing:1.5px;text-transform:uppercase;
                            color:{tcolor};margin-bottom:6px;">Detection Signals</div>
                {reasons_html}
            </div>""", unsafe_allow_html=True)

        # ══ BLOCKED GATE — stop here ═════════════════════════════════════════
        if spoof_blocked:
            st.markdown(f"""
            <div class="gate-blocked">
                <div class="gate-blocked-icon">🚫</div>
                <div class="gate-blocked-title">Pipeline Blocked</div>
                <div class="gate-blocked-sub">
                    Deepfake analysis is disabled for images flagged as spoof attacks.
                    This prevents false-negative results from manipulated or recaptured photos.
                    Spoof Score: <strong style="color:var(--crimson);">{sc_pct}</strong>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#5c6478;">
                    ↑ Adjust Spoof Threshold in sidebar to override, or use bypass (debug only)
                </div>
            </div>""", unsafe_allow_html=True)

        # ══ STAGE 2 — DEEPFAKE RESULTS ═══════════════════════════════════════
        if df_done:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Stage 02 — Deepfake Analysis</div>', unsafe_allow_html=True)

            label   = dfr["label"]
            conf    = dfr["conf"]
            probs   = dfr["probs"]
            ai_p    = float(probs[0])
            real_p  = float(probs[1])
            uncert  = conf < df_thresh

            if uncert:
                v_type, v_icon, v_tag, v_title = "warn","⚡","Uncertain Result","Low Confidence"
                v_note = f"Confidence {conf:.1%} is below your {df_thresh:.0%} threshold. Manual review recommended."
            elif label == "ai":
                v_type, v_icon, v_tag, v_title = "deepfake","⚠","Deepfake Detected","AI-Generated Image"
                v_note = f"EfficientNet-B0 classified this as AI-generated with {conf:.1%} confidence."
            else:
                v_type, v_icon, v_tag, v_title = "real","✓","Authentic","Real Photograph"
                v_note = f"EfficientNet-B0 classified this as a genuine photograph with {conf:.1%} confidence."

            st.markdown(f"""
            <div class="verdict-banner {v_type}">
                <div class="vb-icon">{v_icon}</div>
                <div>
                    <div class="vb-tag">{v_tag}</div>
                    <div class="vb-title">{v_title}</div>
                    <div class="vb-note">{v_note}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Risk bar
            risk = ai_p
            rc   = "#f04b5a" if risk > 0.65 else ("#f0a84b" if risk > 0.35 else "#2dd4a0")
            rl   = "HIGH RISK" if risk > 0.65 else ("MEDIUM RISK" if risk > 0.35 else "LOW RISK")
            st.markdown(f"""
            <div class="risk-wrap">
                <div class="risk-head">
                    <span class="risk-label">Manipulation Risk Score</span>
                    <span class="risk-score" style="color:{rc};">{risk:.1%} — {rl}</span>
                </div>
                <div class="risk-track">
                    <div class="risk-fill" style="width:{risk*100:.1f}%;
                         background:linear-gradient(90deg,#2dd4a0,{rc});"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Metric tiles
            mc = "#f04b5a" if label=="ai" else "#2dd4a0"
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-tile">
                    <div class="mt-label">Confidence</div>
                    <div class="mt-val" style="color:{mc};">{conf:.1%}</div>
                    <div class="mt-sub">Model certainty</div>
                </div>
                <div class="metric-tile">
                    <div class="mt-label">Verdict</div>
                    <div class="mt-val" style="color:{mc};font-size:0.9rem;">
                        {"AI-GEN" if label=="ai" else "REAL"}
                    </div>
                    <div class="mt-sub">Classification</div>
                </div>
                <div class="metric-tile">
                    <div class="mt-label">Gate</div>
                    <div class="mt-val" style="color:#2dd4a0;font-size:0.9rem;">PASS</div>
                    <div class="mt-sub">Spoof cleared</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="sec-label">Probability Distribution</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div>
                <div class="prob-row-wrap">
                    <div class="prob-row-head">
                        <span class="prob-row-name">
                            <span class="prob-dot" style="background:#f04b5a;"></span>
                            AI-Generated / Deepfake
                        </span>
                        <span class="prob-row-pct">{ai_p:.4f}</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{ai_p*100:.2f}%;background:#f04b5a;"></div>
                    </div>
                </div>
                <div class="prob-row-wrap">
                    <div class="prob-row-head">
                        <span class="prob-row-name">
                            <span class="prob-dot" style="background:#2dd4a0;"></span>
                            Real / Authentic
                        </span>
                        <span class="prob-row-pct">{real_p:.4f}</span>
                    </div>
                    <div class="prob-track">
                        <div class="prob-fill" style="width:{real_p*100:.2f}%;background:#2dd4a0;"></div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            if show_raw:
                st.markdown(f"""
                <div style="background:var(--bg-deep);border-radius:6px;padding:0.8rem 1rem;
                            font-family:'DM Mono',monospace;font-size:0.68rem;color:#5c6478;margin-top:8px;">
                    raw → ai: {ai_p:.6f} &nbsp;·&nbsp; real: {real_p:.6f}
                </div>""", unsafe_allow_html=True)

            # ── Stage 3: Grad-CAM ─────────────────────────────────────────────
            if show_gradcam:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="sec-label">Stage 03 — Grad-CAM Explainability</div>',
                            unsafe_allow_html=True)
                if GRADCAM_AVAILABLE:
                    with st.spinner("Generating attention heatmap..."):
                        cam_img = run_gradcam(deepfake_model, pil_img)
                    if cam_img:
                        st.image(cam_img, use_column_width=True,
                                 caption="🔴 High attention   ·   🔵 Low attention")
                        buf = io.BytesIO()
                        cam_img.save(buf, format="PNG")
                        st.download_button("↓ Export Grad-CAM (PNG)", buf.getvalue(),
                                           "deepshield_gradcam.png", "image/png",
                                           use_container_width=True)
                else:
                    st.markdown("""
                    <div style="background:var(--bg-float);border:1px solid var(--line);
                                border-radius:6px;padding:1rem;text-align:center;
                                font-family:'DM Mono',monospace;font-size:0.68rem;color:#5c6478;">
                        Enable heatmaps: <code style="color:#3d8ef8;">pip install grad-cam</code>
                    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="footer">
    <div class="footer-l">
        DeepShield Enterprise · Deepfake &amp; Spoof Attack Detection in Cyber Threats<br>
        Final Year B.Tech — Artificial Intelligence &amp; Data Science
    </div>
    <div class="footer-r">
        EfficientNet-B0 · PyTorch · OpenCV · Grad-CAM<br>
        Device: {DEVICE.upper()} &nbsp;·&nbsp; v3.0.0
    </div>
</div>""", unsafe_allow_html=True)