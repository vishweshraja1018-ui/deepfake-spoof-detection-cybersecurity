"""
DeepShield Enterprise v3.0
Auto-run Pipeline: Spoof Gate (silent) → Deepfake Detection → Grad-CAM
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

# ── Optional Grad-CAM ─────────────────────────────────────────────────────────
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
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════════════════════
# PROFESSIONAL UI THEME
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500;600&display=swap');

:root{
  --bg: #0b0f17;
  --panel: #0f1624;
  --panel2:#101a2b;
  --line: rgba(255,255,255,0.08);
  --line2: rgba(255,255,255,0.12);

  --text: #e7eefc;
  --muted:#9aa7c4;
  --dim:#6f7c98;

  --blue:#3b82f6;
  --green:#22c55e;
  --amber:#f59e0b;
  --red:#ef4444;

  --radius: 14px;
  --radius_sm: 12px;
}

html, body, .stApp { background: var(--bg) !important; }
*{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
code, pre, .mono { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important; }

.block-container { padding-top: 1.25rem; max-width: 1320px; }
[data-testid="stHeader"]{ background: transparent; }
[data-testid="stSidebar"]{ display:none; }

.topbar{
  display:flex; align-items:center; justify-content:space-between;
  padding: 14px 18px;
  border:1px solid var(--line);
  border-radius: var(--radius);
  background: rgba(15,22,36,0.82);
  backdrop-filter: blur(10px);
  margin-bottom: 16px;
}
.brand{ display:flex; gap:12px; align-items:center; }
.brand-mark{
  width: 34px; height:34px;
  border-radius: 10px;
  background: radial-gradient(circle at 30% 20%, rgba(59,130,246,0.5), rgba(59,130,246,0.15)),
              linear-gradient(135deg, rgba(59,130,246,0.95), rgba(124,58,237,0.85));
  border: 1px solid rgba(255,255,255,0.10);
}
.brand-name{ font-size: 1.0rem; font-weight: 700; color: var(--text); line-height: 1.1; }
.brand-sub{ font-size: 0.72rem; color: var(--dim); margin-top: 2px; }

.badges{ display:flex; gap:10px; align-items:center; }
.badge{
  border:1px solid var(--line);
  background: rgba(16,26,43,0.70);
  border-radius: 999px;
  padding: 6px 10px;
  font-size: 0.72rem;
  color: var(--muted);
}
.badge strong{ color: var(--text); font-weight: 600; }

.card{
  background: linear-gradient(180deg, rgba(16,26,43,0.95), rgba(15,22,36,0.92));
  border: 1px solid var(--line);
  border-radius: var(--radius);
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}
.card-title{
  font-size: 0.82rem;
  color: var(--muted);
  letter-spacing: 0.03em;
  margin-bottom: 12px;
  font-weight: 700;
  text-transform: uppercase;
}

.divider{ height:1px; background: var(--line); margin: 12px 0; }

.meta{
  display:grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 8px;
  margin-top: 10px;
}
.meta .m{
  border:1px solid var(--line);
  background: rgba(16,26,43,0.55);
  border-radius: 12px;
  padding: 10px;
}
.meta .m .l{
  font-size:0.62rem; color: var(--dim);
  letter-spacing:0.08em; text-transform: uppercase;
}
.meta .m .v{
  margin-top:4px;
  font-size:0.82rem; color: var(--text);
  font-family: "JetBrains Mono", ui-monospace, monospace;
}

/* Verdict banner */
.verdict{
  border-radius: var(--radius);
  border: 1px solid var(--line);
  padding: 16px 16px;
  background: rgba(16,26,43,0.55);
}
.verdict .row{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
.verdict .k{ font-size:0.68rem; color: var(--dim); letter-spacing:0.10em; text-transform: uppercase; }
.verdict .big{
  margin-top: 4px;
  font-size: 1.55rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--text);
}
.verdict .note{
  margin-top: 6px;
  font-size: 0.92rem;
  color: var(--muted);
  line-height: 1.5;
}
.pill{
  font-size:0.72rem;
  border-radius: 999px;
  border:1px solid var(--line);
  padding: 6px 10px;
  color: var(--muted);
  background: rgba(16,26,43,0.70);
}
.pill strong{ color: var(--text); }

/* Bars */
.bar{
  height: 10px;
  border-radius: 999px;
  background: rgba(255,255,255,0.06);
  overflow: hidden;
  border:1px solid rgba(255,255,255,0.06);
}
.fill{ height:100%; border-radius: 999px; }

.section-title{
  font-size:0.78rem;
  letter-spacing:0.10em;
  text-transform: uppercase;
  color: var(--dim);
  font-weight: 700;
  margin: 10px 0 8px;
}

.footer{
  margin-top: 18px;
  padding-top: 14px;
  border-top: 1px solid var(--line);
  display:flex;
  justify-content: space-between;
  gap: 14px;
  color: var(--dim);
  font-size: 0.78rem;
}

[data-testid="stFileUploader"] > div {
  background: rgba(16,26,43,0.55) !important;
  border: 1.5px dashed var(--line2) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] label { display:none !important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS (hidden from users)
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = Path("outputs/best_model.pth")
MODEL_NAME  = "efficientnet_b0"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 224
CLASS_NAMES = ["ai", "real"]

# Production defaults
SPOOF_THRESHOLD = 0.45     # silent gate
DF_THRESHOLD    = 0.72     # review threshold
ENABLE_GRADCAM  = True     # show if available

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])


# ═══════════════════════════════════════════════════════════════════════════════
# SPOOF DETECTOR (silent)
# ═══════════════════════════════════════════════════════════════════════════════
def _blur_score(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _glare_score(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return float((v > 245).astype(np.uint8).mean())

def _moire_score(gray):
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1.0)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * 0.08)
    low = mag[cy-r:cy+r, cx-r:cx+r]
    total = float(mag.mean())
    low_e = float(low.mean())
    return float(max(total - low_e, 0.0) / (total + 1e-6))

def _border_score(gray):
    edges = cv2.Canny(gray, 80, 160)
    edges = (edges > 0).astype(np.float32)
    h, w = edges.shape
    m = int(min(h, w) * 0.06)
    top = float(edges[:m, :].mean())
    bottom = float(edges[h-m:, :].mean())
    left = float(edges[:, :m].mean())
    right = float(edges[:, w-m:].mean())
    return float((top + bottom + left + right) / 4.0)

def detect_spoof(pil_img: Image.Image, threshold: float = SPOOF_THRESHOLD):
    img_bgr = cv2.cvtColor(np.array(pil_img.resize((512, 512))), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur   = _blur_score(gray)
    glare  = _glare_score(img_bgr)
    moire  = _moire_score(gray)
    border = _border_score(gray)

    spoof_score = 0.0
    if blur < 60:   spoof_score += 0.30
    elif blur < 120:spoof_score += 0.15
    if glare > 0.015: spoof_score += 0.20
    if moire > 0.45:  spoof_score += 0.25
    if border > 0.10: spoof_score += 0.25

    spoof_score = float(np.clip(spoof_score, 0.0, 1.0))
    is_spoof = spoof_score >= float(threshold)
    return {"is_spoof": bool(is_spoof), "spoof_score": spoof_score}


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

def run_deepfake(model, pil_img: Image.Image):
    x = TFM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].detach().cpu()
    idx = int(torch.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def run_gradcam(model, pil_img: Image.Image):
    if not GRADCAM_AVAILABLE:
        return None
    pil_r = pil_img.resize((IMG_SIZE, IMG_SIZE))
    rgb_arr = np.array(pil_r).astype(np.float32) / 255.0
    inp = TFM(pil_r).unsqueeze(0).to(DEVICE)
    cam = GradCAM(model=model, target_layers=[model.conv_head])
    heat = cam(input_tensor=inp)[0]
    overlay = show_cam_on_image(rgb_arr, heat, use_rgb=True)
    return Image.fromarray(overlay)


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
deepfake_model, model_loaded = load_deepfake_model()

st.markdown(f"""
<div class="topbar">
  <div class="brand">
    <div class="brand-mark"></div>
    <div>
      <div class="brand-name">DeepShield Enterprise</div>
      <div class="brand-sub">Forensic Intelligence Platform · v3.0</div>
    </div>
  </div>
  <div class="badges">
    <div class="badge">Device: <strong>{DEVICE.upper()}</strong></div>
    <div class="badge">Model: <strong>{"Loaded" if model_loaded else "Demo"}</strong></div>
    <div class="badge">Mode: <strong>Auto-run</strong></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
for k, v in {
    "last_sig": None,
    "spoof": None,
    "df": None,
    "cam": None,
    "blocked": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.25, 1.0], gap="large")

# ── LEFT: Upload + Grad-CAM side-by-side ──────────────────────────────────────
with col_left:
    st.markdown('<div class="card"><div class="card-title">Input & Explanation</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
        key="uploader",
    )

    if uploaded:
        file_bytes = uploaded.getvalue()
        sig = (uploaded.name, len(file_bytes), hash(file_bytes))

        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        w, h = pil_img.size
        kb = len(file_bytes) / 1024
        fmt = (uploaded.type.split("/")[-1] if uploaded.type else "image").upper()

        # Auto-run only when new file
        if st.session_state.last_sig != sig:
            st.session_state.last_sig = sig
            st.session_state.spoof = None
            st.session_state.df = None
            st.session_state.cam = None
            st.session_state.blocked = False

            with st.spinner("Analyzing..."):
                time.sleep(0.10)

                # Silent spoof gate
                sr = detect_spoof(pil_img, threshold=SPOOF_THRESHOLD)
                st.session_state.spoof = sr

                if sr["is_spoof"]:
                    st.session_state.blocked = True
                else:
                    # Deepfake detection
                    label, conf, probs = run_deepfake(deepfake_model, pil_img)
                    st.session_state.df = {"label": label, "conf": conf, "probs": probs}

                    # Grad-CAM
                    if ENABLE_GRADCAM and GRADCAM_AVAILABLE:
                        cam_img = run_gradcam(deepfake_model, pil_img)
                        st.session_state.cam = cam_img

        # Two columns: image | gradcam
        a, b = st.columns(2, gap="medium")
        with a:
            st.markdown('<div class="section-title">Uploaded image</div>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)

        with b:
            st.markdown('<div class="section-title">Grad-CAM</div>', unsafe_allow_html=True)
            if st.session_state.blocked:
                st.info("Grad-CAM disabled (input flagged by spoof gate).")
            else:
                if not ENABLE_GRADCAM:
                    st.info("Grad-CAM disabled by configuration.")
                elif not GRADCAM_AVAILABLE:
                    st.info("Install Grad-CAM: pip install grad-cam")
                elif st.session_state.cam is None:
                    st.info("Grad-CAM not available for this run.")
                else:
                    st.image(st.session_state.cam, use_container_width=True)
                    buf = io.BytesIO()
                    st.session_state.cam.save(buf, format="PNG")
                    st.download_button(
                        "Export Grad-CAM (PNG)",
                        buf.getvalue(),
                        "deepshield_gradcam.png",
                        "image/png",
                        use_container_width=True,
                    )

        st.markdown(f"""
        <div class="meta">
          <div class="m"><div class="l">Width</div><div class="v">{w}px</div></div>
          <div class="m"><div class="l">Height</div><div class="v">{h}px</div></div>
          <div class="m"><div class="l">Size</div><div class="v">{kb:.1f} KB</div></div>
          <div class="m"><div class="l">Format</div><div class="v">{fmt}</div></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Upload an image to automatically run detection and show Grad-CAM.")

    st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: Highlight AI/REAL (no Stage 1 shown) ───────────────────────────────
with col_right:
    st.markdown('<div class="card"><div class="card-title">Verdict</div>', unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div class="verdict">
          <div class="row">
            <div>
              <div class="k">Status</div>
              <div class="big">Awaiting input</div>
              <div class="note">Upload an image to view AI / Real classification.</div>
            </div>
            <div class="pill mono">Auto-run</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        blocked = bool(st.session_state.blocked)
        sr = st.session_state.spoof
        dfr = st.session_state.df

        if blocked:
            spoof_score = float(sr["spoof_score"]) if sr else 0.0
            st.markdown(f"""
            <div class="verdict">
              <div class="row">
                <div>
                  <div class="k">Result</div>
                  <div class="big" style="color: var(--red);">Input Rejected</div>
                  <div class="note">
                    The uploaded image was flagged by the spoof gate. Please upload a clean original photo
                    (not recaptured from screen/print).<br/>
                    Spoof score: <span class="mono">{spoof_score:.2%}</span>
                  </div>
                </div>
                <div class="pill mono">Gate: <strong style="color: var(--red);">Blocked</strong></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            label = dfr["label"]
            conf = float(dfr["conf"])
            probs = dfr["probs"]
            ai_p = float(probs[0])
            real_p = float(probs[1])

            # Highlight output
            uncertain = conf < DF_THRESHOLD
            if uncertain:
                verdict = "Uncertain"
                color = "var(--amber)"
                note = f"Low confidence: {conf:.2%} (review threshold {DF_THRESHOLD:.2%})."
                pill = "Review"
            elif label == "ai":
                verdict = "AI-Generated"
                color = "var(--red)"
                note = f"Model confidence: {conf:.2%}."
                pill = "AI"
            else:
                verdict = "Real Photograph"
                color = "var(--green)"
                note = f"Model confidence: {conf:.2%}."
                pill = "Real"

            st.markdown(f"""
            <div class="verdict">
              <div class="row">
                <div>
                  <div class="k">Result</div>
                  <div class="big" style="color:{color};">{verdict}</div>
                  <div class="note">{note}</div>
                </div>
                <div class="pill mono">Label: <strong style="color:{color};">{pill}</strong></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probability</div>', unsafe_allow_html=True)

            # AI bar
            st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <div style="color:var(--muted); font-size:0.90rem;">AI / Deepfake</div>
                <div class="mono" style="color:var(--text); font-size:0.84rem;">{ai_p:.6f}</div>
              </div>
              <div class="bar"><div class="fill" style="width:{ai_p*100:.2f}%; background: rgba(239,68,68,0.85);"></div></div>
            </div>
            """, unsafe_allow_html=True)

            # Real bar
            st.markdown(f"""
            <div style="margin-bottom:6px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <div style="color:var(--muted); font-size:0.90rem;">Real</div>
                <div class="mono" style="color:var(--text); font-size:0.84rem;">{real_p:.6f}</div>
              </div>
              <div class="bar"><div class="fill" style="width:{real_p*100:.2f}%; background: rgba(34,197,94,0.85);"></div></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


st.markdown(f"""
<div class="footer">
  <div>
    DeepShield Enterprise · Deepfake & Spoof Attack Detection in Cyber Threats<br/>
    Final Year B.Tech — Artificial Intelligence & Data Science
  </div>
  <div style="text-align:right;">
    EfficientNet-B0 · PyTorch · OpenCV · Grad-CAM<br/>
    Device: {DEVICE.upper()} · v3.0
  </div>
</div>
""", unsafe_allow_html=True)