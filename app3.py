"""
DeepShield Enterprise v3.0
Auto-run Pipeline: Spoof Gate (silent) → Deepfake Detection → Grad-CAM
Add-ons: PDF Report · Face-overlap reasoning · Forensic views (ELA + Noise)
Final Year B.Tech — Artificial Intelligence & Data Science
"""

import io
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
import timm
from PIL import Image, ImageChops, ImageEnhance
from torchvision import transforms

# ── Optional Grad-CAM ─────────────────────────────────────────────────────────
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

# ── PDF Report (ReportLab) ────────────────────────────────────────────────────
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


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
  --bg:#0b0f17;
  --line:rgba(255,255,255,0.08);
  --text:#e7eefc;
  --muted:#9aa7c4;
  --dim:#6f7c98;
  --green:#22c55e;
  --amber:#f59e0b;
  --red:#ef4444;
  --radius:14px;
}

html, body, .stApp { background: var(--bg) !important; }
*{ font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
code, pre, .mono { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important; }

.block-container { padding-top: 1.25rem; max-width: 1320px; }
[data-testid="stHeader"]{ background: transparent; }
[data-testid="stSidebar"]{ display:none; }

.topbar{
  display:flex; align-items:center; justify-content:space-between;
  padding:14px 18px;
  border:1px solid var(--line);
  border-radius: var(--radius);
  background: rgba(15,22,36,0.82);
  backdrop-filter: blur(10px);
  margin-bottom: 16px;
}
.brand{ display:flex; gap:12px; align-items:center; }
.brand-mark{
  width:34px; height:34px;
  border-radius:10px;
  background: radial-gradient(circle at 30% 20%, rgba(59,130,246,0.5), rgba(59,130,246,0.15)),
              linear-gradient(135deg, rgba(59,130,246,0.95), rgba(124,58,237,0.85));
  border:1px solid rgba(255,255,255,0.10);
}
.brand-name{ font-size:1.0rem; font-weight:700; color:var(--text); line-height:1.1; }
.brand-sub{ font-size:0.72rem; color:var(--dim); margin-top:2px; }

.badges{ display:flex; gap:10px; align-items:center; }
.badge{
  border:1px solid var(--line);
  background: rgba(16,26,43,0.70);
  border-radius: 999px;
  padding:6px 10px;
  font-size:0.72rem;
  color:var(--muted);
}
.badge strong{ color:var(--text); font-weight:600; }

.card{
  background: linear-gradient(180deg, rgba(16,26,43,0.95), rgba(15,22,36,0.92));
  border:1px solid var(--line);
  border-radius: var(--radius);
  padding:18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}
.card-title{
  font-size:0.82rem;
  color:var(--muted);
  letter-spacing:0.03em;
  margin-bottom:12px;
  font-weight:700;
  text-transform: uppercase;
}

.divider{ height:1px; background: var(--line); margin: 12px 0; }

.meta{
  display:grid;
  grid-template-columns: repeat(4, 1fr);
  gap:8px;
  margin-top:10px;
}
.meta .m{
  border:1px solid var(--line);
  background: rgba(16,26,43,0.55);
  border-radius: 12px;
  padding:10px;
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
  border:1px solid var(--line);
  padding:16px 16px;
  background: rgba(16,26,43,0.55);
}
.verdict .row{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
.verdict .k{ font-size:0.68rem; color: var(--dim); letter-spacing:0.10em; text-transform: uppercase; }
.verdict .big{ margin-top:4px; font-size:1.55rem; font-weight:800; letter-spacing:-0.02em; color:var(--text); }
.verdict .note{ margin-top:6px; font-size:0.92rem; color:var(--muted); line-height:1.5; }
.pill{
  font-size:0.72rem;
  border-radius:999px;
  border:1px solid var(--line);
  padding:6px 10px;
  color:var(--muted);
  background: rgba(16,26,43,0.70);
}
.pill strong{ color: var(--text); }

/* Bars */
.bar{
  height:10px;
  border-radius:999px;
  background: rgba(255,255,255,0.06);
  overflow:hidden;
  border:1px solid rgba(255,255,255,0.06);
}
.fill{ height:100%; border-radius:999px; }

.section-title{
  font-size:0.78rem;
  letter-spacing:0.10em;
  text-transform: uppercase;
  color: var(--dim);
  font-weight: 700;
  margin: 10px 0 8px;
}

/* Explanation card */
.explain{
  margin-top: 12px;
  border-radius: var(--radius);
  border:1px solid var(--line);
  background: rgba(16,26,43,0.45);
  padding:14px 14px;
}
.explain .head{ display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:10px; }
.explain .title{
  font-size:0.82rem; font-weight:800;
  letter-spacing:0.08em; text-transform: uppercase; color: var(--muted);
}
.explain .chip{
  font-size:0.72rem;
  border-radius:999px;
  border:1px solid var(--line);
  padding:5px 9px;
  color: var(--muted);
  background: rgba(16,26,43,0.70);
}
.explain ul{
  margin: 0.2rem 0 0.4rem 1.0rem;
  color: var(--muted);
  font-size: 0.90rem;
  line-height: 1.55;
}
.explain .sub{ font-size:0.86rem; color: var(--dim); margin-top: 8px; }

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
  border: 1.5px dashed rgba(255,255,255,0.12) !important;
  border-radius: var(--radius) !important;
}
[data-testid="stFileUploader"] label { display:none !important; }
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

SPOOF_THRESHOLD = 0.45
DF_THRESHOLD    = 0.72
ENABLE_GRADCAM  = True

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
])

# ═══════════════════════════════════════════════════════════════════════════════
# OPENCV DNN FACE MODEL (Best) + Haar fallback
# ═══════════════════════════════════════════════════════════════════════════════
DNN_PROTO = Path("models/deploy.prototxt")
DNN_MODEL = Path("models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

@st.cache_resource(show_spinner=False)
def load_face_dnn():
    if not (DNN_PROTO.exists() and DNN_MODEL.exists()):
        return None
    try:
        return cv2.dnn.readNetFromCaffe(str(DNN_PROTO), str(DNN_MODEL))
    except Exception:
        return None

def detect_faces_bbox(pil_img: Image.Image, conf_thresh: float = 0.60):
    """
    Returns list of bboxes in normalized coords: (x1,y1,x2,y2) in [0..1].
    Uses OpenCV DNN if model files exist, else Haar fallback.
    """
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    net = load_face_dnn()
    if net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        det = net.forward()

        boxes = []
        for i in range(det.shape[2]):
            confidence = float(det[0, 0, i, 2])
            if confidence < conf_thresh:
                continue
            box = det[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w, x2); y2 = min(h, y2)
            if (x2 - x1) > 40 and (y2 - y1) > 40:
                boxes.append((x1 / w, y1 / h, x2 / w, y2 / h))
        return boxes

    # Haar fallback
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

    return [(x / w, y / h, (x + fw) / w, (y + fh) / h) for (x, y, fw, fh) in faces]


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
    if blur < 60:    spoof_score += 0.30
    elif blur < 120: spoof_score += 0.15
    if glare > 0.015: spoof_score += 0.20
    if moire > 0.45:  spoof_score += 0.25
    if border > 0.10: spoof_score += 0.25

    spoof_score = float(np.clip(spoof_score, 0.0, 1.0))
    is_spoof = spoof_score >= float(threshold)
    return {"is_spoof": bool(is_spoof), "spoof_score": spoof_score}


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL + GRADCAM
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
# FORENSIC VIEWS: ELA + NOISE RESIDUAL
# ═══════════════════════════════════════════════════════════════════════════════
def compute_ela(pil_img: Image.Image, quality: int = 90, scale: float = 10.0):
    rgb = pil_img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(rgb, recompressed)
    diff = ImageEnhance.Contrast(diff).enhance(scale)
    return diff

def compute_noise_residual(pil_img: Image.Image, sigma: float = 1.3):
    arr = np.array(pil_img.convert("RGB"))
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    resid = cv2.absdiff(arr, blur)
    resid = cv2.normalize(resid, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(resid)


# ═══════════════════════════════════════════════════════════════════════════════
# FACE OVERLAP (works with DNN/Haar face boxes)
# ═══════════════════════════════════════════════════════════════════════════════
def cam_heat_from_overlay(cam_pil: Image.Image):
    if cam_pil is None:
        return None
    arr = np.asarray(cam_pil).astype(np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.clip(r - (0.5 * g + 0.5 * b), 0, 1)

def face_overlap_score(cam_pil: Image.Image, face_boxes_norm, heat_thresh_q: float = 0.85):
    heat = cam_heat_from_overlay(cam_pil)
    if heat is None:
        return None
    hh, ww = heat.shape

    flat = heat.reshape(-1)
    q = float(np.quantile(flat, float(heat_thresh_q)))
    hot = (heat >= q).astype(np.uint8)

    if not face_boxes_norm:
        return {"faces_found": 0, "hot_overlap_ratio": 0.0, "hot_face_share": 0.0}

    face_mask = np.zeros((hh, ww), dtype=np.uint8)
    for (x1, y1, x2, y2) in face_boxes_norm:
        X1 = int(np.clip(x1 * ww, 0, ww - 1))
        Y1 = int(np.clip(y1 * hh, 0, hh - 1))
        X2 = int(np.clip(x2 * ww, 0, ww))
        Y2 = int(np.clip(y2 * hh, 0, hh))
        face_mask[Y1:Y2, X1:X2] = 1

    hot_total = float(hot.sum() + 1e-6)
    face_total = float(face_mask.sum() + 1e-6)
    hot_in_face = float((hot * face_mask).sum())

    return {
        "faces_found": int(len(face_boxes_norm)),
        "hot_overlap_ratio": float(np.clip(hot_in_face / hot_total, 0, 1)),
        "hot_face_share": float(np.clip(hot_in_face / face_total, 0, 1)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# WHY EXPLANATION
# ═══════════════════════════════════════════════════════════════════════════════
def gradcam_focus_score(cam_pil: Image.Image):
    if cam_pil is None:
        return None
    arr = np.asarray(cam_pil).astype(np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    heat = np.clip(r - (0.5*g + 0.5*b), 0, 1)
    flat = heat.reshape(-1)
    if flat.size < 10:
        return None
    q80 = np.quantile(flat, 0.80)
    top = flat[flat >= q80]
    total = float(flat.sum() + 1e-6)
    focus = float(np.clip(float(top.sum()) / total, 0, 1))
    q90 = np.quantile(flat, 0.90)
    peaks = float((flat >= q90).mean())
    return {"focus": focus, "peaks": peaks}

def build_explanation(label: str, conf: float, ai_p: float, real_p: float, cam_metrics, face_metrics):
    gap = float(abs(ai_p - real_p))
    conf_level = "high" if conf >= 0.90 else ("medium" if conf >= DF_THRESHOLD else "low")
    bullets = []

    if conf_level == "high":
        bullets.append("High confidence decision with strong probability separation.")
    elif conf_level == "medium":
        bullets.append("Moderate confidence decision; probabilities are separated but not extreme.")
    else:
        bullets.append("Low confidence decision; manual review recommended.")

    bullets.append(f"Probability gap is {gap:.2%}, indicating how strongly the model favors one class.")

    if cam_metrics is None:
        bullets.append("Grad-CAM attention summary not available for this run.")
    else:
        focus = cam_metrics["focus"]
        peaks = cam_metrics["peaks"]
        if focus >= 0.62:
            bullets.append("Model attention is concentrated, suggesting reliance on localized visual cues.")
        elif focus >= 0.48:
            bullets.append("Model attention is moderately focused; cues appear partly localized and partly global.")
        else:
            bullets.append("Model attention is spread, which often happens when cues are weak or distributed.")
        if peaks >= 0.12:
            bullets.append("Strong attention peaks detected, indicating distinct high-signal regions.")
        else:
            bullets.append("Attention peaks are mild; no extremely strong localized signal was found.")

    if face_metrics is None:
        bullets.append("Face-overlap reasoning not available (no Grad-CAM heat or face detection).")
    else:
        if face_metrics["faces_found"] == 0:
            bullets.append("No face detected; model may rely more on background or global image artifacts.")
        else:
            overlap = face_metrics["hot_overlap_ratio"]
            if overlap >= 0.55:
                bullets.append("Most high-attention regions overlap the face area, meaning the model relied mainly on facial cues.")
            elif overlap >= 0.30:
                bullets.append("A moderate portion of high-attention overlaps the face; both face and background cues contribute.")
            else:
                bullets.append("High-attention regions are mostly outside the face, suggesting reliance on background or compression artifacts.")

    if label == "ai":
        bullets.append("AI-generated images can contain subtle synthetic texture patterns that CNNs learn to detect.")
        action = "Recommended: verify with source file/metadata and test multiple images from the same source."
    else:
        bullets.append("Real photos often show consistent natural detail transitions and sensor noise patterns.")
        action = "Recommended: if you expected AI here, consider retraining with more similar AI samples and augmentations."

    if conf < DF_THRESHOLD:
        action = "Recommended: treat as uncertain. Upload a higher-quality original and verify via source/metadata."

    return bullets, action


# ═══════════════════════════════════════════════════════════════════════════════
# PDF REPORT
# ═══════════════════════════════════════════════════════════════════════════════
def pil_to_png_bytes(pil_img: Image.Image, max_w: int = 900):
    img = pil_img.copy().convert("RGB")
    w, h = img.size
    if w > max_w:
        nh = int(h * (max_w / w))
        img = img.resize((max_w, nh))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def build_pdf_report(
    case_id: str,
    timestamp: str,
    device: str,
    verdict: str,
    conf: float,
    ai_p: float,
    real_p: float,
    bullets,
    action: str,
    face_metrics,
    input_img: Image.Image,
    cam_img: Image.Image | None,
    ela_img: Image.Image | None,
    noise_img: Image.Image | None,
):
    if not REPORTLAB_AVAILABLE:
        return None

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4

    def draw_title(y):
        c.setFont("Helvetica-Bold", 16)
        c.drawString(40, y, "DeepShield Enterprise — Forensic Report")
        c.setFont("Helvetica", 9)
        c.drawString(40, y - 14, f"Case ID: {case_id}    Timestamp: {timestamp}    Device: {device}")

    def draw_kv(y, k, v):
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, k)
        c.setFont("Helvetica", 10)
        c.drawString(170, y, str(v))

    def draw_paragraph(y, lines, leading=13, max_lines=999):
        c.setFont("Helvetica", 10)
        count = 0
        for line in lines:
            if count >= max_lines:
                break
            c.drawString(52, y, f"- {line}")
            y -= leading
            count += 1
        return y

    def draw_image(pil_img, x, y, w):
        if pil_img is None:
            return
        png = pil_to_png_bytes(pil_img, max_w=int(w))
        ir = ImageReader(io.BytesIO(png))
        iw, ih = pil_img.size
        ratio = ih / max(iw, 1)
        hh = w * ratio
        c.drawImage(ir, x, y - hh, width=w, height=hh, preserveAspectRatio=True, mask='auto')

    y = H - 40
    draw_title(y)
    y -= 40

    draw_kv(y, "Verdict", verdict); y -= 16
    draw_kv(y, "Confidence", f"{conf:.2%}"); y -= 16
    draw_kv(y, "AI Probability", f"{ai_p:.6f}"); y -= 16
    draw_kv(y, "Real Probability", f"{real_p:.6f}"); y -= 16

    if face_metrics:
        y -= 6
        draw_kv(y, "Faces Found", face_metrics.get("faces_found", 0)); y -= 16
        draw_kv(y, "Hot-overlap in Face", f"{face_metrics.get('hot_overlap_ratio', 0.0):.2%}"); y -= 16

    y -= 8
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Why this result")
    y -= 16
    y = draw_paragraph(y, bullets, leading=13, max_lines=12)

    y -= 6
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Recommended action")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(40, y, action[:110])
    y -= 26

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Visual evidence")
    y -= 12

    col_w = (W - 40*2 - 12) / 2
    draw_image(input_img, 40, y, col_w)
    draw_image(cam_img, 40 + col_w + 12, y, col_w)

    c.showPage()

    y = H - 40
    draw_title(y)
    y -= 44
    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Forensic views"); y -= 14
    c.setFont("Helvetica", 10)
    c.drawString(40, y, "ELA highlights potential compression inconsistencies; Noise residual emphasizes high-frequency artifacts.")
    y -= 18

    col_w = (W - 40*2 - 12) / 2
    draw_image(ela_img, 40, y, col_w)
    draw_image(noise_img, 40 + col_w + 12, y, col_w)

    c.save()
    buf.seek(0)
    return buf.getvalue()


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
    "blocked": False,
    "spoof": None,
    "df": None,
    "cam": None,
    "cam_metrics": None,
    "face_boxes": None,
    "face_metrics": None,
    "ela": None,
    "noise": None,
    "why_bullets": None,
    "why_action": None,
    "case_id": None,
    "report_pdf": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.25, 1.0], gap="large")

# ── LEFT: Upload + Grad-CAM + Forensic Views ─────────────────────────────────
with col_left:
    st.markdown('<div class="card"><div class="card-title">Input & Visual Evidence</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
        key="uploader",
    )

    pil_img = None
    if uploaded:
        file_bytes = uploaded.getvalue()
        sig = (uploaded.name, len(file_bytes), hash(file_bytes))

        pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        w, h = pil_img.size
        kb = len(file_bytes) / 1024
        fmt = (uploaded.type.split("/")[-1] if uploaded.type else "image").upper()

        if load_face_dnn() is None:
            st.info("Face-overlap uses OpenCV DNN. Download 2 face model files into /models for best accuracy (Haar fallback is active).")

        # Auto-run only when new file
        if st.session_state.last_sig != sig:
            st.session_state.last_sig = sig
            st.session_state.blocked = False
            st.session_state.spoof = None
            st.session_state.df = None
            st.session_state.cam = None
            st.session_state.cam_metrics = None
            st.session_state.face_boxes = None
            st.session_state.face_metrics = None
            st.session_state.ela = None
            st.session_state.noise = None
            st.session_state.why_bullets = None
            st.session_state.why_action = None
            st.session_state.report_pdf = None

            st.session_state.case_id = datetime.now().strftime("%Y%m%d-%H%M%S")

            with st.spinner("Analyzing..."):
                time.sleep(0.08)

                sr = detect_spoof(pil_img, threshold=SPOOF_THRESHOLD)
                st.session_state.spoof = sr

                if sr["is_spoof"]:
                    st.session_state.blocked = True
                else:
                    label, conf, probs = run_deepfake(deepfake_model, pil_img)
                    st.session_state.df = {"label": label, "conf": conf, "probs": probs}

                    if ENABLE_GRADCAM and GRADCAM_AVAILABLE:
                        cam_img = run_gradcam(deepfake_model, pil_img)
                        st.session_state.cam = cam_img
                        st.session_state.cam_metrics = gradcam_focus_score(cam_img)

                    st.session_state.face_boxes = detect_faces_bbox(pil_img)
                    if st.session_state.cam is not None:
                        st.session_state.face_metrics = face_overlap_score(
                            st.session_state.cam,
                            st.session_state.face_boxes or [],
                            heat_thresh_q=0.85
                        )

                    st.session_state.ela = compute_ela(pil_img, quality=90, scale=10.0)
                    st.session_state.noise = compute_noise_residual(pil_img, sigma=1.3)

                    ai_p = float(probs[0]); real_p = float(probs[1])
                    st.session_state.why_bullets, st.session_state.why_action = build_explanation(
                        label=label,
                        conf=float(conf),
                        ai_p=ai_p,
                        real_p=real_p,
                        cam_metrics=st.session_state.cam_metrics,
                        face_metrics=st.session_state.face_metrics,
                    )

        a, b = st.columns(2, gap="medium")
        with a:
            st.markdown('<div class="section-title">Uploaded image</div>', unsafe_allow_html=True)
            st.image(pil_img, use_container_width=True)

        with b:
            st.markdown('<div class="section-title">Grad-CAM</div>', unsafe_allow_html=True)
            if st.session_state.blocked:
                st.info("Grad-CAM disabled (input flagged by spoof gate).")
            else:
                if not GRADCAM_AVAILABLE:
                    st.info("Grad-CAM not installed. Install: pip install grad-cam")
                elif st.session_state.cam is None:
                    st.info("Grad-CAM not available for this run.")
                else:
                    st.image(st.session_state.cam, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Forensic views</div>', unsafe_allow_html=True)

        t1, t2 = st.tabs(["ELA", "Noise residual"])
        with t1:
            if st.session_state.ela is not None:
                st.image(st.session_state.ela, use_container_width=True, caption="Error Level Analysis (ELA)")
        with t2:
            if st.session_state.noise is not None:
                st.image(st.session_state.noise, use_container_width=True, caption="Noise residual (high-frequency emphasis)")

        st.markdown(f"""
        <div class="meta">
          <div class="m"><div class="l">Width</div><div class="v">{w}px</div></div>
          <div class="m"><div class="l">Height</div><div class="v">{h}px</div></div>
          <div class="m"><div class="l">Size</div><div class="v">{kb:.1f} KB</div></div>
          <div class="m"><div class="l">Format</div><div class="v">{fmt}</div></div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("Upload an image to auto-run detection, Grad-CAM, and forensic views.")

    st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: Verdict + Why + PDF report ─────────────────────────────────────────
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
                    The image was flagged by the spoof gate. Upload a clean original photo
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

            st.markdown(f"""
            <div style="margin-bottom:12px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <div style="color:var(--muted); font-size:0.90rem;">AI / Deepfake</div>
                <div class="mono" style="color:var(--text); font-size:0.84rem;">{ai_p:.6f}</div>
              </div>
              <div class="bar"><div class="fill" style="width:{ai_p*100:.2f}%; background: rgba(239,68,68,0.85);"></div></div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-bottom:6px;">
              <div style="display:flex; justify-content:space-between; margin-bottom:6px;">
                <div style="color:var(--muted); font-size:0.90rem;">Real</div>
                <div class="mono" style="color:var(--text); font-size:0.84rem;">{real_p:.6f}</div>
              </div>
              <div class="bar"><div class="fill" style="width:{real_p*100:.2f}%; background: rgba(34,197,94,0.85);"></div></div>
            </div>
            """, unsafe_allow_html=True)

            fm = st.session_state.face_metrics
            if fm is not None:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="pill mono">Faces: <strong>{fm.get("faces_found",0)}</strong> '
                    f'· Face attention overlap: <strong>{fm.get("hot_overlap_ratio",0.0):.2%}</strong></div>',
                    unsafe_allow_html=True
                )

            bullets = st.session_state.why_bullets or []
            action = st.session_state.why_action or ""
            conf_tag = "High" if conf >= 0.90 else ("Medium" if conf >= DF_THRESHOLD else "Low")

            st.markdown(f"""
            <div class="explain">
              <div class="head">
                <div class="title">Why this result</div>
                <div class="chip mono">Confidence: <strong>{conf_tag}</strong></div>
              </div>
              <ul>
                {''.join([f'<li>{b}</li>' for b in bullets])}
              </ul>
              <div class="sub"><span class="mono">Recommended action:</span> {action}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            if not REPORTLAB_AVAILABLE:
                st.info("PDF report requires reportlab. Install: pip install reportlab")
            else:
                if st.button("Generate PDF Report", use_container_width=True):
                    case_id = st.session_state.case_id or datetime.now().strftime("%Y%m%d-%H%M%S")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    pdf = build_pdf_report(
                        case_id=case_id,
                        timestamp=timestamp,
                        device=DEVICE.upper(),
                        verdict=verdict,
                        conf=conf,
                        ai_p=ai_p,
                        real_p=real_p,
                        bullets=bullets,
                        action=action,
                        face_metrics=fm,
                        input_img=pil_img,
                        cam_img=st.session_state.cam,
                        ela_img=st.session_state.ela,
                        noise_img=st.session_state.noise,
                    )
                    st.session_state.report_pdf = pdf

                if st.session_state.report_pdf:
                    st.download_button(
                        "Download PDF Report",
                        st.session_state.report_pdf,
                        file_name=f"DeepShield_Report_{st.session_state.case_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

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