"""
DeepShield Enterprise v4.1
Fixed + Improved: Prediction + Admin Review Queue + Verified Dataset + Retrain Trigger
Fixes: NameError for ADMIN_ICON, sidebar navigation, and other minor issues
"""

import csv
import hashlib
import io
import shutil
import subprocess
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

# ── Optional PDF report ───────────────────────────────────────────────────────
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DeepShield Enterprise",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# THEME / CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

:root{
  --bg:#0b0f17;
  --card:#101a2b;
  --line:rgba(255,255,255,0.08);
  --text:#e7eefc;
  --muted:#9aa7c4;
  --dim:#6f7c98;
  --green:#22c55e;
  --amber:#f59e0b;
  --red:#ef4444;
  --blue:#3b82f6;
  --radius:14px;
}
html, body, .stApp { background: var(--bg) !important; }
*{ font-family: Inter, system-ui, sans-serif; }
code, pre, .mono { font-family: "JetBrains Mono", monospace !important; }

.block-container { max-width: 1350px; padding-top: 1rem; }
[data-testid="stHeader"]{ background: transparent; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(16,26,43,0.98) !important;
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .stRadio label {
    color: var(--muted) !important;
    font-size: 0.88rem;
}
[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    color: var(--text) !important;
}

.topbar{
  display:flex; justify-content:space-between; align-items:center;
  padding:14px 18px; margin-bottom:16px;
  border:1px solid var(--line); border-radius:var(--radius);
  background: rgba(16,26,43,0.88);
}
.brand-wrap{ display:flex; gap:12px; align-items:center; }
.brand-mark{
  width:34px; height:34px; border-radius:10px;
  background: linear-gradient(135deg, rgba(59,130,246,0.95), rgba(124,58,237,0.85));
  display:flex; align-items:center; justify-content:center;
  font-size:1.1rem;
}
.brand-title{ color:var(--text); font-size:1rem; font-weight:700; }
.brand-sub{ color:var(--dim); font-size:0.75rem; margin-top:2px; }

.badge{
  display:inline-block;
  padding:6px 10px;
  border:1px solid var(--line);
  border-radius:999px;
  color:var(--muted);
  background: rgba(16,26,43,0.75);
  font-size:0.74rem;
  margin-left:8px;
}

.card{
  background: linear-gradient(180deg, rgba(16,26,43,0.95), rgba(15,22,36,0.92));
  border:1px solid var(--line);
  border-radius: var(--radius);
  padding:18px;
  box-shadow: 0 10px 28px rgba(0,0,0,0.22);
}
.card-title{
  font-size:0.82rem;
  color:var(--muted);
  letter-spacing:0.04em;
  text-transform:uppercase;
  font-weight:700;
  margin-bottom:12px;
}
.section-title{
  font-size:0.78rem;
  letter-spacing:0.10em;
  text-transform: uppercase;
  color: var(--dim);
  font-weight: 700;
  margin: 10px 0 8px;
}
.divider{ height:1px; background: var(--line); margin: 12px 0; }

.verdict{
  border-radius: var(--radius);
  border:1px solid var(--line);
  padding:16px;
  background: rgba(16,26,43,0.55);
}
.pill{
  font-size:0.74rem;
  border-radius:999px;
  border:1px solid var(--line);
  padding:6px 10px;
  color:var(--muted);
  background: rgba(16,26,43,0.70);
  display:inline-block;
}
.meta{
  display:grid;
  grid-template-columns: repeat(4, 1fr);
  gap:8px;
  margin-top:12px;
}
.m{
  border:1px solid var(--line);
  background: rgba(16,26,43,0.55);
  border-radius: 12px;
  padding:10px;
}
.m .l{
  font-size:0.62rem;
  color: var(--dim);
  letter-spacing:0.08em;
  text-transform: uppercase;
}
.m .v{
  margin-top:4px;
  font-size:0.82rem;
  color: var(--text);
  font-family: "JetBrains Mono", monospace;
}
.bar{
  height:10px;
  border-radius:999px;
  background: rgba(255,255,255,0.06);
  overflow:hidden;
  border:1px solid rgba(255,255,255,0.06);
}
.fill{ height:100%; border-radius:999px; }

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

/* Status badges */
.status-pending  { color: var(--amber); }
.status-approved { color: var(--green); }
.status-rejected { color: var(--red);   }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="topbar">
  <div class="brand-wrap">
    <div class="brand-mark">🛡️</div>
    <div>
      <div class="brand-title">DeepShield Enterprise</div>
      <div class="brand-sub">Deepfake &amp; Spoof Attack Detection with Admin Verification</div>
    </div>
  </div>
  <div>
    <span class="badge">EfficientNet-B0</span>
    <span class="badge">Grad-CAM</span>
    <span class="badge">Human-in-the-loop</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
MODEL_PATH   = Path("outputs/best_model.pth")
MODEL_NAME   = "efficientnet_b0"
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE     = 224
CLASS_NAMES  = ["ai", "real"]

SPOOF_THRESHOLD = 0.45
DF_THRESHOLD    = 0.72
ENABLE_GRADCAM  = True

APP_DIR       = Path(".")
DATASET_DIR   = APP_DIR / "dataset"
REVIEW_QUEUE  = APP_DIR / "review_queue"
VERIFIED_AI   = APP_DIR / "verified_data" / "ai"
VERIFIED_REAL = APP_DIR / "verified_data" / "real"
REJECTED_DATA = APP_DIR / "rejected_data"
LOGS_DIR      = APP_DIR / "logs"
MODELS_DIR    = APP_DIR / "models"

REVIEW_LOG    = LOGS_DIR / "review_log.csv"
RETRAIN_LOG   = LOGS_DIR / "retrain_log.csv"
RETRAIN_SCRIPT = APP_DIR / "retrain_model.py"

ADMIN_PASSWORD = "1234"  # change this in production

# Create required directories
for folder in [REVIEW_QUEUE, VERIFIED_AI, VERIFIED_REAL, REJECTED_DATA, LOGS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TFM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
defaults = {
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
    "saved_to_review_queue": False,
    "saved_review_filename": None,
    "admin_authenticated": False,
    "retrain_authenticated": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════════════════════
# FACE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════
DNN_PROTO  = Path("models/deploy.prototxt")
DNN_MODEL  = Path("models/res10_300x300_ssd_iter_140000_fp16.caffemodel")


@st.cache_resource(show_spinner=False)
def load_face_dnn():
    if not (DNN_PROTO.exists() and DNN_MODEL.exists()):
        return None
    try:
        return cv2.dnn.readNetFromCaffe(str(DNN_PROTO), str(DNN_MODEL))
    except Exception:
        return None


def detect_faces_bbox(pil_img: Image.Image, conf_thresh: float = 0.60):
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]

    net = load_face_dnn()
    if net is not None:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
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
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x1) > 40 and (y2 - y1) > 40:
                boxes.append((x1 / w, y1 / h, x2 / w, y2 / h))
        return boxes

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return [(x / w, y / h, (x + fw) / w, (y + fh) / h) for (x, y, fw, fh) in faces]


# ═══════════════════════════════════════════════════════════════════════════════
# SPOOF DETECTOR
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
    low = mag[cy - r:cy + r, cx - r:cx + r]
    total = float(mag.mean())
    low_e = float(low.mean())
    return float(max(total - low_e, 0.0) / (total + 1e-6))


def _border_score(gray):
    edges = cv2.Canny(gray, 80, 160)
    edges = (edges > 0).astype(np.float32)
    h, w = edges.shape
    m = int(min(h, w) * 0.06)
    top    = float(edges[:m, :].mean())
    bottom = float(edges[h - m:, :].mean())
    left   = float(edges[:, :m].mean())
    right  = float(edges[:, w - m:].mean())
    return float((top + bottom + left + right) / 4.0)


def detect_spoof(pil_img: Image.Image, threshold: float = SPOOF_THRESHOLD):
    img_bgr = cv2.cvtColor(np.array(pil_img.resize((512, 512))), cv2.COLOR_RGB2BGR)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    blur   = _blur_score(gray)
    glare  = _glare_score(img_bgr)
    moire  = _moire_score(gray)
    border = _border_score(gray)

    spoof_score = 0.0
    if blur < 60:
        spoof_score += 0.30
    elif blur < 120:
        spoof_score += 0.15
    if glare > 0.015:
        spoof_score += 0.20
    if moire > 0.45:
        spoof_score += 0.25
    if border > 0.10:
        spoof_score += 0.25

    spoof_score = float(np.clip(spoof_score, 0.0, 1.0))
    return {
        "is_spoof": spoof_score >= float(threshold),
        "spoof_score": spoof_score,
        "blur": blur,
        "glare": glare,
        "moire": moire,
        "border": border,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# DEEPFAKE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_deepfake_model():
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model, True
    model.to(DEVICE).eval()
    return model, False


deepfake_model, model_loaded = load_deepfake_model()


def run_deepfake(model, pil_img: Image.Image):
    x = TFM(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].detach().cpu()
    idx = int(torch.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


def run_gradcam(model, pil_img: Image.Image):
    if not GRADCAM_AVAILABLE:
        return None
    pil_r   = pil_img.resize((IMG_SIZE, IMG_SIZE))
    rgb_arr = np.array(pil_r).astype(np.float32) / 255.0
    inp     = TFM(pil_r).unsqueeze(0).to(DEVICE)
    cam     = GradCAM(model=model, target_layers=[model.conv_head])
    heat    = cam(input_tensor=inp)[0]
    overlay = show_cam_on_image(rgb_arr, heat, use_rgb=True)
    return Image.fromarray(overlay)


# ═══════════════════════════════════════════════════════════════════════════════
# FORENSIC IMAGES
# ═══════════════════════════════════════════════════════════════════════════════
def compute_ela(pil_img: Image.Image, quality: int = 90, scale: float = 10.0):
    rgb = pil_img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=int(quality))
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(rgb, recompressed)
    return ImageEnhance.Contrast(diff).enhance(scale)


def compute_noise_residual(pil_img: Image.Image, sigma: float = 1.3):
    arr  = np.array(pil_img.convert("RGB"))
    blur = cv2.GaussianBlur(arr, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    resid = cv2.absdiff(arr, blur)
    resid = cv2.normalize(resid, None, 0, 255, cv2.NORM_MINMAX)
    return Image.fromarray(resid)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLANATION / REASONING
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
    q   = float(np.quantile(heat.reshape(-1), float(heat_thresh_q)))
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

    hot_total    = float(hot.sum() + 1e-6)
    face_total   = float(face_mask.sum() + 1e-6)
    hot_in_face  = float((hot * face_mask).sum())

    return {
        "faces_found":      int(len(face_boxes_norm)),
        "hot_overlap_ratio": float(np.clip(hot_in_face / hot_total, 0, 1)),
        "hot_face_share":    float(np.clip(hot_in_face / face_total, 0, 1)),
    }


def gradcam_focus_score(cam_pil: Image.Image):
    if cam_pil is None:
        return None
    arr = np.asarray(cam_pil).astype(np.float32) / 255.0
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    heat = np.clip(r - (0.5 * g + 0.5 * b), 0, 1)
    flat = heat.reshape(-1)
    if flat.size < 10:
        return None
    q80   = np.quantile(flat, 0.80)
    top   = flat[flat >= q80]
    total = float(flat.sum() + 1e-6)
    focus = float(np.clip(float(top.sum()) / total, 0, 1))
    q90   = np.quantile(flat, 0.90)
    peaks = float((flat >= q90).mean())
    return {"focus": focus, "peaks": peaks}


def build_explanation(label, conf, ai_p, real_p, cam_metrics, face_metrics):
    gap     = float(abs(ai_p - real_p))
    bullets = []

    if conf >= 0.90:
        bullets.append("High confidence result with strong probability separation.")
    elif conf >= DF_THRESHOLD:
        bullets.append("Moderate confidence result.")
    else:
        bullets.append("Low confidence result. Manual review is recommended.")

    bullets.append(f"Probability gap is {gap:.2%} between the two classes.")

    if cam_metrics is None:
        bullets.append("Grad-CAM attention summary is not available.")
    else:
        if cam_metrics["focus"] >= 0.62:
            bullets.append("Model attention is strongly concentrated on specific regions.")
        elif cam_metrics["focus"] >= 0.48:
            bullets.append("Model attention is moderately focused.")
        else:
            bullets.append("Model attention is spread across the image.")

    if face_metrics is None:
        bullets.append("Face-overlap reasoning is not available.")
    else:
        if face_metrics["faces_found"] == 0:
            bullets.append("No face detected. Decision may rely on image-wide artifacts.")
        else:
            overlap = face_metrics["hot_overlap_ratio"]
            if overlap >= 0.55:
                bullets.append("Most hot regions overlap the face area.")
            elif overlap >= 0.30:
                bullets.append("Attention is shared between face and background.")
            else:
                bullets.append("Most hot regions are outside the face area.")

    if label == "ai":
        bullets.append("AI-generated images often contain synthetic texture and blending artifacts.")
        action = "Recommended: verify with metadata, source, and additional samples."
    else:
        bullets.append("Real images often preserve natural sensor noise and detail transitions.")
        action = "Recommended: if this seems wrong, add similar AI samples during retraining."

    if conf < DF_THRESHOLD:
        action = "Recommended: treat as uncertain and review manually before using for training."

    return bullets, action


# ═══════════════════════════════════════════════════════════════════════════════
# REVIEW QUEUE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_image_hash(image_pil: Image.Image):
    return hashlib.md5(image_pil.tobytes()).hexdigest()


def load_review_log():
    if not REVIEW_LOG.exists():
        return []
    with open(REVIEW_LOG, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_review_log(rows):
    with open(REVIEW_LOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "filename", "original_name", "predicted_label",
            "confidence", "admin_final_label", "status", "timestamp",
        ])
        writer.writeheader()
        writer.writerows(rows)


def save_to_review_queue(image_pil, original_name, predicted_label, confidence):
    image_hash = compute_image_hash(image_pil)
    ext        = Path(original_name).suffix.lower() if Path(original_name).suffix else ".png"
    filename   = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{image_hash[:10]}{ext}"
    save_path  = REVIEW_QUEUE / filename

    # prevent exact duplicate by hash
    rows = load_review_log()
    for row in rows:
        if image_hash[:10] in row["filename"]:
            return row["filename"], False

    image_pil.save(save_path)

    file_exists = REVIEW_LOG.exists()
    with open(REVIEW_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "original_name", "predicted_label",
                             "confidence", "admin_final_label", "status", "timestamp"])
        writer.writerow([
            filename, original_name, predicted_label,
            f"{confidence:.6f}", "", "pending",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ])

    return filename, True


def get_pending_reviews():
    return [row for row in load_review_log() if row["status"] == "pending"]


def update_review_status(filename, final_label, status):
    rows = load_review_log()
    for row in rows:
        if row["filename"] == filename:
            row["admin_final_label"] = final_label
            row["status"] = status
    write_review_log(rows)


def move_review_image(filename, target_label):
    src = REVIEW_QUEUE / filename
    if not src.exists():
        return False
    dst = (VERIFIED_AI if target_label.lower() == "ai" else VERIFIED_REAL) / filename
    shutil.move(str(src), str(dst))
    return True


def reject_review_image(filename):
    src = REVIEW_QUEUE / filename
    if src.exists():
        shutil.move(str(src), str(REJECTED_DATA / filename))


def count_verified_images():
    ai_count   = len(list(VERIFIED_AI.glob("*")))
    real_count = len(list(VERIFIED_REAL.glob("*")))
    return ai_count, real_count, ai_count + real_count


def log_retrain_event(total_images, status):
    file_exists = RETRAIN_LOG.exists()
    with open(RETRAIN_LOG, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "verified_images_used", "status"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), total_images, status])


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


def build_pdf_report(case_id, verdict, conf, ai_p, real_p,
                     bullets, action, input_img, cam_img, ela_img, noise_img):
    if not REPORTLAB_AVAILABLE:
        return None

    buf = io.BytesIO()
    c   = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    y = H - 40

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, y, "DeepShield Enterprise — Forensic Report")
    y -= 18
    c.setFont("Helvetica", 9)
    c.drawString(40, y, f"Case ID: {case_id}")
    y -= 18
    c.drawString(40, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 22

    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, f"Verdict: {verdict}")
    y -= 16
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Confidence: {conf:.2%} | AI: {ai_p:.4f} | REAL: {real_p:.4f}")
    y -= 24

    c.setFont("Helvetica-Bold", 11)
    c.drawString(40, y, "Reasoning")
    y -= 16
    c.setFont("Helvetica", 10)
    for bullet in bullets[:6]:
        c.drawString(50, y, f"• {bullet[:100]}")
        y -= 14

    y -= 8
    c.drawString(40, y, f"Recommended Action: {action[:120]}")
    y -= 26

    def draw_img(pil_img, x, y_top, w=240, h=160, title=""):
        if pil_img is None:
            return
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y_top, title)
        data = pil_to_png_bytes(pil_img)
        c.drawImage(ImageReader(io.BytesIO(data)), x, y_top - h - 8,
                    width=w, height=h, preserveAspectRatio=True)

    draw_img(input_img, 40, y, title="Input Image")
    draw_img(cam_img,  300, y, title="Grad-CAM")
    y -= 190
    draw_img(ela_img,   40, y, title="ELA")
    draw_img(noise_img, 300, y, title="Noise Residual")

    c.showPage()
    c.save()
    buf.seek(0)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION  (FIX: removed ADMIN_ICON reference)
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        """
        <div style='padding:12px 0 8px;'>
          <div style='font-size:0.7rem;letter-spacing:0.12em;text-transform:uppercase;
                      color:#6f7c98;font-weight:700;margin-bottom:12px;'>Navigation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Select page",
        ["🔍 Prediction", "🔐 Admin Review", "🔄 Retrain"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style='border:1px solid rgba(255,255,255,0.07);border-radius:10px;padding:12px;
                    background:rgba(16,26,43,0.6);'>
          <div style='font-size:0.68rem;color:#6f7c98;text-transform:uppercase;
                      letter-spacing:0.08em;font-weight:700;margin-bottom:8px;'>System Info</div>
          <div style='font-size:0.78rem;color:#9aa7c4;line-height:1.8;'>
            Model: EfficientNet-B0<br/>
            Framework: PyTorch + timm<br/>
            Forensics: OpenCV + ELA<br/>
            XAI: Grad-CAM<br/>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not model_loaded:
        st.markdown(
            "<div style='margin-top:12px;padding:10px;border-radius:8px;"
            "background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.25);"
            "font-size:0.78rem;color:#f59e0b;'>"
            "⚠️ No trained model found at <code>outputs/best_model.pth</code>. "
            "Running with random weights.</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🔍 Prediction":
    col_left, col_right = st.columns([1.25, 1.0], gap="large")

    with col_left:
        st.markdown('<div class="card"><div class="card-title">Input &amp; Visual Evidence</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload image",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            label_visibility="collapsed",
            key="uploader",
        )

        pil_img = None
        w = h = 0
        kb = 0.0
        fmt = "-"

        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            sig        = (uploaded.name, len(file_bytes), hash(file_bytes))
            pil_img    = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            w, h       = pil_img.size
            kb         = len(file_bytes) / 1024
            fmt        = uploaded.type.split("/")[-1].upper() if uploaded.type else "IMAGE"

            if st.session_state.last_sig != sig:
                # Reset all state for new image
                st.session_state.last_sig             = sig
                st.session_state.blocked              = False
                st.session_state.spoof                = None
                st.session_state.df                   = None
                st.session_state.cam                  = None
                st.session_state.cam_metrics          = None
                st.session_state.face_boxes           = None
                st.session_state.face_metrics         = None
                st.session_state.ela                  = None
                st.session_state.noise                = None
                st.session_state.why_bullets          = None
                st.session_state.why_action           = None
                st.session_state.report_pdf           = None
                st.session_state.saved_to_review_queue = False
                st.session_state.saved_review_filename = None
                st.session_state.case_id              = datetime.now().strftime("%Y%m%d-%H%M%S")

                with st.spinner("Analyzing image…"):
                    time.sleep(0.05)

                    sr = detect_spoof(pil_img, threshold=SPOOF_THRESHOLD)
                    st.session_state.spoof = sr

                    if sr["is_spoof"]:
                        st.session_state.blocked = True
                    else:
                        label, conf, probs = run_deepfake(deepfake_model, pil_img)
                        st.session_state.df = {"label": label, "conf": conf, "probs": probs}

                        if ENABLE_GRADCAM and GRADCAM_AVAILABLE:
                            cam_img = run_gradcam(deepfake_model, pil_img)
                            st.session_state.cam         = cam_img
                            st.session_state.cam_metrics = gradcam_focus_score(cam_img)

                        st.session_state.face_boxes = detect_faces_bbox(pil_img)
                        if st.session_state.cam is not None:
                            st.session_state.face_metrics = face_overlap_score(
                                st.session_state.cam,
                                st.session_state.face_boxes or [],
                                heat_thresh_q=0.85,
                            )

                        st.session_state.ela   = compute_ela(pil_img, quality=90, scale=10.0)
                        st.session_state.noise = compute_noise_residual(pil_img, sigma=1.3)

                        ai_p   = float(probs[0])
                        real_p = float(probs[1])
                        st.session_state.why_bullets, st.session_state.why_action = build_explanation(
                            label=label, conf=float(conf),
                            ai_p=ai_p, real_p=real_p,
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
                    st.info("Grad-CAM disabled — spoof gate rejected the input.")
                elif not GRADCAM_AVAILABLE:
                    st.info("Grad-CAM not installed.  `pip install grad-cam`")
                elif st.session_state.cam is None:
                    st.info("Grad-CAM not available for this run.")
                else:
                    st.image(st.session_state.cam, use_container_width=True)

            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Forensic views</div>', unsafe_allow_html=True)

            tab1, tab2 = st.tabs(["📊 ELA", "🔬 Noise residual"])
            with tab1:
                if st.session_state.ela is not None:
                    st.image(st.session_state.ela, use_container_width=True)
                    st.caption("Error Level Analysis — brighter areas indicate potential manipulation.")
            with tab2:
                if st.session_state.noise is not None:
                    st.image(st.session_state.noise, use_container_width=True)
                    st.caption("Noise Residual — uneven patterns may indicate synthesis artifacts.")

            st.markdown(
                f"""
                <div class="meta">
                  <div class="m"><div class="l">Width</div><div class="v">{w}px</div></div>
                  <div class="m"><div class="l">Height</div><div class="v">{h}px</div></div>
                  <div class="m"><div class="l">Size</div><div class="v">{kb:.1f} KB</div></div>
                  <div class="m"><div class="l">Format</div><div class="v">{fmt}</div></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Spoof detail row
            if st.session_state.spoof:
                sp = st.session_state.spoof
                st.markdown(
                    f"""
                    <div class="meta" style="margin-top:8px;">
                      <div class="m"><div class="l">Blur</div><div class="v">{sp['blur']:.1f}</div></div>
                      <div class="m"><div class="l">Glare</div><div class="v">{sp['glare']:.4f}</div></div>
                      <div class="m"><div class="l">Moiré</div><div class="v">{sp['moire']:.4f}</div></div>
                      <div class="m"><div class="l">Border</div><div class="v">{sp['border']:.4f}</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("Upload an image to run spoof detection, deepfake prediction, Grad-CAM, and forensic views.")

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Right column — Verdict ─────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="card"><div class="card-title">Verdict</div>', unsafe_allow_html=True)

        if uploaded is None:
            st.markdown(
                """
                <div class="verdict">
                  <div style="font-size:0.68rem;color:#6f7c98;letter-spacing:0.1em;text-transform:uppercase;">Status</div>
                  <div style="font-size:1.55rem;font-weight:800;color:#e7eefc;margin-top:6px;">Awaiting input</div>
                  <div style="color:#9aa7c4;margin-top:8px;">Upload an image to view AI / Real classification.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if st.session_state.blocked:
                spoof_score = float(st.session_state.spoof["spoof_score"]) if st.session_state.spoof else 0.0
                st.markdown(
                    f"""
                    <div class="verdict" style="border-color:rgba(239,68,68,0.4);">
                      <div style="font-size:0.68rem;color:#6f7c98;letter-spacing:0.1em;text-transform:uppercase;">Status</div>
                      <div style="font-size:1.55rem;font-weight:800;color:#ef4444;margin-top:6px;">🚫 SPOOF DETECTED</div>
                      <div style="color:#9aa7c4;margin-top:8px;">
                        Spoof score: <strong style="color:#ef4444;">{spoof_score:.2%}</strong><br/>
                        Input rejected by the spoof gate. Deepfake analysis skipped.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                dfr    = st.session_state.df
                label  = dfr["label"]
                conf   = float(dfr["conf"])
                probs  = dfr["probs"]
                ai_p   = float(probs[0])
                real_p = float(probs[1])

                uncertain = conf < DF_THRESHOLD
                if uncertain:
                    verdict = "Uncertain"
                    color   = "#f59e0b"
                    icon    = "⚠️"
                elif label == "ai":
                    verdict = "AI-Generated"
                    color   = "#ef4444"
                    icon    = "🤖"
                else:
                    verdict = "Real Photograph"
                    color   = "#22c55e"
                    icon    = "✅"

                st.markdown(
                    f"""
                    <div class="verdict">
                      <div style="font-size:0.68rem;color:#6f7c98;letter-spacing:0.1em;text-transform:uppercase;">Result</div>
                      <div style="font-size:1.55rem;font-weight:800;color:{color};margin-top:6px;">{icon} {verdict}</div>
                      <div style="color:#9aa7c4;margin-top:8px;">Model confidence: <strong style="color:{color};">{conf:.2%}</strong></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Probability</div>', unsafe_allow_html=True)

                st.markdown(
                    f"""
                    <div style="margin-bottom:12px;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <div style="color:#9aa7c4;">🤖 AI / Deepfake</div>
                        <div class="mono" style="color:#e7eefc;">{ai_p:.6f}</div>
                      </div>
                      <div class="bar"><div class="fill" style="width:{ai_p*100:.2f}%;background:rgba(239,68,68,0.85);"></div></div>
                    </div>
                    <div style="margin-bottom:12px;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                        <div style="color:#9aa7c4;">✅ Real</div>
                        <div class="mono" style="color:#e7eefc;">{real_p:.6f}</div>
                      </div>
                      <div class="bar"><div class="fill" style="width:{real_p*100:.2f}%;background:rgba(34,197,94,0.85);"></div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                bullets = st.session_state.why_bullets or []
                action  = st.session_state.why_action or ""

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Why this result</div>', unsafe_allow_html=True)
                for bullet in bullets:
                    st.write(f"• {bullet}")
                st.info(action)

                fm = st.session_state.face_metrics
                if fm is not None:
                    st.caption(
                        f"Faces detected: {fm.get('faces_found', 0)} | "
                        f"Face attention overlap: {fm.get('hot_overlap_ratio', 0.0):.2%}"
                    )

                # Auto-save to review queue (once per image)
                if not st.session_state.saved_to_review_queue:
                    review_filename, is_new = save_to_review_queue(
                        image_pil=pil_img,
                        original_name=uploaded.name,
                        predicted_label=label.upper(),
                        confidence=conf,
                    )
                    st.session_state.saved_to_review_queue    = True
                    st.session_state.saved_review_filename    = review_filename
                    if is_new:
                        st.success(f"✔ Saved to review queue: `{review_filename}`")
                    else:
                        st.warning(f"Duplicate detected. Existing queue item: `{review_filename}`")

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                if REPORTLAB_AVAILABLE:
                    if st.button("📄 Generate PDF Report", use_container_width=True):
                        st.session_state.report_pdf = build_pdf_report(
                            case_id=st.session_state.case_id,
                            verdict=verdict, conf=conf,
                            ai_p=ai_p, real_p=real_p,
                            bullets=bullets, action=action,
                            input_img=pil_img,
                            cam_img=st.session_state.cam,
                            ela_img=st.session_state.ela,
                            noise_img=st.session_state.noise,
                        )
                    if st.session_state.report_pdf:
                        st.download_button(
                            "⬇ Download PDF Report",
                            st.session_state.report_pdf,
                            file_name=f"DeepShield_Report_{st.session_state.case_id}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.info("Install reportlab for PDF reports: `pip install reportlab`")

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ADMIN REVIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔐 Admin Review":
    st.markdown('<div class="card"><div class="card-title">Admin Review Queue</div>', unsafe_allow_html=True)

    if not st.session_state.admin_authenticated:
        pwd = st.text_input("Enter admin password", type="password", key="admin_pwd_input")
        if st.button("Unlock Admin Panel", use_container_width=True):
            if pwd == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # Authenticated
    col_hdr, col_logout = st.columns([4, 1])
    with col_logout:
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()

    pending_items = get_pending_reviews()
    all_rows      = load_review_log()

    # Stats row
    approved = sum(1 for r in all_rows if r["status"] == "approved")
    rejected = sum(1 for r in all_rows if r["status"] == "rejected")
    pending  = len(pending_items)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total in Log",  len(all_rows))
    c2.metric("Pending",       pending)
    c3.metric("Approved",      approved)
    c4.metric("Rejected",      rejected)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not pending_items:
        st.success("✅ No pending images in review queue.")
    else:
        selected_filename = st.selectbox(
            f"Select image to review ({pending} pending)",
            [item["filename"] for item in pending_items],
        )
        selected_item = next((item for item in pending_items if item["filename"] == selected_filename), None)

        if selected_item:
            image_path = REVIEW_QUEUE / selected_item["filename"]
            col1, col2 = st.columns([1.2, 1.0], gap="large")

            with col1:
                if image_path.exists():
                    st.image(str(image_path), caption=selected_item["original_name"], use_container_width=True)
                else:
                    st.error("⚠️ Image file not found in review_queue directory.")

            with col2:
                pred_label = selected_item["predicted_label"]
                pred_conf  = float(selected_item["confidence"])
                color_map  = {"AI": "#ef4444", "REAL": "#22c55e"}
                label_color = color_map.get(pred_label, "#9aa7c4")

                st.markdown(
                    f"""
                    <div style='border:1px solid rgba(255,255,255,0.08);border-radius:12px;
                                padding:14px;background:rgba(16,26,43,0.55);margin-bottom:12px;'>
                      <div style='font-size:0.68rem;color:#6f7c98;text-transform:uppercase;
                                  letter-spacing:0.08em;margin-bottom:6px;'>Model Prediction</div>
                      <div style='font-size:1.3rem;font-weight:800;color:{label_color};'>{pred_label}</div>
                      <div style='color:#9aa7c4;font-size:0.82rem;margin-top:4px;'>
                        Confidence: <strong style='color:{label_color};'>{pred_conf:.2%}</strong>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write(f"**Filename:** `{selected_item['filename']}`")
                st.write(f"**Original:** `{selected_item['original_name']}`")
                st.write(f"**Uploaded:** {selected_item['timestamp']}")

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Admin Decision</div>', unsafe_allow_html=True)
                decision = st.radio("Choose final label", ["AI", "REAL", "REJECT"], horizontal=True)

                if st.button("✔ Submit Review", use_container_width=True, type="primary"):
                    if decision in ["AI", "REAL"]:
                        ok = move_review_image(selected_item["filename"], decision)
                        if ok:
                            update_review_status(selected_item["filename"], decision, "approved")
                            st.success(f"Image approved as **{decision}** and moved to `verified_data/{decision.lower()}`")
                            st.rerun()
                        else:
                            st.error("Failed to move image file.")
                    else:
                        reject_review_image(selected_item["filename"])
                        update_review_status(selected_item["filename"], "REJECT", "rejected")
                        st.warning("Image moved to `rejected_data`.")
                        st.rerun()

    # Full log table
    if all_rows:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Full Review Log</div>', unsafe_allow_html=True)
        st.dataframe(
            all_rows,
            use_container_width=True,
            column_order=["filename", "original_name", "predicted_label",
                          "confidence", "admin_final_label", "status", "timestamp"],
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — RETRAIN
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔄 Retrain":
    st.markdown('<div class="card"><div class="card-title">Retrain Model</div>', unsafe_allow_html=True)

    if not st.session_state.retrain_authenticated:
        pwd = st.text_input("Enter admin password", type="password", key="retrain_pwd_input")
        if st.button("Unlock Retrain Panel", use_container_width=True):
            if pwd == ADMIN_PASSWORD:
                st.session_state.retrain_authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    col_hdr2, col_logout2 = st.columns([4, 1])
    with col_logout2:
        if st.button("🔒 Logout", use_container_width=True, key="retrain_logout"):
            st.session_state.retrain_authenticated = False
            st.rerun()

    ai_count, real_count, total_count = count_verified_images()

    c1, c2, c3 = st.columns(3)
    c1.metric("Verified AI",   ai_count)
    c2.metric("Verified REAL", real_count)
    c3.metric("Total Verified", total_count)

    threshold_ok = total_count >= 20

    if threshold_ok:
        st.success(f"✅ {total_count} verified images available. Ready to retrain.")
    else:
        st.warning(f"⚠️ Only {total_count} verified images. Recommend at least 20 (ideally 50+).")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("### Retrain Workflow")
    st.markdown("""
1. Verify samples in **Admin Review** (mark as AI or REAL).
2. Verified samples are stored in `verified_data/ai` and `verified_data/real`.
3. Click **Start Retraining** — this runs `retrain_model.py`.
4. Update `retrain_model.py` with your full training pipeline.
5. A new model checkpoint will replace `outputs/best_model.pth`.
""")

    # Show retrain log if it exists
    if RETRAIN_LOG.exists():
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Retrain History</div>', unsafe_allow_html=True)
        with open(RETRAIN_LOG, "r", encoding="utf-8") as f:
            retrain_rows = list(csv.DictReader(f))
        if retrain_rows:
            st.dataframe(retrain_rows, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if not threshold_ok:
        st.button("🚫 Start Retraining (need more data)", disabled=True, use_container_width=True)
    else:
        if st.button("🚀 Start Retraining", use_container_width=True, type="primary"):
            if not RETRAIN_SCRIPT.exists():
                log_retrain_event(total_count, "failed_missing_retrain_script")
                st.error("`retrain_model.py` not found in the project root.")
            else:
                with st.spinner("Running retraining script… this may take a while."):
                    try:
                        result = subprocess.run(
                            ["python", str(RETRAIN_SCRIPT)],
                            capture_output=True,
                            text=True,
                            timeout=3600,
                        )
                        if result.returncode == 0:
                            log_retrain_event(total_count, "success")
                            st.success("✅ Retraining completed successfully.")
                            if result.stdout.strip():
                                st.code(result.stdout[:4000])
                        else:
                            log_retrain_event(total_count, "failed_runtime_error")
                            st.error("Retraining script exited with an error.")
                            if result.stderr.strip():
                                st.code(result.stderr[:4000])
                    except subprocess.TimeoutExpired:
                        log_retrain_event(total_count, "failed_timeout")
                        st.error("Retraining timed out after 1 hour.")
                    except Exception as e:
                        log_retrain_event(total_count, "failed_exception")
                        st.error(f"Retraining failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"""
<div class="footer">
  <div>
    DeepShield Enterprise · Deepfake &amp; Spoof Attack Detection in Cyber Threats<br/>
    B.Tech Final Year Project · AI &amp; Data Science · St. Joseph College of Engineering
  </div>
  <div style="text-align:right;">
    EfficientNet-B0 · PyTorch · OpenCV · Grad-CAM · ELA<br/>
    Device: {DEVICE.upper()} · v4.1
  </div>
</div>
""",
    unsafe_allow_html=True,
)