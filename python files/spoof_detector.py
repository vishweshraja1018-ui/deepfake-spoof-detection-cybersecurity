import cv2
import numpy as np
import spoof_detector
print("USING spoof_detector FROM:", spoof_detector.__file__)        

def _blur_score(gray: np.ndarray) -> float:
    # Higher = sharper, Lower = more blur
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def _glare_score(img_bgr: np.ndarray) -> float:
    # Glare often produces large bright saturated regions
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    bright = (v > 245).astype(np.uint8)
    ratio = float(bright.mean())  # 0 to 1
    return ratio

def _moiré_score(gray: np.ndarray) -> float:
    # Approx: high-frequency energy ratio using FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    mag = np.log(np.abs(fshift) + 1.0)

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    # low frequency center window
    r = int(min(h, w) * 0.08)
    low = mag[cy - r: cy + r, cx - r: cx + r]

    total_energy = float(mag.mean())
    low_energy = float(low.mean())
    high_energy = max(total_energy - low_energy, 0.0)

    # ratio-like value
    return float(high_energy / (total_energy + 1e-6))

def _border_rect_score(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)

    # Convert edges from {0,255} to {0,1}
    edges = (edges > 0).astype(np.float32)

    h, w = edges.shape
    m = int(min(h, w) * 0.06)  # border margin

    top = float(edges[:m, :].mean())
    bottom = float(edges[h-m:, :].mean())
    left = float(edges[:, :m].mean())
    right = float(edges[:, w-m:].mean())

    border_strength = (top + bottom + left + right) / 4.0
    return float(border_strength)  # now 0..1

def detect_spoof(image_bgr: np.ndarray) -> dict:
    """
    Returns:
      {
        "is_spoof": bool,
        "spoof_score": float (0..1),
        "reasons": [str],
        "signals": { ... raw scores ... }
      }
    """
    if image_bgr is None or image_bgr.size == 0:
        return {"is_spoof": True, "spoof_score": 1.0, "reasons": ["Invalid image"], "signals": {}}

    # Resize for stable scoring
    img = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = _blur_score(gray)              # e.g. < 60 often blurred
    glare = _glare_score(img)             # > 0.02 can be glare
    moire = _moiré_score(gray)            # higher means more HF energy
    border = _border_rect_score(gray)     # higher means more edge at borders

    reasons = []
    spoof_score = 0.0

    # --- Rules (prototype-friendly thresholds) ---
    # 1) Blur
    if blur < 60:
        reasons.append(f"High blur (blur_score={blur:.1f})")
        spoof_score += 0.30
    elif blur < 120:
        reasons.append(f"Medium blur (blur_score={blur:.1f})")
        spoof_score += 0.15

    # 2) Glare
    if glare > 0.015:
        reasons.append(f"Possible screen glare (glare_ratio={glare:.3f})")
        spoof_score += 0.20

    # 3) Moiré / recapture-ish high frequency pattern
    if moire > 0.45:
        reasons.append(f"Moiré-like high-frequency patterns (moire_score={moire:.2f})")
        spoof_score += 0.25

    # 4) Screen border hints
    if border > 0.10:
        reasons.append(f"Strong border edges (border_score={border:.3f})")
        spoof_score += 0.25

    spoof_score = float(min(max(spoof_score, 0.0), 1.0))
    is_spoof = spoof_score >= 0.45  # decision threshold

    return {
        "is_spoof": bool(is_spoof),
        "spoof_score": spoof_score,
        "reasons": reasons if reasons else ["No strong spoof signals detected"],
        "signals": {
            "blur_score": blur,
            "glare_ratio": glare,
            "moire_score": moire,
            "border_score": border,
        }
    }