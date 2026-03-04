import cv2
import shutil
from spoof_detector import detect_spoof
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm

MODEL_NAME = "efficientnet_b0"
MODEL_PATH = Path(r"C:\Users\vishw\Documents\ai_image_detector\outputs\best_model.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

CLASS_NAMES = ["ai", "real"]
VALID_EXTS = {".jpg", ".jpeg", ".png"}


tfm = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=2)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


def predict_one(model, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu()
    pred_idx = int(torch.argmax(probs))
    return pred_idx, float(probs[pred_idx])


def iter_images(folder: Path):
    if not folder.exists():
        return []
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in VALID_EXTS]


def main():
    # ✅ IMPORTANT: This must match your folder exactly
    base = Path(r"C:\Users\vishw\Documents\ai_image_detector\new_test_jpg")

    # ✅ Where spoof images are saved
    spoof_output_dir = Path(r"C:\Users\vishw\Documents\ai_image_detector\outputs\spoof_detected")
    spoof_output_dir.mkdir(parents=True, exist_ok=True)

    print("BASE PATH:", base)
    print("Exists?:", base.exists())
    if not base.exists():
        print("❌ Base folder not found. Check folder name/path.")
        return

    model = load_model()
    print("✅ Model loaded from:", MODEL_PATH)
    print("✅ Running on:", DEVICE)

    total = 0
    correct = 0
    spoof_skipped = 0
    unreadable = 0

    for label_name in ["ai", "real"]:
        folder = base / label_name
        print(f"\nTesting folder: {folder} (true={label_name})")
        print("Folder exists?:", folder.exists())

        files = iter_images(folder)
        print("Images found:", len(files))

        if not files:
            continue

        true_idx = CLASS_NAMES.index(label_name)

        for img_path in files:
            # ---------------- SPOOF CHECK (START) ----------------
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                unreadable += 1
                print(f"⚠️ Cannot read image: {img_path.name}")
                continue

            spoof = detect_spoof(img_bgr)
            if spoof["is_spoof"]:
                spoof_skipped += 1

                # Save spoof image
                dest_path = spoof_output_dir / img_path.name
                try:
                    shutil.copy(str(img_path), str(dest_path))
                except Exception as e:
                    print(f"⚠️ Could not copy spoof image: {img_path.name} | {e}")

                print(f"🚨 SPOOF: {img_path.name} | score={spoof['spoof_score']:.2f} | reasons={spoof['reasons']}")
                continue
            # -----------------------------------------------------

            # Deepfake / AI detection
            pred_idx, conf = predict_one(model, img_path)
            pred_label = CLASS_NAMES[pred_idx]

            total += 1
            if pred_idx == true_idx:
                correct += 1
                mark = "✅"
            else:
                mark = "❌"

            print(f"{mark} {img_path.name} -> pred={pred_label} conf={conf:.3f}")

    acc = correct / total if total > 0 else 0.0

    print("\n---------------- FINAL SUMMARY ----------------")
    print(f"Spoof images detected & saved: {spoof_skipped}")
    print(f"Unreadable images skipped: {unreadable}")
    print(f"Images tested (non-spoof): {total}")
    print(f"Accuracy (non-spoof images): {acc:.4f} ({correct}/{total})")
    print(f"Spoof images stored at: {spoof_output_dir}")


if __name__ == "__main__":
    main()