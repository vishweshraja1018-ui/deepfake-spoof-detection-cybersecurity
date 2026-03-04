import os
from pathlib import Path
from PIL import Image

DATASET = Path("dataset")

# output folder
OUT = Path("dataset_clean")
SPLITS = ["train", "val", "test"]
CLASSES = ["ai", "real"]

OUT.mkdir(exist_ok=True)

MAX_SIZE = 512      # resize keeping aspect ratio
JPEG_QUALITY = 92   # good quality but standardizes compression

def process_image(src_path: Path, dst_path: Path):
    try:
        img = Image.open(src_path).convert("RGB")
        img.thumbnail((MAX_SIZE, MAX_SIZE))  # keeps aspect ratio
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path.with_suffix(".jpg"), "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception as e:
        print(f"❌ Skip: {src_path} ({e})")
        return False

def main():
    total = 0
    kept = 0

    for split in SPLITS:
        for cls in CLASSES:
            src_dir = DATASET / split / cls
            if not src_dir.exists():
                print(f"⚠️ Missing: {src_dir}")
                continue

            for fname in os.listdir(src_dir):
                src_path = src_dir / fname

                # skip non-images safely
                if src_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
                    print(f"⏭️ Not an image, skipping: {src_path.name}")
                    continue

                dst_path = OUT / split / cls / (src_path.stem + ".jpg")
                total += 1
                if process_image(src_path, dst_path):
                    kept += 1

    print(f"\n✅ Done. Processed: {kept}/{total} images")
    print(f"✅ New dataset folder: {OUT.resolve()}")

if __name__ == "__main__":
    main()
