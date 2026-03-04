import os
from pathlib import Path
from PIL import Image

# Change these if needed
INPUT_DIR = Path("new_test")
OUTPUT_DIR = Path("new_test_jpg")
MAX_SIZE = 512                   # resize keeping aspect ratio (set None to skip)
JPEG_QUALITY = 92

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

def convert_one(src: Path, dst: Path):
    try:
        img = Image.open(src).convert("RGB")

        # Resize (keeps aspect ratio)
        if MAX_SIZE is not None:
            img.thumbnail((MAX_SIZE, MAX_SIZE))

        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return True
    except Exception as e:
        print(f"❌ Failed: {src} -> {e}")
        return False

def main():
    if not INPUT_DIR.exists():
        print(f"❌ INPUT_DIR not found: {INPUT_DIR.resolve()}")
        return

    total = 0
    converted = 0
    skipped = 0

    for root, _, files in os.walk(INPUT_DIR):
        root_path = Path(root)

        for f in files:
            src = root_path / f
            ext = src.suffix.lower()

            # Skip non-images (like .mp4)
            if ext not in IMG_EXTS:
                skipped += 1
                continue

            # Keep same relative folder structure
            rel = src.relative_to(INPUT_DIR)
            dst = OUTPUT_DIR / rel
            dst = dst.with_suffix(".jpg")  # force jpg

            total += 1
            if convert_one(src, dst):
                converted += 1

    print("\n✅ DONE")
    print(f"Input:  {INPUT_DIR.resolve()}")
    print(f"Output: {OUTPUT_DIR.resolve()}")
    print(f"Images processed: {converted}/{total}")
    print(f"Skipped non-images: {skipped}")

if __name__ == "__main__":
    main()
