import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf

PROJECT_DIR = r"C:\Users\vishw\Documents\ai_image_detector"
NEW_TEST_DIR = os.path.join(PROJECT_DIR, "new_test")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "transfer_model.keras")

IMG_SIZE = (224, 224)

# 1) Check folder exists
if not os.path.exists(NEW_TEST_DIR):
    raise FileNotFoundError(f"❌ new_test folder not found: {NEW_TEST_DIR}")

# 2) Find images recursively (includes subfolders)
patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG", "*.webp", "*.WEBP"]
image_paths = []
for p in patterns:
    image_paths.extend(glob.glob(os.path.join(NEW_TEST_DIR, "**", p), recursive=True))

print(f"✅ Found {len(image_paths)} image(s) inside: {NEW_TEST_DIR}")

if len(image_paths) == 0:
    print("❌ No images found. Put images directly inside new_test/ OR check file formats.")
    exit()

# 3) Load model
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)

results = []
skipped = 0

for img_path in image_paths:
    try:
        # load image safely using keras (better than cv2 for random formats)
        img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # (1,224,224,3)

        pred = float(model.predict(x, verbose=0)[0][0])

        label = "REAL" if pred >= 0.5 else "AI"
        confidence = pred if pred >= 0.5 else (1 - pred)

        results.append({
            "file_path": img_path,
            "file_name": os.path.basename(img_path),
            "prediction": label,
            "confidence": round(confidence, 4),
            "raw_score": round(pred, 4)
        })

        print(f"{os.path.basename(img_path)} → {label} (conf={confidence:.4f}, score={pred:.4f})")

    except Exception as e:
        skipped += 1
        print(f"⚠ Skipped {img_path} (error: {e})")

# 4) Save CSV
df = pd.DataFrame(results)
csv_path = os.path.join(PROJECT_DIR, "new_test_results.csv")
df.to_csv(csv_path, index=False)

print("\n✅ Saved CSV:", csv_path)
print(f"✅ Total predicted: {len(results)} | Skipped: {skipped}")