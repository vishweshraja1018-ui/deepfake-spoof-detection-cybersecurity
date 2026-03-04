import os, glob
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

PROJECT_DIR = r"C:\Users\vishw\Documents\ai_image_detector"
NEW_TEST_DIR = os.path.join(PROJECT_DIR, "new_test")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "transfer_model.keras")
IMG_SIZE = (224, 224)

# Haar face detector (built-in, no extra install)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

model = tf.keras.models.load_model(MODEL_PATH)

patterns = ["*.jpg","*.jpeg","*.png","*.webp","*.JPG","*.JPEG","*.PNG","*.WEBP"]
image_paths = []
for p in patterns:
    image_paths += glob.glob(os.path.join(NEW_TEST_DIR, "**", p), recursive=True)

print("Found:", len(image_paths), "images")

results = []
for path in image_paths:
    bgr = cv2.imread(path)
    if bgr is None:
        continue

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # If face found: crop the largest face
    if len(faces) > 0:
        x,y,w,h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
        face = bgr[y:y+h, x:x+w]
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        used = "face_crop"
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        used = "full_image"

    rgb = cv2.resize(rgb, IMG_SIZE)
    x_in = np.expand_dims(rgb.astype(np.float32), axis=0)

    score = float(model.predict(x_in, verbose=0)[0][0])  # score ~1 => REAL, ~0 => AI
    pred = "REAL" if score >= 0.5 else "AI"
    conf = score if pred == "REAL" else (1-score)

    results.append({
        "file": os.path.basename(path),
        "used": used,
        "prediction": pred,
        "confidence": round(conf, 4),
        "raw_score": round(score, 4)
    })

df = pd.DataFrame(results)
out_csv = os.path.join(PROJECT_DIR, "new_test_results_facecrop.csv")
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)
print(df.head(10))