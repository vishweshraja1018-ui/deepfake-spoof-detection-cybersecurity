import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# -----------------------------
# PATHS
# -----------------------------
PROJECT_DIR = r"C:\Users\vishw\Documents\ai_image_detector"
NEW_TEST_DIR = os.path.join(PROJECT_DIR, "new_test")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "transfer_model.keras")

IMG_SIZE = (224, 224)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

print("✅ Model loaded successfully")

# -----------------------------
# PREDICT IMAGES
# -----------------------------
results = []

for filename in os.listdir(NEW_TEST_DIR):
    file_path = os.path.join(NEW_TEST_DIR, filename)

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        img = cv2.imread(file_path)
        img = cv2.resize(img, IMG_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img, verbose=0)[0][0]

        label = "REAL" if prediction > 0.5 else "AI (Deepfake)"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        results.append({
            "filename": filename,
            "prediction": label,
            "confidence": round(float(confidence), 4)
        })

        print(f"{filename} → {label} (Confidence: {confidence:.4f})")

# -----------------------------
# SAVE RESULTS
# -----------------------------
df = pd.DataFrame(results)
csv_path = os.path.join(PROJECT_DIR, "new_test_results.csv")
df.to_csv(csv_path, index=False)

print("\n✅ Results saved to:", csv_path)
print("\n🎯 Prediction completed.")