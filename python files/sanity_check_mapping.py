import os, random
import numpy as np
import tensorflow as tf

PROJECT_DIR = r"C:\Users\vishw\Documents\ai_image_detector"
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "transfer_model.keras")

AI_DIR = os.path.join(PROJECT_DIR, "dataset", "test", "ai")
REAL_DIR = os.path.join(PROJECT_DIR, "dataset", "test", "real")

IMG_SIZE = (224, 224)

def predict_one(path):
    img = tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    p = float(model.predict(x, verbose=0)[0][0])
    return p

model = tf.keras.models.load_model(MODEL_PATH)

ai_sample = random.choice([os.path.join(AI_DIR, f) for f in os.listdir(AI_DIR) if f.lower().endswith(("jpg","jpeg","png","webp"))])
real_sample = random.choice([os.path.join(REAL_DIR, f) for f in os.listdir(REAL_DIR) if f.lower().endswith(("jpg","jpeg","png","webp"))])

p_ai = predict_one(ai_sample)
p_real = predict_one(real_sample)

print("AI sample:", os.path.basename(ai_sample), "score=", round(p_ai,4))
print("REAL sample:", os.path.basename(real_sample), "score=", round(p_real,4))

print("\nInterpretation (based on your training classes ['ai','real']):")
print("score near 0.0 => AI")
print("score near 1.0 => REAL")