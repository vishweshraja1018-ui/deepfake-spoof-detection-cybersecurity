import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_DIR = r"C:\Users\vishw\Documents\ai_image_detector"
TEST_DIR  = os.path.join(PROJECT_DIR, "dataset", "test")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "transfer_model.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16

test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

CLASS_NAMES = test_ds.class_names
print("Classes:", CLASS_NAMES)

model = tf.keras.models.load_model(MODEL_PATH)

y_true, y_pred = [], []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0).ravel()
    y_true.extend(labels.numpy().ravel())
    y_pred.extend((probs >= 0.5).astype(int))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print("\n✅ Evaluation Done.")