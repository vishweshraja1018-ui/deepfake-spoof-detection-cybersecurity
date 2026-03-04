import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# PATHS (YOUR SYSTEM)
# -----------------------------
PROJECT_DIR = r"C:\Users\vishw\Documents\ai_image_detector"
TRAIN_DIR = os.path.join(PROJECT_DIR, "dataset", "train")
VAL_DIR   = os.path.join(PROJECT_DIR, "dataset", "val")
TEST_DIR  = os.path.join(PROJECT_DIR, "dataset", "test")

os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS_STAGE1 = 6
EPOCHS_STAGE2 = 6
MODEL_NAME = "efficientnetb0"   # change to "resnet50" if you want
OUT_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "transfer_model.keras")

# -----------------------------
# LOAD DATA
# -----------------------------
train_ds = image_dataset_from_directory(
    TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=True, seed=42
)

val_ds = image_dataset_from_directory(
    VAL_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=True, seed=42
)

test_ds = image_dataset_from_directory(
    TEST_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    label_mode="binary", shuffle=False
)

print("Class names:", train_ds.class_names)
CLASS_NAMES = train_ds.class_names
# Expect: ['ai', 'real']

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds   = val_ds.cache().prefetch(AUTOTUNE)
test_ds  = test_ds.cache().prefetch(AUTOTUNE)

# -----------------------------
# AUGMENTATION
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])

# -----------------------------
# BASE MODEL
# -----------------------------
if MODEL_NAME.lower() == "resnet50":
    preprocess = tf.keras.applications.resnet50.preprocess_input
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
    )
elif MODEL_NAME.lower() == "efficientnetb0":
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,)
    )
else:
    raise ValueError("MODEL_NAME must be 'resnet50' or 'efficientnetb0'")

base_model.trainable = False

# -----------------------------
# BUILD MODEL
# -----------------------------
inputs = layers.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3, min_lr=1e-6),
    ModelCheckpoint(OUT_MODEL_PATH, save_best_only=True),
]

# -----------------------------
# TRAIN STAGE 1
# -----------------------------
print("\n=== Stage 1: Training top layers (base frozen) ===")
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE1, callbacks=callbacks)

# -----------------------------
# TRAIN STAGE 2 (FINE-TUNE)
# -----------------------------
print("\n=== Stage 2: Fine-tuning last layers ===")
base_model.trainable = True

# Fine-tune only last layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_STAGE2, callbacks=callbacks)

# -----------------------------
# FINAL EVALUATION ON TEST
# -----------------------------
print("\n=== Final Test Evaluation ===")
y_true, y_pred = [], []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0).ravel()
    y_true.extend(labels.numpy().ravel())
    y_pred.extend((probs >= 0.5).astype(int))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

print(f"\n✅ Model saved at: {OUT_MODEL_PATH}")