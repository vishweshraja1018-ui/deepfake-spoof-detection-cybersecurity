import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------
# CONFIG
# -----------------------------
PROJECT_DIR = r"C:\Users\vishw\Documents\ai_image_detector"
TRAIN_DIR = os.path.join(PROJECT_DIR, "dataset", "train")
TEST_DIR  = os.path.join(PROJECT_DIR, "dataset", "test")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_STAGE1 = 8      # train only top layers
EPOCHS_STAGE2 = 8      # fine-tune last layers
MODEL_NAME = "resnet50"  # change to "efficientnetb0" if you want

OUT_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "transfer_model.keras")
os.makedirs(os.path.join(PROJECT_DIR, "models"), exist_ok=True)

# -----------------------------
# 1) LOAD DATASETS
# -----------------------------
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=42
)

test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# -----------------------------
# 2) DATA AUGMENTATION
# -----------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.15),
])

# -----------------------------
# 3) BASE MODEL (TRANSFER LEARNING)
# -----------------------------
if MODEL_NAME.lower() == "resnet50":
    preprocess = tf.keras.applications.resnet50.preprocess_input
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )
elif MODEL_NAME.lower() == "efficientnetb0":
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=IMG_SIZE + (3,)
    )
else:
    raise ValueError("MODEL_NAME must be 'resnet50' or 'efficientnetb0'")

base_model.trainable = False  # Stage 1: freeze

# -----------------------------
# 4) BUILD MODEL HEAD
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
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -----------------------------
# 5) CALLBACKS
# -----------------------------
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ReduceLROnPlateau(patience=2, factor=0.3, min_lr=1e-6),
    ModelCheckpoint(OUT_MODEL_PATH, save_best_only=True)
]

# -----------------------------
# 6) TRAIN STAGE 1 (FROZEN BASE)
# -----------------------------
print("\n=== Stage 1: Training top layers (base frozen) ===")
hist1 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_STAGE1,
    callbacks=callbacks
)

# -----------------------------
# 7) TRAIN STAGE 2 (FINE-TUNING)
# -----------------------------
print("\n=== Stage 2: Fine-tuning last layers ===")
base_model.trainable = True

# Fine-tune only the last ~30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

hist2 = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks
)

# -----------------------------
# 8) EVALUATION
# -----------------------------
print("\n=== Final Evaluation ===")
y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images, verbose=0).ravel()
    y_true.extend(labels.numpy().ravel())
    y_pred.extend((preds >= 0.5).astype(int))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["fake", "real"]))

print(f"\nSaved model to: {OUT_MODEL_PATH}")

# -----------------------------
# 9) PLOT ACCURACY
# -----------------------------
def plot_history(h, title):
    plt.figure()
    plt.plot(h.history.get("accuracy", []), label="train_acc")
    plt.plot(h.history.get("val_accuracy", []), label="val_acc")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()

plot_history(hist1, "Stage 1 Accuracy")
plot_history(hist2, "Stage 2 Accuracy")