import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ✅ CHANGE THESE PATHS (your dataset folders)
TRAIN_DIR = r"C:\Users\vishw\Documents\ai_image_detector\dataset\train"
VAL_DIR   = r"C:\Users\vishw\Documents\ai_image_detector\dataset\val"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

def make_datasets():
    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=SEED,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR,
        label_mode="binary",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    # speed
    train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds

def build_model():
    data_aug = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

    base = keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    base.trainable = False  # first train only the head

    inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = data_aug(inputs)
    x = keras.applications.efficientnet.preprocess_input(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )
    return model, base

def main():
    train_ds, val_ds = make_datasets()
    model, base = build_model()

    callbacks = [
        keras.callbacks.ModelCheckpoint("best_efficientnet.keras", save_best_only=True, monitor="val_auc", mode="max"),
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max"),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_loss"),
    ]

    # ✅ Phase 1: train head
    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

    # ✅ Phase 2: fine-tune last layers (boost accuracy)
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)
    model.save("final_efficientnet.keras")
    print("✅ Saved: best_efficientnet.keras and final_efficientnet.keras")

if __name__ == "__main__":
    main()