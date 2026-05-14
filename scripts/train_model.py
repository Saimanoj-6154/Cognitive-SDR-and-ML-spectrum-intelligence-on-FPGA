import os
import sys
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"       # suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

FEAT_DIR  = "dataset/features"
MODEL_DIR = "ml/saved_models"
PLOT_DIR  = "results/plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

NUM_CLASSES  = 6
EPOCHS       = 50
BATCH_SIZE   = 32
LR           = 0.001
VAL_SPLIT    = 0.15
TEST_SPLIT   = 0.15
SEED         = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ── BUILD MODEL ───────────────────────────────────────────────
def build_model(input_dim: int = 8, num_classes: int = 6) -> keras.Model:
    """
    Small Dense NN for channel quality classification.
    Lightweight by design — suitable for soft-core CPU on FPGA PS.
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name="spectral_features"),

        layers.Dense(64, activation="relu", name="dense_1"),
        layers.BatchNormalization(name="bn_1"),
        layers.Dropout(0.30, name="drop_1"),

        layers.Dense(32, activation="relu", name="dense_2"),
        layers.BatchNormalization(name="bn_2"),
        layers.Dropout(0.20, name="drop_2"),

        layers.Dense(num_classes, activation="softmax", name="channel_quality"),
    ], name="cognitive_radio_nn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── PLOT TRAINING HISTORY ─────────────────────────────────────
def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(history.history["loss"],     label="Train loss",  color="#185FA5")
    axes[0].plot(history.history["val_loss"], label="Val loss",    color="#E24B4A", linestyle="--")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"],     label="Train acc",  color="#1D9E75")
    axes[1].plot(history.history["val_accuracy"], label="Val acc",    color="#BA7517", linestyle="--")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training & Validation Accuracy")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Dense NN Training — Cognitive Radio Spectrum Allocation", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training plot → {save_path}")


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Dense NN Training — TensorFlow/Keras")
    print("=" * 60)

    # Load features and labels
    X = np.load(os.path.join(FEAT_DIR, "features.npy"))
    y = np.load(os.path.join(FEAT_DIR, "labels.npy"))

    print(f"  Features : {X.shape}  (dtype={X.dtype})")
    print(f"  Labels   : {y.shape}  classes={np.unique(y)}")

    # One-hot encode
    lb = LabelBinarizer()
    Y  = lb.fit_transform(y)
    if Y.shape[1] == 1:                 # binary edge case
        Y = np.hstack([1 - Y, Y])

    # Train / Val / Test split
    X_trainval, X_test, Y_trainval, Y_test, y_trainval, y_test = (
        train_test_split(X, Y, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y)
    )
    X_train, X_val, Y_train, Y_val = (
        train_test_split(X_trainval, Y_trainval, test_size=VAL_SPLIT / (1 - TEST_SPLIT),
                         random_state=SEED)
    )

    print(f"\n  Train : {X_train.shape[0]} samples")
    print(f"  Val   : {X_val.shape[0]} samples")
    print(f"  Test  : {X_test.shape[0]} samples")

    # Build model
    model = build_model(input_dim=X.shape[1], num_classes=NUM_CLASSES)
    model.summary()
    print()

    # Callbacks
    ckpt_path = os.path.join(MODEL_DIR, "best_model.keras")
    cb_list = [
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_accuracy",
                                  save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=5, min_lr=1e-5, verbose=1),
    ]

    print("  Training …")
    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb_list,
        verbose=1,
    )

    # Final test evaluation
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\n  ✓ Test loss     : {loss:.4f}")
    print(f"  ✓ Test accuracy : {acc * 100:.2f}%")

    # Save final model + weights
    model.save(os.path.join(MODEL_DIR, "cognitive_radio_model.keras"))
    model.save_weights(os.path.join(MODEL_DIR, "weights.h5"))

    # Save training metadata
    meta = {
        "test_accuracy"    : float(acc),
        "test_loss"        : float(loss),
        "epochs_trained"   : len(history.history["loss"]),
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "architecture"     : "Dense(64,ReLU)->Dense(32,ReLU)->Dense(6,Softmax)",
        "total_params"     : int(model.count_params()),
        "num_classes"      : NUM_CLASSES,
        "input_features"   : int(X.shape[1]),
    }
    with open(os.path.join(MODEL_DIR, "training_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Save history
    np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(MODEL_DIR, "Y_test.npy"), Y_test)
    np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)

    # Plot
    plot_history(history, os.path.join(PLOT_DIR, "training_history.png"))

    print(f"\n  Model saved → {MODEL_DIR}/")
    print(f"  Total params : {model.count_params():,}")
    print("  Done!\n")


if __name__ == "__main__":
    main()
