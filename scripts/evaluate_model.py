import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)

MODEL_DIR = "ml/saved_models"
PLOT_DIR  = "results/plots"
LOG_DIR   = "results/logs"
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

NUM_CHANNELS  = 6
CHANNEL_NAMES = [f"Ch{i+1}" for i in range(NUM_CHANNELS)]
FREQ_LABELS   = ["2412","2422","2432","2437","2447","2462"]


# ── LOAD MODEL + DATA ─────────────────────────────────────────
def load_artifacts():
    model  = tf.keras.models.load_model(
        os.path.join(MODEL_DIR, "cognitive_radio_model.keras"))
    X_test = np.load(os.path.join(MODEL_DIR, "X_test.npy"))
    Y_test = np.load(os.path.join(MODEL_DIR, "Y_test.npy"))
    y_test = np.load(os.path.join(MODEL_DIR, "y_test.npy"))
    return model, X_test, Y_test, y_test


# ── CONFUSION MATRIX ─────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
                linewidths=.5, ax=ax, vmin=0, vmax=1,
                annot_kws={"size": 10})
    ax.set_xlabel("Predicted Channel", fontsize=11)
    ax.set_ylabel("True Best Channel", fontsize=11)
    ax.set_title("Confusion Matrix — Channel Quality Classification", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── PER-CHANNEL ACCURACY ─────────────────────────────────────
def plot_per_channel_accuracy(y_true, y_pred, save_path):
    per_ch_acc = []
    for ch in range(NUM_CHANNELS):
        mask = y_true == ch
        if mask.sum() == 0:
            per_ch_acc.append(0.0)
        else:
            per_ch_acc.append(accuracy_score(y_true[mask], y_pred[mask]))

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#1D9E75" if a >= 0.90 else "#EF9F27" if a >= 0.80
              else "#E24B4A" for a in per_ch_acc]
    bars = ax.bar(CHANNEL_NAMES, [a * 100 for a in per_ch_acc],
                  color=colors, edgecolor="white", linewidth=0.5)
    for bar, acc in zip(bars, per_ch_acc):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{acc*100:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("WiFi Channel")
    ax.set_title("Per-Channel Classification Accuracy")
    ax.set_ylim(0, 108)
    ax.axhline(90, color="#185FA5", linestyle="--", linewidth=1, label="90% threshold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add frequency labels
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(range(NUM_CHANNELS))
    ax2.set_xticklabels([f"{f} MHz" for f in FREQ_LABELS], fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── ALLOCATION PERFORMANCE ────────────────────────────────────
def plot_allocation_performance(model, save_path):
    """
    Simulate 200 allocation cycles and compare:
      - Random allocation   (baseline)
      - Energy-detect only  (sensing only)
      - Dense NN allocation (proposed)
    """
    from scripts.extract_features import extract_feature_vector
    from simulation.sensing.energy_detection import detect_occupancy
    from scripts.generate_dataset import generate_single_iq, iq_to_psd

    np.random.seed(0)
    N_TRIALS = 200

    random_correct, sensing_correct, nn_correct = 0, 0, 0

    for _ in range(N_TRIALS):
        true_occ  = [np.random.rand() > 0.45 for _ in range(NUM_CHANNELS)]
        free_chs  = [i for i, b in enumerate(true_occ) if not b]
        if not free_chs:
            continue
        best_ch   = free_chs[0]

        # 1. Random
        random_correct += int(np.random.choice(range(NUM_CHANNELS)) in free_chs)

        # 2. Energy detection (sensing only)
        iq            = generate_single_iq(true_occ, duration_ms=0.5)
        psd_db, psd_l = iq_to_psd(iq)
        detected_occ  = detect_occupancy(psd_db, num_channels=NUM_CHANNELS)
        detected_free = [i for i, b in enumerate(detected_occ) if not b]
        if detected_free:
            sensing_correct += int(detected_free[0] in free_chs)

        # 3. Dense NN
        feats  = extract_feature_vector(psd_l).reshape(1, -1)
        scores = model.predict(feats, verbose=0)[0]
        nn_ch  = int(np.argmax(scores))
        nn_correct += int(nn_ch in free_chs)

    acc_r  = random_correct  / N_TRIALS * 100
    acc_s  = sensing_correct / N_TRIALS * 100
    acc_nn = nn_correct      / N_TRIALS * 100

    fig, ax = plt.subplots(figsize=(7, 4.5))
    methods = ["Random\nAllocation", "Energy\nDetection\nOnly", "Dense NN\n(Proposed)"]
    accs    = [acc_r, acc_s, acc_nn]
    cols    = ["#888780", "#185FA5", "#1D9E75"]
    bars    = ax.bar(methods, accs, color=cols, edgecolor="white", linewidth=0.5, width=0.45)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, f"{acc:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_ylabel("Correct Allocation Rate (%)")
    ax.set_title("Dynamic Spectrum Allocation — Method Comparison")
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, label="Random baseline (50%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")
    return acc_r, acc_s, acc_nn


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Model Evaluation — Cognitive Radio Project")
    print("=" * 60)

    model, X_test, Y_test, y_test = load_artifacts()
    print(f"  Loaded model : {model.count_params():,} params")
    print(f"  Test samples : {len(y_test)}")

    # Predictions
    Y_pred_prob = model.predict(X_test, verbose=0)
    y_pred      = np.argmax(Y_pred_prob, axis=1)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")
    report = classification_report(
        y_test, y_pred,
        target_names=CHANNEL_NAMES,
        output_dict=True
    )

    print(f"\n  Overall accuracy   : {acc * 100:.2f}%")
    print(f"  Weighted F1-score  : {f1 * 100:.2f}%")
    print()
    print(classification_report(y_test, y_pred, target_names=CHANNEL_NAMES))

    # Plots
    plot_confusion_matrix(y_test, y_pred,
                          os.path.join(PLOT_DIR, "confusion_matrix.png"))
    plot_per_channel_accuracy(y_test, y_pred,
                              os.path.join(PLOT_DIR, "per_channel_accuracy.png"))

    # Allocation comparison (requires simulation modules)
    try:
        acc_r, acc_s, acc_nn = plot_allocation_performance(
            model, os.path.join(PLOT_DIR, "allocation_comparison.png"))
    except Exception as e:
        print(f"  [!] Allocation plot skipped: {e}")
        acc_r, acc_s, acc_nn = None, None, None

    # Save metrics
    metrics = {
        "overall_accuracy"     : float(acc),
        "weighted_f1"          : float(f1),
        "per_class_report"     : report,
        "random_alloc_acc"     : acc_r,
        "sensing_only_acc"     : acc_s,
        "proposed_nn_acc"      : acc_nn,
    }
    with open(os.path.join(LOG_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Metrics saved → {LOG_DIR}/evaluation_metrics.json")
    print("  Done!\n")


if __name__ == "__main__":
    main()
