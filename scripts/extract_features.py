import os
import numpy as np
from tqdm import tqdm

IN_PROC  = "dataset/processed"
OUT_FEAT = "dataset/features"
os.makedirs(OUT_FEAT, exist_ok=True)


# ── FEATURE FUNCTIONS ──────────────────────────────────────────
def _total_power(psd: np.ndarray) -> float:
    return float(np.sum(psd) / len(psd))


def _spectral_centroid(psd: np.ndarray) -> float:
    idx = np.arange(len(psd), dtype=np.float32)
    return float(np.sum(idx * psd) / (np.sum(psd) + 1e-9) / len(psd))


def _spectral_bandwidth(psd: np.ndarray) -> float:
    peak = float(np.max(psd))
    return float(np.sum(psd > 0.3 * peak) / len(psd))


def _rms_level(psd: np.ndarray) -> float:
    return float(np.sqrt(np.mean(psd ** 2)))


def _kurtosis(psd: np.ndarray) -> float:
    rms = _rms_level(psd)
    mu4 = float(np.mean((psd - rms) ** 4))
    return float(np.clip(mu4 / (rms ** 4 + 1e-9) / 10.0, 0.0, 1.0))


def _skewness(psd: np.ndarray) -> float:
    rms = _rms_level(psd)
    mu3 = float(np.mean((psd - rms) ** 3))
    raw = mu3 / (rms ** 3 + 1e-9)
    return float(np.clip((raw + 3.0) / 6.0, 0.0, 1.0))


def _peak_snr(psd: np.ndarray) -> float:
    peak      = float(np.max(psd))
    noise_est = float(np.percentile(psd, 20))
    return float(np.clip((peak - noise_est) / (peak + 1e-9), 0.0, 1.0))


def _spectral_entropy(psd: np.ndarray) -> float:
    p = psd / (np.sum(psd) + 1e-9)
    return float(-np.sum(p * np.log(p + 1e-9)) / np.log(len(psd) + 1))


# ── MAIN EXTRACTOR ─────────────────────────────────────────────
def extract_feature_vector(psd_lin: np.ndarray) -> np.ndarray:
    """
    Extract 8-dimensional feature vector from a single PSD frame.

    Args:
        psd_lin : normalised linear PSD, shape (FFT_SIZE//2,)

    Returns:
        feats : float32 ndarray, shape (8,), values in [0, 1]
    """
    feats = np.array([
        _total_power(psd_lin),
        _spectral_centroid(psd_lin),
        _spectral_bandwidth(psd_lin),
        _rms_level(psd_lin),
        _kurtosis(psd_lin),
        _skewness(psd_lin),
        _peak_snr(psd_lin),
        _spectral_entropy(psd_lin),
    ], dtype=np.float32)

    return np.clip(feats, 0.0, 1.0)


def main():
    print("=" * 60)
    print("  Feature Extraction — Cognitive Radio Project")
    print("=" * 60)

    # Load processed PSD
    psd_lin = np.load(os.path.join(IN_PROC, "psd_lin.npy"))
    labels  = np.load(os.path.join(IN_PROC, "labels.npy"))

    print(f"  Loaded PSD  : {psd_lin.shape}")
    print(f"  Loaded labels: {labels.shape}")
    print(f"  Extracting 8 features per sample …")

    features = np.zeros((len(psd_lin), 8), dtype=np.float32)

    for i, psd in enumerate(tqdm(psd_lin, desc="Extracting")):
        features[i] = extract_feature_vector(psd)

    print(f"\n  Feature matrix shape : {features.shape}")
    print(f"  Feature min/max per dim:")
    for j, name in enumerate(
        ["total_power","centroid","bandwidth","rms",
         "kurtosis","skewness","peak_snr","entropy"]
    ):
        print(f"    [{j}] {name:20s}: "
              f"min={features[:,j].min():.4f}  "
              f"max={features[:,j].max():.4f}  "
              f"mean={features[:,j].mean():.4f}")

    # Save
    np.save(os.path.join(OUT_FEAT, "features.npy"), features)
    np.save(os.path.join(OUT_FEAT, "labels.npy"),   labels)

    print(f"\n  Saved → {OUT_FEAT}/features.npy  {OUT_FEAT}/labels.npy")
    print("  Done!\n")


if __name__ == "__main__":
    main()
