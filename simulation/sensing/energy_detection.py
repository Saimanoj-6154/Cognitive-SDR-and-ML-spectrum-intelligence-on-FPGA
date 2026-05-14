import numpy as np
from scipy.fft import fft
from typing import List, Tuple

SAMPLE_RATE   = 2.4e6
FFT_SIZE      = 1024
NUM_CHANNELS  = 6
NOISE_DB      = -88.0           # dBm noise floor
THRESHOLD_DB  = -75.0           # Energy detection threshold (dBm)

CHANNEL_OFFSETS_MHZ = [0.0, 10.0, 20.0, 25.0, 35.0, 50.0]
CHANNEL_BW_MHZ      = 4.0       # Guard band ±2 MHz each side


# ── ENERGY DETECTION ──────────────────────────────────────────
def detect_occupancy(psd_db: np.ndarray,
                     num_channels: int = NUM_CHANNELS,
                     threshold_db: float = THRESHOLD_DB) -> List[bool]:
    """
    Detect channel occupancy from FFT PSD using energy threshold.

    Args:
        psd_db       : dBm PSD array, shape (FFT_SIZE//2,)
        num_channels : number of channels to check
        threshold_db : detection threshold in dBm

    Returns:
        occupancy : list of bool, True = channel occupied
    """
    N = len(psd_db)
    occupancy = []

    for i in range(num_channels):
        offset_bin = int(CHANNEL_OFFSETS_MHZ[i] / (SAMPLE_RATE / 1e6 / 2) * N)
        bw_bins    = int(CHANNEL_BW_MHZ      / (SAMPLE_RATE / 1e6 / 2) * N)

        lo = max(0, offset_bin - bw_bins)
        hi = min(N - 1, offset_bin + bw_bins)

        ch_power = float(np.max(psd_db[lo:hi + 1]))
        occupancy.append(ch_power > threshold_db)

    return occupancy


def energy_detection_iq(iq_samples: np.ndarray,
                         threshold_db: float = THRESHOLD_DB) -> Tuple[List[bool], np.ndarray]:
    """
    Full pipeline: IQ → FFT → Energy Detection.

    Returns:
        occupancy : list of bool per channel
        psd_db    : full PSD array in dBm
    """
    frame  = iq_samples[:FFT_SIZE]
    window = np.hanning(FFT_SIZE)
    X      = fft(frame * window, n=FFT_SIZE)
    psd    = np.abs(X[:FFT_SIZE // 2]) ** 2 / (FFT_SIZE ** 2)
    psd_db = 10 * np.log10(psd + 1e-20) + 30

    occupancy = detect_occupancy(psd_db, threshold_db=threshold_db)
    return occupancy, psd_db


def compute_pfa_pd(psd_db: np.ndarray,
                   true_occupancy: List[bool],
                   thresholds: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute ROC curve (Probability of False Alarm vs Detection).
    Used for paper Figure: ROC curve.

    Returns:
        pfa : array of false alarm probabilities
        pd  : array of detection probabilities
    """
    if thresholds is None:
        thresholds = np.linspace(psd_db.min(), psd_db.max(), 100)

    pfa_list, pd_list = [], []

    for thresh in thresholds:
        detected = [psd_db[max(0, int(CHANNEL_OFFSETS_MHZ[i]/(SAMPLE_RATE/1e6/2)*len(psd_db))-5):
                           int(CHANNEL_OFFSETS_MHZ[i]/(SAMPLE_RATE/1e6/2)*len(psd_db))+5].max()
                    > thresh for i in range(NUM_CHANNELS)]

        tp = sum(d and t for d, t in zip(detected, true_occupancy))
        fp = sum(d and not t for d, t in zip(detected, true_occupancy))
        tn = sum(not d and not t for d, t in zip(detected, true_occupancy))
        fn = sum(not d and t for d, t in zip(detected, true_occupancy))

        pd_val  = tp / (tp + fn + 1e-9)
        pfa_val = fp / (fp + tn + 1e-9)
        pfa_list.append(pfa_val)
        pd_list.append(pd_val)

    return np.array(pfa_list), np.array(pd_list)


# ── MAIN (standalone test) ────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")))
    from scripts.generate_dataset import generate_single_iq

    np.random.seed(42)
    true_occ = [True, False, True, False, True, False]
    iq       = generate_single_iq(true_occ, duration_ms=1.0)

    detected, psd_db = energy_detection_iq(iq)

    print("Energy Detection Results:")
    print(f"  True occupancy     : {true_occ}")
    print(f"  Detected occupancy : {detected}")

    correct = sum(d == t for d, t in zip(detected, true_occ))
    print(f"  Sensing accuracy   : {correct}/{NUM_CHANNELS} = {correct/NUM_CHANNELS*100:.1f}%")
