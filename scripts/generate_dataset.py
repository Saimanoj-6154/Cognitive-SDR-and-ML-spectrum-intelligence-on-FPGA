import os
import numpy as np
from scipy.fft import fft
from scipy.signal import butter, lfilter
import json
from tqdm import tqdm

# ── CONFIG ─────────────────────────────────────────────────────
SAMPLE_RATE   = 2.4e6          # 2.4 MS/s  (HackRF / ADALM-PLUTO)
FFT_SIZE      = 1024           # Must match FPGA Verilog FFT size
NUM_CHANNELS  = 6
FREQ_START    = 2.412e9        # 2.412 GHz  (WiFi Ch 1)
FREQ_STEP     = 5e6            # 5 MHz channel spacing
NOISE_DB      = -88            # Thermal noise floor (dBm)
SIGNAL_DB     = -55            # Occupied channel power (dBm)
SAMPLES_PER_CLASS = 500        # Per occupancy pattern
SEED          = 42

CHANNEL_NAMES = [f"Ch{i+1}" for i in range(NUM_CHANNELS)]
CHANNEL_FREQS = [FREQ_START + i * FREQ_STEP for i in range(NUM_CHANNELS)]

OUT_RAW  = "dataset/raw_iq_samples"
OUT_PROC = "dataset/processed"
os.makedirs(OUT_RAW,  exist_ok=True)
os.makedirs(OUT_PROC, exist_ok=True)

np.random.seed(SEED)


# ── HELPERS ────────────────────────────────────────────────────
def noise_power_watts(db: float) -> float:
    return 10 ** ((db - 30) / 10)


def generate_single_iq(occupancy: list, duration_ms: float = 1.0) -> np.ndarray:
    """
    Generate complex I/Q samples for a given channel occupancy pattern.

    Args:
        occupancy   : list of bool, True = channel i is occupied
        duration_ms : capture window in milliseconds

    Returns:
        iq : complex128 ndarray, shape (N,)
    """
    N  = int(SAMPLE_RATE * duration_ms / 1000)
    t  = np.arange(N) / SAMPLE_RATE

    # Thermal noise floor
    np_watts = noise_power_watts(NOISE_DB)
    iq = (np.random.randn(N) + 1j * np.random.randn(N)) * np.sqrt(np_watts / 2)

    for ch_idx, busy in enumerate(occupancy):
        if not busy:
            continue

        # Offset from baseband centre
        fc_offset = CHANNEL_FREQS[ch_idx] - FREQ_START - (NUM_CHANNELS * FREQ_STEP / 2)
        sig_watts = noise_power_watts(SIGNAL_DB)
        amp       = np.sqrt(sig_watts)

        # Simulate multi-subcarrier (OFDM-like) signal
        num_sc = np.random.randint(6, 16)
        for sc in range(num_sc):
            f_sc  = fc_offset + (sc - num_sc // 2) * 312.5e3
            phase = np.random.uniform(0, 2 * np.pi)
            iq   += (amp / num_sc) * np.exp(1j * (2 * np.pi * f_sc * t + phase))

        # Add random amplitude fading (Rayleigh channel)
        fade = np.abs(np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
        iq  *= fade if np.random.rand() > 0.5 else 1.0

    return iq


def iq_to_psd(iq: np.ndarray) -> np.ndarray:
    """
    Compute Power Spectral Density from IQ samples.
    Mimics FPGA FFT output (Xilinx LogiCORE 1024-pt FFT).

    Returns:
        psd_db  : dBm power per FFT bin, shape (FFT_SIZE // 2,)
        psd_lin : normalised linear PSD [0,1]
    """
    frame  = iq[:FFT_SIZE]
    window = np.hanning(FFT_SIZE)
    X      = fft(frame * window, n=FFT_SIZE)
    psd    = np.abs(X[:FFT_SIZE // 2]) ** 2 / (FFT_SIZE ** 2)
    psd_db = 10 * np.log10(psd + 1e-20) + 30          # Watts → dBm
    psd_lin = (psd_db - psd_db.min()) / (psd_db.max() - psd_db.min() + 1e-9)
    return psd_db, psd_lin


def label_from_occupancy(occupancy: list) -> int:
    """
    Map channel occupancy to best-free-channel label (0-5).
    Returns index of the highest-quality (lowest interference) free channel.
    If all busy, returns channel with lowest energy.
    """
    free_channels = [i for i, busy in enumerate(occupancy) if not busy]
    if free_channels:
        return int(np.random.choice(free_channels))
    # All busy: pick the one least occupied (fallback)
    return int(np.argmin([int(b) for b in occupancy]))


# ── MAIN GENERATION LOOP ───────────────────────────────────────
def main():
    print("=" * 60)
    print("  RF Dataset Generation — Cognitive Radio Project")
    print("=" * 60)
    print(f"  Channels     : {NUM_CHANNELS}  ({', '.join(CHANNEL_NAMES)})")
    print(f"  Sample rate  : {SAMPLE_RATE/1e6:.1f} MS/s")
    print(f"  FFT size     : {FFT_SIZE} pts")
    print(f"  Noise floor  : {NOISE_DB} dBm")
    print(f"  Signal power : {SIGNAL_DB} dBm")
    print(f"  Samples/class: {SAMPLES_PER_CLASS}")
    print()

    raw_iq_list   = []
    psd_db_list   = []
    psd_lin_list  = []
    labels        = []
    occupancy_log = []

    # Generate all 2^6 = 64 occupancy patterns, sample each
    patterns = []
    for mask in range(2 ** NUM_CHANNELS):
        occ = [(mask >> i) & 1 == 1 for i in range(NUM_CHANNELS)]
        patterns.append(occ)

    print(f"  Generating {len(patterns)} occupancy patterns × {SAMPLES_PER_CLASS} samples each …")

    for occ in tqdm(patterns, desc="Patterns"):
        for _ in range(SAMPLES_PER_CLASS):
            iq              = generate_single_iq(occ, duration_ms=1.0)
            psd_db, psd_lin = iq_to_psd(iq)
            label           = label_from_occupancy(occ)

            raw_iq_list.append(np.stack([iq.real, iq.imag], axis=-1)[:FFT_SIZE])
            psd_db_list.append(psd_db)
            psd_lin_list.append(psd_lin)
            labels.append(label)
            occupancy_log.append(occ)

    # Convert to arrays
    raw_iq   = np.array(raw_iq_list,  dtype=np.float32)
    psd_db   = np.array(psd_db_list,  dtype=np.float32)
    psd_lin  = np.array(psd_lin_list, dtype=np.float32)
    labels   = np.array(labels,       dtype=np.int32)

    print(f"\n  Dataset shape  : IQ={raw_iq.shape}  PSD_lin={psd_lin.shape}")
    print(f"  Label distribution: {np.bincount(labels)}")

    # Save raw IQ
    np.save(os.path.join(OUT_RAW, "iq_samples.npy"),   raw_iq)
    np.save(os.path.join(OUT_RAW, "occupancy_log.npy"), np.array(occupancy_log))

    # Save processed PSD
    np.save(os.path.join(OUT_PROC, "psd_db.npy"),  psd_db)
    np.save(os.path.join(OUT_PROC, "psd_lin.npy"), psd_lin)
    np.save(os.path.join(OUT_PROC, "labels.npy"),  labels)

    # Metadata
    meta = {
        "sample_rate"   : SAMPLE_RATE,
        "fft_size"      : FFT_SIZE,
        "num_channels"  : NUM_CHANNELS,
        "channel_freqs" : CHANNEL_FREQS,
        "num_samples"   : int(len(labels)),
        "label_counts"  : np.bincount(labels).tolist(),
        "noise_db"      : NOISE_DB,
        "signal_db"     : SIGNAL_DB,
    }
    with open(os.path.join(OUT_PROC, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Saved → {OUT_RAW}/  and  {OUT_PROC}/")
    print("  Done!\n")


if __name__ == "__main__":
    main()
