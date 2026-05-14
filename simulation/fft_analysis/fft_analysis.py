import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import welch

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

SAMPLE_RATE = 2.4e6
FFT_SIZE    = 1024
PLOT_DIR    = "results/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

CHANNEL_FREQS_MHZ = [0.0, 10.0, 20.0, 25.0, 35.0, 50.0]   # offsets from baseband
NUM_CHANNELS      = 6


# ── CORE FFT FUNCTION (FPGA equivalent) ───────────────────────
def fpga_fft_1024(iq_samples: np.ndarray, apply_window: bool = True):
    """
    Simulates Xilinx LogiCORE 1024-point FFT IP.

    In hardware (Verilog):
      - Radix-2 Cooley–Tukey butterfly
      - Pipelined, 10-stage (log2 1024)
      - Latency: ~50 clock cycles @ 100 MHz → 0.5 µs
      - Output: 32-bit fixed-point |X[k]|²

    Args:
        iq_samples   : complex IQ frame, shape (≥1024,)
        apply_window : apply Hann window (matches FPGA windowing block)

    Returns:
        freqs_mhz : frequency axis (MHz), shape (512,)
        psd_db    : power spectral density (dBm), shape (512,)
        psd_lin   : normalised linear PSD [0,1], shape (512,)
        X_complex : complex FFT output (for phase analysis), shape (512,)
    """
    frame = iq_samples[:FFT_SIZE].copy()

    if apply_window:
        window = np.hanning(FFT_SIZE)
        frame  = frame * window

    X      = fft(frame, n=FFT_SIZE)
    X_half = X[:FFT_SIZE // 2]

    # Power |X[k]|² normalised by FFT length (matches FPGA fixed-point output)
    psd    = np.abs(X_half) ** 2 / (FFT_SIZE ** 2)
    psd_db = 10 * np.log10(psd + 1e-20) + 30     # → dBm

    freqs_hz  = fftfreq(FFT_SIZE, d=1.0 / SAMPLE_RATE)[:FFT_SIZE // 2]
    freqs_mhz = freqs_hz / 1e6

    psd_lin = (psd_db - psd_db.min()) / (psd_db.max() - psd_db.min() + 1e-9)

    return freqs_mhz, psd_db, psd_lin, X_half


def welch_psd(iq_samples: np.ndarray):
    """
    Welch's method PSD (higher frequency resolution, used for analysis).
    FPGA uses single FFT frame; Welch used for paper figures.
    """
    f, Pxx = welch(iq_samples, fs=SAMPLE_RATE,
                   nperseg=FFT_SIZE, return_onesided=False)
    psd_db = 10 * np.log10(np.abs(Pxx) + 1e-20) + 30
    return f / 1e6, psd_db


# ── CHANNEL DETECTION FROM FFT OUTPUT ─────────────────────────
def detect_channels_from_fft(psd_db: np.ndarray,
                              freqs_mhz: np.ndarray,
                              threshold_db: float = -75.0) -> dict:
    """
    Detect which channels are occupied from FFT output bins.
    Simulates FPGA energy detection comparator block.

    Returns:
        channel_status : dict {ch_name: {'occupied': bool, 'power_db': float}}
    """
    result = {}
    for i, offset_mhz in enumerate(CHANNEL_FREQS_MHZ):
        # Find FFT bins within ±2 MHz of channel centre
        mask       = np.abs(freqs_mhz - offset_mhz) < 2.0
        if not mask.any():
            result[f"Ch{i+1}"] = {"occupied": False, "power_db": -100.0}
            continue
        ch_power   = float(np.max(psd_db[mask]))
        is_occupied = ch_power > threshold_db
        result[f"Ch{i+1}"] = {"occupied": is_occupied, "power_db": ch_power}
    return result


# ── PLOT FFT ANALYSIS ─────────────────────────────────────────
def plot_fft_analysis(freqs_mhz, psd_db, psd_lin,
                      occupancy: list, save_path: str):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # Top: PSD in dBm
    axes[0].plot(freqs_mhz, psd_db, color="#185FA5", linewidth=0.8)
    axes[0].fill_between(freqs_mhz, psd_db, psd_db.min(),
                         alpha=0.15, color="#185FA5")
    axes[0].set_ylabel("Power (dBm)")
    axes[0].set_title("FPGA FFT Output — Power Spectral Density (1024-pt)")
    axes[0].grid(alpha=0.3)
    axes[0].axhline(-75, color="#E24B4A", linestyle="--",
                    linewidth=1, label="Energy detection threshold")

    # Mark channels
    for i, (offset, busy) in enumerate(zip(CHANNEL_FREQS_MHZ, occupancy)):
        color = "#E24B4A" if busy else "#1D9E75"
        label = "Busy" if busy else "Free"
        axes[0].axvspan(offset - 2, offset + 2, alpha=0.12, color=color)
        axes[0].text(offset, psd_db.max() * 0.92,
                     f"Ch{i+1}\n({label})",
                     ha="center", fontsize=7,
                     color="#501313" if busy else "#085041")
    axes[0].legend(fontsize=8)

    # Bottom: Normalised PSD (feature extraction input)
    axes[1].bar(range(len(psd_lin)), psd_lin,
                color="#BA7517", alpha=0.7, width=1.0)
    axes[1].set_xlabel("FFT Bin Index (0 – 511)")
    axes[1].set_ylabel("Normalised Power [0, 1]")
    axes[1].set_title("Normalised PSD — Input to Feature Extractor")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {save_path}")


# ── MAIN ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FFT Analysis — FPGA Simulation (SciPy)")
    print("=" * 60)

    # Import here to avoid circular imports
    sys.path.insert(0, os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../..")))
    from scripts.generate_dataset import generate_single_iq

    # Test with 3 busy channels
    occupancy = [True, False, True, False, False, True]
    print(f"  Test occupancy : {occupancy}")

    iq = generate_single_iq(occupancy, duration_ms=2.0)
    freqs_mhz, psd_db, psd_lin, X_c = fpga_fft_1024(iq)

    print(f"  FFT output shape : {psd_db.shape}")
    print(f"  PSD range        : {psd_db.min():.1f} – {psd_db.max():.1f} dBm")

    # Channel detection
    ch_status = detect_channels_from_fft(psd_db, freqs_mhz)
    print("\n  Channel Detection Results:")
    for ch, info in ch_status.items():
        status = "BUSY" if info["occupied"] else "FREE"
        print(f"    {ch}: {status}  ({info['power_db']:.1f} dBm)")

    # Plot
    plot_fft_analysis(freqs_mhz, psd_db, psd_lin, occupancy,
                      os.path.join(PLOT_DIR, "fft_analysis.png"))
    print("\n  Done!\n")


if __name__ == "__main__":
    main()
