import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import os, sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")))

NUM_CHANNELS  = 6
CHANNEL_NAMES = [f"Ch{i+1}" for i in range(NUM_CHANNELS)]
CHANNEL_FREQS = [2412, 2422, 2432, 2437, 2447, 2462]   # MHz


@dataclass
class AllocationResult:
    selected_channel  : int
    channel_name      : str
    frequency_mhz     : int
    quality_score     : float
    is_free           : bool
    nn_confidence     : float
    reason            : str


# ── ALLOCATION STRATEGIES ─────────────────────────────────────
def allocate_random(occupancy: List[bool]) -> AllocationResult:
    """Baseline: random free channel selection."""
    free = [i for i, busy in enumerate(occupancy) if not busy]
    if not free:
        ch = int(np.random.randint(0, NUM_CHANNELS))
    else:
        ch = int(np.random.choice(free))
    return AllocationResult(
        selected_channel=ch, channel_name=CHANNEL_NAMES[ch],
        frequency_mhz=CHANNEL_FREQS[ch], quality_score=0.5,
        is_free=not occupancy[ch], nn_confidence=0.0,
        reason="Random selection"
    )


def allocate_sensing_only(occupancy: List[bool]) -> AllocationResult:
    """Energy detection only: first free channel."""
    free = [i for i, busy in enumerate(occupancy) if not busy]
    if not free:
        ch = 0
        reason = "All busy — defaulting to Ch1"
    else:
        ch     = free[0]
        reason = f"First free channel (energy detection)"
    return AllocationResult(
        selected_channel=ch, channel_name=CHANNEL_NAMES[ch],
        frequency_mhz=CHANNEL_FREQS[ch], quality_score=0.6,
        is_free=not occupancy[ch], nn_confidence=0.0,
        reason=reason
    )


def allocate_nn(nn_scores: np.ndarray,
                occupancy: List[bool],
                weight_sensing: float = 0.4,
                weight_nn:      float = 0.6) -> AllocationResult:
    """
    Dense NN + Energy Detection fusion allocation.

    Combines:
      combined_score[i] = w_sensing * (1-busy[i]) + w_nn * nn_score[i]

    Args:
        nn_scores       : softmax output from Dense NN, shape (6,)
        occupancy       : energy detection result, list of bool
        weight_sensing  : weight for sensing result
        weight_nn       : weight for NN quality prediction

    Returns:
        AllocationResult with selected channel info
    """
    # Sensing mask: 0 if busy, 1 if free
    sensing_mask = np.array([0.0 if busy else 1.0 for busy in occupancy])

    # Combined score
    combined = weight_sensing * sensing_mask + weight_nn * nn_scores

    ch            = int(np.argmax(combined))
    quality_score = float(combined[ch])
    nn_conf       = float(nn_scores[ch])

    reason = (f"NN quality={nn_conf:.2f} + "
              f"sensing={'free' if not occupancy[ch] else 'busy'} → "
              f"combined={quality_score:.2f}")

    return AllocationResult(
        selected_channel=ch, channel_name=CHANNEL_NAMES[ch],
        frequency_mhz=CHANNEL_FREQS[ch], quality_score=quality_score,
        is_free=not occupancy[ch], nn_confidence=nn_conf,
        reason=reason
    )


# ── FULL ALLOCATION PIPELINE ──────────────────────────────────
class CognitiveAllocator:
    """
    Main allocation engine — wraps model loading + inference + decision.
    """

    def __init__(self, model_path: str = "ml/saved_models/cognitive_radio_model.keras"):
        import tensorflow as tf
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        self.model       = tf.keras.models.load_model(model_path)
        self.alloc_log   = []
        self.cycle_count = 0
        print(f"  Allocator loaded model: {model_path}")

    def allocate(self, psd_lin: np.ndarray,
                 occupancy: List[bool]) -> AllocationResult:
        """
        Run full pipeline: features → NN → allocation decision.

        Args:
            psd_lin   : normalised FFT PSD, shape (512,)
            occupancy : energy detection result

        Returns:
            AllocationResult
        """
        from scripts.extract_features import extract_feature_vector

        # Extract 8 features
        feats    = extract_feature_vector(psd_lin).reshape(1, -1)

        # Dense NN inference
        nn_scores = self.model.predict(feats, verbose=0)[0]

        # Allocation decision
        result = allocate_nn(nn_scores, occupancy)

        # Log
        self.cycle_count += 1
        self.alloc_log.append({
            "cycle"          : self.cycle_count,
            "selected"       : result.channel_name,
            "freq_mhz"       : result.frequency_mhz,
            "quality_score"  : float(result.quality_score),
            "is_free"        : result.is_free,
            "nn_confidence"  : float(result.nn_confidence),
        })
        return result

    def get_summary(self) -> dict:
        if not self.alloc_log:
            return {}
        free_count  = sum(1 for r in self.alloc_log if r["is_free"])
        avg_quality = float(np.mean([r["quality_score"] for r in self.alloc_log]))
        ch_dist     = {f"Ch{i+1}": sum(1 for r in self.alloc_log
                       if r["selected"] == f"Ch{i+1}") for i in range(NUM_CHANNELS)}
        return {
            "total_cycles"  : self.cycle_count,
            "free_rate"     : free_count / self.cycle_count,
            "avg_quality"   : avg_quality,
            "channel_dist"  : ch_dist,
        }


# ── STANDALONE TEST ───────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    print("Dynamic Spectrum Allocator — standalone test\n")

    # Simulate 10 allocation decisions
    for trial in range(10):
        occupancy  = [np.random.rand() > 0.5 for _ in range(NUM_CHANNELS)]
        nn_scores  = np.random.dirichlet(np.ones(NUM_CHANNELS))   # simulate NN output

        result_r  = allocate_random(occupancy)
        result_s  = allocate_sensing_only(occupancy)
        result_nn = allocate_nn(nn_scores, occupancy)

        print(f"Trial {trial+1:02d}: Busy={[i+1 for i,b in enumerate(occupancy) if b]}")
        print(f"  Random     → {result_r.channel_name}  (free={result_r.is_free})")
        print(f"  Sensing    → {result_s.channel_name}  (free={result_s.is_free})")
        print(f"  NN Alloc   → {result_nn.channel_name}  (free={result_nn.is_free}, "
              f"q={result_nn.quality_score:.2f})")
        print()
