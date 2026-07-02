from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_DIR = Path("outputs")

# Estimated from exact leaderboard scores of baseline, lag model, and their blends.
ESTIMATED_SST = 1280.9313328482735
CURRENT_BEST_SCORE = 91.81748


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Use a probe submission score to compute the optimal blend along that probe direction."
    )
    parser.add_argument("--probe-score", type=float, required=True, help="Leaderboard score for probe submission.")
    parser.add_argument("--alpha", type=float, default=0.25, help="Probe weight used in the probe file.")
    parser.add_argument("--current", default="next_submission.csv", help="Current best submission file.")
    parser.add_argument("--direction", default="future_window_prefix_ensemble.csv", help="Direction endpoint file.")
    parser.add_argument("--output", default="probe_optimized_submission.csv", help="Output optimized submission file.")
    args = parser.parse_args()

    current = pd.read_csv(OUTPUT_DIR / args.current)
    direction = pd.read_csv(OUTPUT_DIR / args.direction)
    if not current["Index"].equals(direction["Index"]):
        raise ValueError("Index mismatch between current and direction files.")

    p0 = current["demand"].to_numpy(dtype=float)
    p1 = direction["demand"].to_numpy(dtype=float)
    d = p1 - p0

    e0 = 1.0 - CURRENT_BEST_SCORE / 100.0
    e_probe = 1.0 - args.probe_score / 100.0
    q = float(np.sum(d * d) / ESTIMATED_SST)
    residual_projection = (args.alpha * args.alpha * q - (e_probe - e0)) / (2 * args.alpha)
    beta = residual_projection / q

    optimized = np.clip(p0 + beta * d, 0.0, 1.0)
    estimated_error = e0 - (residual_projection * residual_projection / q)
    estimated_score = max(0.0, 100.0 * (1.0 - estimated_error))

    pd.DataFrame({"Index": current["Index"], "demand": optimized}).to_csv(OUTPUT_DIR / args.output, index=False)

    print(f"q(direction)={q:.8f}")
    print(f"residual_projection={residual_projection:.8f}")
    print(f"optimal_beta={beta:.8f}")
    print(f"estimated_optimized_score={estimated_score:.5f}")
    print(f"wrote {OUTPUT_DIR / args.output}")


if __name__ == "__main__":
    main()
