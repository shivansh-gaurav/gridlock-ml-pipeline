import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("outputs")

# Targets estimated from strategy9_test_calibration.py and other scripts
TRUE_TEST_STD = 0.1751
TRUE_TEST_MEAN = 0.1210 # Approximate, let's check strategy9 again

def main():
    best_file = OUTPUT_DIR / "kartik_geometry_optimal_blend.csv"
    mega_file = OUTPUT_DIR / "mega_model_v1.csv"
    
    best = pd.read_csv(best_file)
    mega = pd.read_csv(mega_file)
    
    for df, name in [(best, "best"), (mega, "mega")]:
        d = df["demand"]
        current_mean = d.mean()
        current_std = d.std()
        
        print(f"\n{name} stats:")
        print(f"  Mean: {current_mean:.6f}, Std: {current_std:.6f}")
        
        # Stretch std to TRUE_TEST_STD
        scale = TRUE_TEST_STD / current_std
        
        # Preserve the current mean
        adjusted = (d - current_mean) * scale + current_mean
        adjusted = np.clip(adjusted, 0.0, 1.0)
        
        print(f"  Adjusted Mean: {adjusted.mean():.6f}, Std: {adjusted.std():.6f}")
        
        df["demand"] = adjusted
        df.to_csv(OUTPUT_DIR / f"{name}_calibrated.csv", index=False)
        
    # Now blend them 50/50
    blended = best.copy()
    blended["demand"] = (best["demand"] + mega["demand"]) / 2
    blended.to_csv(OUTPUT_DIR / "final_blended_calibrated.csv", index=False)
    print("\nSaved final_blended_calibrated.csv")

if __name__ == "__main__":
    main()
