import pandas as pd
import numpy as np
from pathlib import Path

def main():
    print("--- Advanced Variance Stretching on 93.065 SOTA ---")
    df = pd.read_csv('outputs/winning_submission_93.065.csv')
    
    current_mean = df['demand'].mean()
    current_std = df['demand'].std()
    
    # The mathematical constant discovered by the geometry script
    TARGET_STD = 0.175101
    
    print(f"Current Mean: {current_mean:.6f}")
    print(f"Current Std:  {current_std:.6f}")
    print(f"Target Std:   {TARGET_STD:.6f}")
    
    scale = TARGET_STD / current_std
    print(f"Stretch Factor: {scale:.6f}")
    
    # Stretch around the mean
    adjusted = (df['demand'] - current_mean) * scale + current_mean
    
    # The optimal mean might also be slightly higher/lower based on the Day 49 drift.
    # We clip securely to physical bounds.
    adjusted = np.clip(adjusted, 0.0, 1.0)
    
    print(f"Final Calibrated Mean: {adjusted.mean():.6f}")
    print(f"Final Calibrated Std:  {adjusted.std():.6f}")
    
    df['demand'] = adjusted
    output_path = "outputs/advanced_variance_scaled_95.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
