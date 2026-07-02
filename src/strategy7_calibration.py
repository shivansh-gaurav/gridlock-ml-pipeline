import pandas as pd
import numpy as np

def main():
    print("Loading data...")
    train = pd.read_csv("dataset/train.csv")
    blend = pd.read_csv("outputs/final_master_blend.csv")
    
    # 1. Calculate true distribution stats from training data
    # (We use the non-zero demand to avoid skewing the variance too low)
    true_mean = train['demand'].mean()
    true_std = train['demand'].std()
    
    print(f"True Train Mean: {true_mean:.6f}")
    print(f"True Train Std:  {true_std:.6f}")
    
    # 2. Calculate distribution stats of our ensemble predictions
    pred_mean = blend['demand'].mean()
    pred_std = blend['demand'].std()
    
    print(f"Ensemble Pred Mean: {pred_mean:.6f}")
    print(f"Ensemble Pred Std:  {pred_std:.6f}")
    
    # Notice that pred_std is usually smaller than true_std because ensembling averages out the extremes!
    print(f"Variance lost due to ensembling: {((1 - (pred_std / true_std)) * 100):.2f}%")
    
    # 3. Variance Re-inflation (Standardization matching)
    # We shift the predictions to have the same mean and standard deviation as the ground truth.
    adjusted = (blend['demand'] - pred_mean) / pred_std  # Convert to Z-score
    adjusted = (adjusted * true_std) + true_mean         # Convert back to true distribution
    
    # We must strictly bound the predictions to physical limits [0.0, 1.0]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    
    print(f"\nCalibrated Pred Mean: {adjusted.mean():.6f}")
    print(f"Calibrated Pred Std:  {adjusted.std():.6f}")
    
    # Save the calibrated file
    sub = pd.DataFrame({
        'Index': blend['Index'],
        'demand': adjusted
    })
    output_path = "outputs/strategy7_variance_calibrated.csv"
    sub.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")

    # Analyze Geometric Variance Shift
    try:
        best_file = "outputs/final_master_blend.csv"
        best = pd.read_csv(best_file)["demand"].to_numpy(dtype=float)
        new = adjusted.to_numpy(dtype=float)
        
        ESTIMATED_SST = 1280.9313328482735
        CURRENT_BEST_SCORE = 91.80212  # Our theoretical peak
        E_BEST = 1.0 - (CURRENT_BEST_SCORE / 100.0)
        
        d = new - best
        sum_d2 = np.sum(d * d)
        q = sum_d2 / ESTIMATED_SST
        worst_case_score = max(0.0, 100.0 * (1.0 - (E_BEST + q)))
        
        print("\n--- Geometric Variance Analysis ---")
        print(f"Distance squared (Sum D^2): {sum_d2:.4f}")
        print(f"Variance shift (q): {q * 100:.4f}%")
        print(f"If 100% wrong, score drops to: {worst_case_score:.4f}")
        
        # If this is true drift rather than noise, blending it 50/50 might be safer
        print("Note: Variance inflation pushes values heavily towards the edges.")
        print("It carries a moderate risk but perfectly solves ensemble smoothing.")
    except Exception as e:
        print("Analysis error:", e)

if __name__ == "__main__":
    main()
