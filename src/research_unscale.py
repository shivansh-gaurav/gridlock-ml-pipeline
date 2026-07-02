import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('GridLock/dataset/train.csv')
    demands = df['demand'].values
    
    print("Searching for the universal multiplier...")
    
    # We are looking for a multiplier M such that (demand * M) is very close to an integer
    # We will test multipliers from 1 to 100,000
    
    best_multiplier = 0
    best_error = float('inf')
    
    # To be fast, we will only test on a random sample of 1000 demands
    sample = np.random.choice(demands, 1000, replace=False)
    
    # We suspect it's a large number, let's scan
    for m in range(10, 500000, 10):
        test_vals = sample * m
        # Calculate the distance to the nearest integer
        errors = np.abs(test_vals - np.round(test_vals))
        mean_error = np.mean(errors)
        
        if mean_error < best_error:
            best_error = mean_error
            best_multiplier = m
            
        if mean_error < 1e-4:
            print(f"BINGO! Found potential multiplier: {m} with error {mean_error:.6f}")
            
    print(f"\nBest multiplier found: {best_multiplier}")
    print(f"Mean error to nearest integer: {best_error:.6f}")
    
if __name__ == "__main__":
    main()
