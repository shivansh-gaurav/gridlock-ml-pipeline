import pandas as pd
from pathlib import Path

def main():
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    # Load our highest scoring submission
    best_sub = pd.read_csv("outputs/kartik_geometry_optimal_blend.csv")
    
    # Append the pseudo-labels to test data
    test["demand"] = best_sub["demand"]
    
    # Ensure test columns match train columns
    test = test[train.columns]
    
    # Combine train and pseudo-test
    combined = pd.concat([train, test], ignore_index=True)
    
    output_path = "dataset/train_pseudo.csv"
    combined.to_csv(output_path, index=False)
    print(f"Created pseudo-labeled training set: {output_path}")
    print(f"Original train size: {len(train)}")
    print(f"Pseudo-labeled test size: {len(test)}")
    print(f"Combined train size: {len(combined)}")

if __name__ == "__main__":
    main()
