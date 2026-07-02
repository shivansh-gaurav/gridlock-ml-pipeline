import pandas as pd
import numpy as np

def find_denominator(series, max_den=5000, tol=1e-5):
    # Filter out 0 and 1
    s = series[(series > 0) & (series < 1)].copy()
    if len(s) == 0:
        return 1
        
    for den in range(1, max_den + 1):
        # Multiply by denominator
        vals = s * den
        # Distance to nearest integer
        diffs = np.abs(vals - np.round(vals))
        
        # If the max difference across all rows is less than tolerance, we found it
        if diffs.max() < tol:
            return den
            
    return None

def main():
    train = pd.read_csv("dataset/train.csv")
    
    # Let's try to find a global denominator first
    print("Searching for global denominator...")
    global_den = find_denominator(train['demand'], max_den=10000, tol=1e-4)
    
    if global_den:
        print(f"FOUND GLOBAL DENOMINATOR: {global_den}")
    else:
        print("No global denominator found. Searching per RoadType and NumberofLanes...")
        
        results = []
        grouped = train.groupby(['RoadType', 'NumberofLanes'])
        for name, group in grouped:
            den = find_denominator(group['demand'], max_den=10000, tol=1e-4)
            results.append({
                'RoadType': name[0],
                'NumberofLanes': name[1],
                'Capacity': den,
                'Rows': len(group)
            })
            
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))
        
if __name__ == "__main__":
    main()
