import pandas as pd
import numpy as np
from fractions import Fraction
from collections import Counter

def main():
    train = pd.read_csv("dataset/train.csv")
    print("Finding denominators using Fraction...")
    
    geohashes = train['geohash'].unique()
    denominators = {}
    
    for geo in geohashes:
        s = train[train['geohash'] == geo]['demand'].dropna().values
        s = s[(s > 0) & (s < 1)]
        if len(s) == 0:
            continue
            
        # Check first non-zero value
        val = s[0]
        f = Fraction(val).limit_denominator(200000)
        den = f.denominator
        
        # Verify if this denominator works for all values
        vals = s * den
        diffs = np.abs(vals - np.round(vals))
        if np.max(diffs) < 1e-5:
            denominators[geo] = den
            
    print(f"Found {len(denominators)} out of {len(geohashes)} denominators.")
    if denominators:
        c = Counter(denominators.values())
        print("Top denominators:", c.most_common(10))

if __name__ == "__main__":
    main()
