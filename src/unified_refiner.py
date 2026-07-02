import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Unified SOTA Behavioral Refiner ---")
    sota = pd.read_csv('outputs/winning_submission_93.065.csv')
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    # 1. Re-calculate Archetypes in memory to ensure 100% alignment
    print("Clustering geohashes...")
    d48 = train[train.day == 48].copy()
    stats = d48.groupby('geohash')['demand'].agg(['mean', 'std']).rename(columns={'mean': 'vol', 'std': 'vola'})
    
    # Clustering
    stats = stats.dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(stats)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    stats['cluster'] = kmeans.fit_predict(X)
    
    # Identify 'Core' (Highest volume cluster)
    core_cluster = stats.groupby('cluster')['vol'].mean().idxmax()
    stats['is_core'] = (stats['cluster'] == core_cluster)
    
    print(f"Detected {stats['is_core'].sum()} Core roads and {len(stats) - stats['is_core'].sum()} Suburb roads.")
    
    # 2. Merge with SOTA
    merged = pd.merge(test[['Index', 'geohash', 'timestamp']], sota, on='Index')
    merged = pd.merge(merged, stats[['is_core']], on='geohash', how='left')
    merged['is_core'] = merged['is_core'].fillna(False)
    
    # 3. Apply Multipliers
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    t_norm = (merged['time_minutes'] - 135) / (825 - 135)
    
    # THE REFINEMENT:
    # Core roads: +2% progression
    # Suburb roads: -4% decay
    core_mask = (merged['is_core'] == True).values
    suburb_mask = (merged['is_core'] == False).values
    
    merged.loc[core_mask, 'demand'] *= (1.0 + 0.02 * t_norm[core_mask].values)
    merged.loc[suburb_mask, 'demand'] *= (1.0 - 0.04 * t_norm[suburb_mask].values)
    
    merged['demand'] = merged['demand'].clip(0.0, 1.0)
    
    # 4. VERIFICATION (Crucial)
    diff = (merged['demand'] - sota.set_index('Index').loc[merged['Index']]['demand'].values).abs().sum()
    print(f"Verification: Total Absolute Change from SOTA: {diff:.6f}")
    
    if diff == 0:
        print("ERROR: Multipliers were not applied!")
        return

    output_path = "outputs/FIXED_BEHAVIORAL_SOTA_93.csv"
    merged[['Index', 'demand']].sort_values('Index').to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
