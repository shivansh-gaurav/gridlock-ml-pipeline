import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    print("--- Phase 8: Hierarchical Archetype Analysis (6 Clusters) ---")
    train = pd.read_csv("dataset/train.csv")
    d48 = train[train.day == 48].copy()
    
    # 1. Feature Engineering per Geohash
    stats = d48.groupby('geohash')['demand'].agg(['mean', 'std', 'max']).rename(columns={'mean': 'vol', 'std': 'vola', 'max': 'peak'})
    
    # Day 49 Morning Surge Factor
    d49_m = train[(train.day == 49) & (train.timestamp.str.startswith(('0:', '1:', '2:')))].groupby('geohash')['demand'].mean()
    d48_m = d48[d48.timestamp.str.startswith(('0:', '1:', '2:'))].groupby('geohash')['demand'].mean()
    surge = (d49_m / (d48_m + 1e-6)).rename('surge_factor')
    
    features = pd.concat([stats, surge], axis=1).fillna(0)
    
    # 2. 6-Cluster K-Means
    scaler = StandardScaler()
    X = scaler.fit_transform(features[['vol', 'vola', 'peak', 'surge_factor']])
    
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    features['cluster'] = kmeans.fit_predict(X)
    
    # 3. Analyze Clusters
    summary = features.groupby('cluster').agg({'vol': 'mean', 'surge_factor': 'mean'})
    summary['count'] = features.groupby('cluster').size()
    summary = summary.sort_values('vol', ascending=False)
    
    # Labeling based on behavior
    # Top vol = Ultra-Core
    # Next vol = Arteries
    # Lowest vol = Deep Suburbs
    # High surge but low vol = Outliers/Noise
    
    print("6-Cluster Behavioral Profiles:")
    print(summary)
    
    # Save for the booster
    features[['cluster']].reset_index().to_csv("dataset/geohash_archetypes_v6.csv", index=False)

if __name__ == "__main__":
    main()
