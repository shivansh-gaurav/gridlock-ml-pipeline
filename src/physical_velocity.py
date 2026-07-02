import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 10: Physical Velocity Analysis ---")
    train = pd.read_csv("dataset/train.csv")
    archetypes = pd.read_csv('dataset/geohash_archetypes_v6.csv')
    
    # 1. Filter to Day 48 and Day 49 Morning
    d48 = train[train.day == 48].copy()
    d49 = train[train.day == 49].copy()
    
    # Merge archetypes
    d48 = d48.merge(archetypes, on='geohash', how='left')
    d49 = d49.merge(archetypes, on='geohash', how='left')
    
    # Calculate surge ratio (D49 / D48) at start and end of morning
    def get_surge_at_hour(hour):
        m48 = d48[d48.timestamp.str.startswith(f"{hour}:")].groupby('cluster')['demand'].mean()
        m49 = d49[d49.timestamp.str.startswith(f"{hour}:")].groupby('cluster')['demand'].mean()
        return (m49 / (m48 + 1e-6))

    surge_h0 = get_surge_at_hour(0)
    surge_h2 = get_surge_at_hour(2)
    
    # Measured Velocity: Change in surge ratio over 2 hours
    velocity = (surge_h2 - surge_h0) / 2.0
    # Fill NaN velocities with the mean velocity of all clusters
    velocity = velocity.fillna(velocity.mean())
    
    print("Measured Surge Velocities per Cluster (per hour):")
    print(velocity)
    
    # 2. Apply measured velocities
    sota = pd.read_csv('outputs/winning_submission_93.065.csv')
    test = pd.read_csv('dataset/test.csv')
    
    merged = pd.merge(sota, test[['Index', 'geohash', 'timestamp']], on='Index')
    merged = pd.merge(merged, archetypes, on='geohash', how='left')
    
    # Default to cluster with median volume if missing
    default_cluster = 2
    merged['cluster'] = merged['cluster'].fillna(default_cluster).astype(int)
    
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    t_elapsed_hours = (merged['time_minutes'] - 135) / 60.0
    
    for c_id in velocity.index:
        v = velocity[c_id]
        mask = (merged['cluster'] == c_id).values
        # Capping trend at -40% to +10% to ensure physical stability
        trend_factor = 1.0 + np.clip(v * t_elapsed_hours[mask].values, -0.40, 0.10)
        merged.loc[mask, 'demand'] *= trend_factor
        
    merged['demand'] = merged['demand'].clip(0.0, 1.0)
    # Final safety fallback: if anything is still NaN, use original SOTA
    fallback_values = sota.set_index('Index').loc[merged['Index']]['demand']
    merged['demand'] = merged['demand'].fillna(pd.Series(fallback_values.values, index=merged.index))
    
    output_path = "outputs/PHYSICAL_VELOCITY_SOTA_95.csv"
    merged[['Index', 'demand']].sort_values('Index').to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")
    print("Final stats:")
    print(merged['demand'].describe())

if __name__ == "__main__":
    main()
