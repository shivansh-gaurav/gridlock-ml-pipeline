import pandas as pd
import numpy as np
import geohash
from sklearn.neighbors import KNeighborsRegressor

def main():
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    # Calculate geohash mean demand from train
    geohash_mean_demand = train.groupby("geohash")["demand"].mean()
    
    train_geos = pd.DataFrame({'geohash': geohash_mean_demand.index})
    train_geos['lat'] = train_geos['geohash'].apply(lambda x: geohash.decode(x)[0])
    train_geos['lon'] = train_geos['geohash'].apply(lambda x: geohash.decode(x)[1])
    train_geos['mean_demand'] = geohash_mean_demand.values
    
    knn_demand = KNeighborsRegressor(n_neighbors=3, weights='distance')
    knn_demand.fit(train_geos[['lat', 'lon']], train_geos['mean_demand'])
    
    # Identify unseen geohashes in test
    train_geohashes = set(train['geohash'].unique())
    test_geohashes = set(test['geohash'].unique())
    unseen_geohashes = list(test_geohashes - train_geohashes)
    
    test_geos = pd.DataFrame({'geohash': unseen_geohashes})
    test_geos['lat'] = test_geos['geohash'].apply(lambda x: geohash.decode(x)[0])
    test_geos['lon'] = test_geos['geohash'].apply(lambda x: geohash.decode(x)[1])
    
    pred_demand = knn_demand.predict(test_geos[['lat', 'lon']])
    impute_demand = dict(zip(unseen_geohashes, pred_demand))
    
    print(f"Imputed mean demand for {len(unseen_geohashes)} unseen geohashes.")
    
    # Adjust predictions for unseen geohashes
    global_mean_demand = train["demand"].mean()
    
    best_sub = pd.read_csv("outputs/kartik_geometry_optimal_blend.csv")
    test_with_sub = test.copy()
    test_with_sub["demand"] = best_sub["demand"]
    
    unseen_mask = test_with_sub["geohash"].isin(unseen_geohashes)
    
    def adjust_prediction(row):
        geohash_val = row["geohash"]
        current_pred = row["demand"]
        imputed_mean = impute_demand[geohash_val]
        adjustment_factor = imputed_mean / global_mean_demand
        return current_pred * adjustment_factor
        
    if unseen_mask.any():
        adjusted_preds = test_with_sub[unseen_mask].apply(adjust_prediction, axis=1)
        test_with_sub.loc[unseen_mask, "demand"] = adjusted_preds
    
    test_with_sub["demand"] = np.clip(test_with_sub["demand"], 0.0, 1.0)
    
    output_path = "outputs/strategy1_knn_submission.csv"
    best_sub["demand"] = test_with_sub["demand"]
    best_sub.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
