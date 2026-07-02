from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "matplotlib-cache"))

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs")
REPORT_DIR = Path("reports")
RANDOM_STATE = 42


def parse_minutes(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["time_minutes"] = frame["timestamp"].map(parse_minutes)
    frame["hour"] = frame["time_minutes"] // 60
    frame["minute"] = frame["time_minutes"] % 60
    frame["slot_15m"] = frame["time_minutes"] // 15
    frame["time_sin"] = np.sin(2 * np.pi * frame["time_minutes"] / (24 * 60))
    frame["time_cos"] = np.cos(2 * np.pi * frame["time_minutes"] / (24 * 60))
    frame["geohash_prefix_4"] = frame["geohash"].str[:4]
    frame["geohash_prefix_5"] = frame["geohash"].str[:5]
    return frame


def prefix_slope(group: pd.DataFrame) -> float:
    clean = group.dropna(subset=["demand"])
    if len(clean) < 2:
        return 0.0
    x = clean["time_minutes"].to_numpy(dtype=float)
    y = clean["demand"].to_numpy(dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator == 0:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denominator)


def make_prefix_stats(prefix: pd.DataFrame) -> pd.DataFrame:
    sorted_prefix = prefix.sort_values(["geohash", "time_minutes"])
    stats = sorted_prefix.groupby("geohash").agg(
        prefix_mean=("demand", "mean"),
        prefix_median=("demand", "median"),
        prefix_min=("demand", "min"),
        prefix_max=("demand", "max"),
        prefix_std=("demand", "std"),
        prefix_count=("demand", "size"),
        prefix_first=("demand", "first"),
        prefix_last=("demand", "last"),
    )
    stats["prefix_slope"] = sorted_prefix.groupby("geohash").apply(prefix_slope, include_groups=False)

    pivot = sorted_prefix.pivot_table(
        index="geohash",
        columns="time_minutes",
        values="demand",
        aggfunc="mean",
    )
    pivot = pivot.rename(columns={column: f"prefix_at_{int(column)}" for column in pivot.columns})
    return stats.join(pivot, how="left").reset_index()


def attach_prefix_stats(rows: pd.DataFrame, prefix_stats: pd.DataFrame) -> pd.DataFrame:
    frame = rows.merge(prefix_stats, on="geohash", how="left")
    frame["prefix_range"] = frame["prefix_max"] - frame["prefix_min"]
    frame["prefix_last_minus_first"] = frame["prefix_last"] - frame["prefix_first"]
    return frame


def feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [
        "time_minutes",
        "hour",
        "minute",
        "slot_15m",
        "time_sin",
        "time_cos",
        "NumberofLanes",
        "Temperature",
        "prefix_mean",
        "prefix_median",
        "prefix_min",
        "prefix_max",
        "prefix_std",
        "prefix_count",
        "prefix_first",
        "prefix_last",
        "prefix_slope",
        "prefix_range",
        "prefix_last_minus_first",
    ]
    numeric += [column for column in df.columns if column.startswith("prefix_at_")]
    categorical = [
        "RoadType",
        "LargeVehicles",
        "Landmarks",
        "Weather",
        "geohash_prefix_4",
        "geohash_prefix_5",
    ]
    return numeric, categorical


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ]
    )


def make_models(numeric: list[str], categorical: list[str]) -> dict[str, Pipeline]:
    preprocessor = make_preprocessor(numeric, categorical)
    return {
        "RandomForest": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=350,
                        max_depth=16,
                        min_samples_leaf=3,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "ExtraTrees": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=450,
                        max_depth=16,
                        min_samples_leaf=2,
                        max_features=0.85,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=450,
                        learning_rate=0.035,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=450,
                        max_depth=4,
                        learning_rate=0.035,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        objective="reg:squarederror",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("preprocess", clone(preprocessor)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=450,
                        learning_rate=0.035,
                        num_leaves=31,
                        min_child_samples=25,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        reg_lambda=2.0,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        ),
    }


def metric_row(name: str, y_true: pd.Series, prediction: np.ndarray, split: str) -> dict[str, float | str]:
    prediction = np.clip(prediction, 0.0, 1.0)
    return {
        "split": split,
        "model": name,
        "r2": r2_score(y_true, prediction),
        "score": 100 * r2_score(y_true, prediction),
        "mae": mean_absolute_error(y_true, prediction),
        "rmse": float(np.sqrt(mean_squared_error(y_true, prediction))),
    }


def geohash_holdout_validation(day48_window: pd.DataFrame, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    geohashes = np.array(sorted(day48_window["geohash"].unique()))
    records = []
    models = make_models(numeric, categorical)
    features = numeric + categorical

    for seed in [11, 29, 42, 71, 101]:
        rng = np.random.default_rng(seed)
        valid_geohashes = set(rng.choice(geohashes, size=max(1, int(len(geohashes) * 0.2)), replace=False))
        train_part = day48_window[~day48_window["geohash"].isin(valid_geohashes)]
        valid_part = day48_window[day48_window["geohash"].isin(valid_geohashes)]

        fold_predictions = []
        for name, pipeline in models.items():
            fitted = clone(pipeline)
            fitted.fit(train_part[features], train_part["demand"])
            prediction = np.clip(fitted.predict(valid_part[features]), 0.0, 1.0)
            fold_predictions.append(prediction)
            records.append(metric_row(name, valid_part["demand"], prediction, f"geohash_seed_{seed}"))

        ensemble_prediction = np.mean(fold_predictions, axis=0)
        records.append(metric_row("MeanModelEnsemble", valid_part["demand"], ensemble_prediction, f"geohash_seed_{seed}"))

    return pd.DataFrame(records)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = add_time_features(pd.read_csv(DATA_DIR / "train.csv"))
    test = add_time_features(pd.read_csv(DATA_DIR / "test.csv"))

    day48 = train[train["day"] == 48].copy()
    day49_prefix = train[(train["day"] == 49) & (train["time_minutes"] <= 120)].copy()
    day48_prefix = day48[day48["time_minutes"] <= 120].copy()
    day48_window = day48[(day48["time_minutes"] >= 135) & (day48["time_minutes"] <= 825)].copy()

    day48_prefix_stats = make_prefix_stats(day48_prefix)
    day49_prefix_stats = make_prefix_stats(day49_prefix)

    day48_model_frame = attach_prefix_stats(day48_window, day48_prefix_stats)
    test_model_frame = attach_prefix_stats(test, day49_prefix_stats)
    numeric, categorical = feature_lists(day48_model_frame)
    features = numeric + categorical

    validation = geohash_holdout_validation(day48_model_frame, numeric, categorical)
    validation_summary = (
        validation.groupby("model", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            mean_mae=("mae", "mean"),
            mean_rmse=("rmse", "mean"),
        )
        .sort_values("mean_score", ascending=False)
    )
    validation.to_csv(REPORT_DIR / "future_window_geohash_validation.csv", index=False)
    validation_summary.to_csv(REPORT_DIR / "future_window_geohash_validation_summary.csv", index=False)

    models = make_models(numeric, categorical)
    final_predictions = {}
    for name, pipeline in models.items():
        print(f"Training prefix-window final model: {name}")
        fitted = clone(pipeline)
        fitted.fit(day48_model_frame[features], day48_model_frame["demand"])
        final_predictions[name] = np.clip(fitted.predict(test_model_frame[features]), 0.0, 1.0)

    ensemble_prediction = np.mean(list(final_predictions.values()), axis=0)
    base = pd.read_csv(OUTPUT_DIR / "leaderboard_blend_base_60_lag_40.csv")
    if not base["Index"].equals(test["Index"]):
        base = pd.read_csv(OUTPUT_DIR / "submission.csv")

    output_specs = {
        "future_window_prefix_ensemble.csv": ensemble_prediction,
        "future_window_prefix_lgbm.csv": final_predictions["LightGBM"],
        "future_window_prefix_xgb.csv": final_predictions["XGBoost"],
        "future_window_blend_current_90_prefix_10.csv": 0.9 * base["demand"].to_numpy() + 0.1 * ensemble_prediction,
        "future_window_blend_current_85_prefix_15.csv": 0.85 * base["demand"].to_numpy() + 0.15 * ensemble_prediction,
        "future_window_blend_current_80_prefix_20.csv": 0.8 * base["demand"].to_numpy() + 0.2 * ensemble_prediction,
    }

    manifest_rows = []
    for filename, prediction in output_specs.items():
        prediction = np.clip(prediction, 0.0, 1.0)
        pd.DataFrame({"Index": test["Index"], "demand": prediction}).to_csv(OUTPUT_DIR / filename, index=False)
        manifest_rows.append(
            {
                "file": filename,
                "mean": float(np.mean(prediction)),
                "std": float(np.std(prediction)),
                "min": float(np.min(prediction)),
                "max": float(np.max(prediction)),
                "mean_abs_delta_from_current_best": float(np.mean(np.abs(prediction - base["demand"].to_numpy()))),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "future_window_manifest.csv", index=False)

    print("Validation summary:")
    print(validation_summary.to_string(index=False))
    print("Submission manifest:")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
