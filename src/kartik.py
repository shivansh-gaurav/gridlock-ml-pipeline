"""Traffic demand prediction pipeline.

Run:
    python3 traffic_demand_pipeline.py --train dataset/train.csv

The script follows the requested phases: EDA, feature engineering, baseline,
XGBoost/LightGBM modeling, evaluation, SHAP, and optional lag features.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gridlock")

import geohash
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterSampler, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


RANDOM_STATE = 42
TARGET = "demand"


@dataclass
class Metrics:
    rmse: float
    mae: float
    r2: float


class TrafficFeatureEngineer:
    """Feature engineering with train-fitted imputers, dummies, and scaler."""

    def __init__(self) -> None:
        self.road_type_mode: str | None = None
        self.weather_mode: str | None = None
        self.temp_by_weather: pd.Series | None = None
        self.global_temp_median: float | None = None
        self.geohash_counts: pd.Series | None = None
        self.geohash_mean_demand: pd.Series | None = None
        self.global_mean_demand: float | None = None
        self.dummy_columns: list[str] | None = None
        self.scaler = StandardScaler()
        self.numeric_to_scale = ["Temperature", "NumberofLanes"]

    def fit(self, df: pd.DataFrame) -> "TrafficFeatureEngineer":
        work = self._basic_features(df, fit_modes=True)
        self.temp_by_weather = work.groupby("Weather")["Temperature"].median()
        self.global_temp_median = float(work["Temperature"].median())
        if TARGET in work.columns:
            self.geohash_counts = work.groupby("geohash").size()
            self.geohash_mean_demand = work.groupby("geohash")[TARGET].mean()
            self.global_mean_demand = float(work[TARGET].mean())
        work = self._impute_temperature(work)
        work = self._add_geohash_aggregates(work)
        features = self._encode(work, fit_columns=True)
        self.scaler.fit(features[self.numeric_to_scale])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.dummy_columns is None:
            raise RuntimeError("TrafficFeatureEngineer must be fit before transform.")

        work = self._basic_features(df, fit_modes=False)
        work = self._impute_temperature(work)
        work = self._add_geohash_aggregates(work)
        features = self._encode(work, fit_columns=False)
        features[self.numeric_to_scale] = self.scaler.transform(
            features[self.numeric_to_scale]
        )
        return features

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _basic_features(self, df: pd.DataFrame, fit_modes: bool) -> pd.DataFrame:
        work = df.copy()

        work[["hour", "minute"]] = work["timestamp"].apply(parse_timestamp).tolist()
        work[["lat", "lon"]] = work["geohash"].apply(decode_geohash).tolist()

        if fit_modes:
            self.road_type_mode = mode_or_default(work["RoadType"], "Unknown")
            self.weather_mode = mode_or_default(work["Weather"], "Unknown")

        work["RoadType"] = work["RoadType"].fillna(self.road_type_mode)
        work["Weather"] = work["Weather"].fillna(self.weather_mode)

        work["is_rush_hour"] = (
            work["hour"].between(7, 9) | work["hour"].between(17, 19)
        ).astype(int)
        work["is_night"] = ((work["hour"] >= 22) | (work["hour"] <= 6)).astype(int)
        work["minute_of_day"] = work["hour"] * 60 + work["minute"]
        work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
        work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)
        work["minute_sin"] = np.sin(2 * np.pi * work["minute_of_day"] / 1440)
        work["minute_cos"] = np.cos(2 * np.pi * work["minute_of_day"] / 1440)
        work["LargeVehicles"] = (work["LargeVehicles"] == "Allowed").astype(int)
        work["Landmarks"] = (work["Landmarks"] == "Yes").astype(int)
        return work

    def _impute_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        medians = self.temp_by_weather
        global_median = self.global_temp_median
        if medians is None or global_median is None:
            raise RuntimeError("Temperature imputers are not fitted.")

        weather_medians = work["Weather"].map(medians)
        work["Temperature"] = work["Temperature"].fillna(weather_medians)
        work["Temperature"] = work["Temperature"].fillna(global_median)
        return work

    def _add_geohash_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        if self.geohash_counts is None:
            work["geohash_count"] = 0.0
        else:
            work["geohash_count"] = work["geohash"].map(self.geohash_counts).fillna(0.0)

        if self.geohash_mean_demand is None or self.global_mean_demand is None:
            work["geohash_mean_demand"] = 0.0
        else:
            work["geohash_mean_demand"] = (
                work["geohash"]
                .map(self.geohash_mean_demand)
                .fillna(self.global_mean_demand)
            )
        return work

    def _encode(self, df: pd.DataFrame, fit_columns: bool) -> pd.DataFrame:
        base_cols = [
            "day",
            "hour",
            "minute",
            "minute_of_day",
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "is_rush_hour",
            "is_night",
            "LargeVehicles",
            "Landmarks",
            "Temperature",
            "NumberofLanes",
            "lat",
            "lon",
            "geohash_count",
            "geohash_mean_demand",
        ]
        features = df[base_cols].copy()
        dummies = pd.get_dummies(df[["RoadType", "Weather"]], drop_first=False)

        if fit_columns:
            self.dummy_columns = dummies.columns.tolist()
        elif self.dummy_columns is not None:
            dummies = dummies.reindex(columns=self.dummy_columns, fill_value=0)

        features = pd.concat([features, dummies.astype(int)], axis=1)
        return features


def parse_timestamp(value: str) -> tuple[int, int]:
    hour, minute = str(value).split(":")
    return int(hour), int(minute)


def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)


def mode_or_default(series: pd.Series, default: str) -> str:
    mode = series.dropna().mode()
    return str(mode.iloc[0]) if not mode.empty else default


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def run_eda(df: pd.DataFrame, plot_dir: Path) -> None:
    ensure_dir(plot_dir)
    print("\nPHASE 1 - EDA")
    print(f"Shape: {df.shape}")
    print("\nDtypes:")
    print(df.dtypes)
    print("\nNull counts:")
    print(df.isna().sum())

    eda = df.copy()
    eda[["hour", "minute"]] = eda["timestamp"].apply(parse_timestamp).tolist()
    eda[["lat", "lon"]] = eda["geohash"].apply(decode_geohash).tolist()

    plt.figure(figsize=(9, 5))
    sns.histplot(eda[TARGET], bins=50, kde=True)
    plt.title("Demand Distribution")
    plt.tight_layout()
    plt.savefig(plot_dir / "01_demand_distribution.png", dpi=160)
    plt.close()

    for col, filename in [
        ("RoadType", "02_demand_by_roadtype.png"),
        ("Weather", "03_demand_by_weather.png"),
    ]:
        plt.figure(figsize=(10, 5))
        sns.barplot(data=eda, x=col, y=TARGET, estimator="mean", errorbar=None)
        plt.title(f"Average Demand by {col}")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plot_dir / filename, dpi=160)
        plt.close()

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=eda.groupby("hour", as_index=False)[TARGET].mean(), x="hour", y=TARGET)
    plt.title("Average Demand by Hour")
    plt.tight_layout()
    plt.savefig(plot_dir / "04_demand_by_hour.png", dpi=160)
    plt.close()

    print(f"EDA plots saved to {plot_dir}")


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_df, valid_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    fe = TrafficFeatureEngineer()
    x_train = fe.fit_transform(train_df)
    x_valid = fe.transform(valid_df)
    y_train = train_df[TARGET].copy()
    y_valid = valid_df[TARGET].copy()
    return x_train, x_valid, y_train, y_valid


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Metrics:
    return Metrics(
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def print_metrics(name: str, metrics: Metrics) -> None:
    print(f"{name} RMSE: {metrics.rmse:.6f}")
    print(f"{name} MAE:  {metrics.mae:.6f}")
    print(f"{name} R2:   {metrics.r2:.6f}")


def score_out_of_100(metrics: Metrics) -> float:
    return float(np.clip(metrics.r2 * 100.0, 0.0, 100.0))


def plot_predicted_vs_actual(
    y_true: pd.Series, y_pred: np.ndarray, plot_dir: Path, filename: str
) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.35, edgecolor=None)
    low = min(float(y_true.min()), float(np.min(y_pred)))
    high = max(float(y_true.max()), float(np.max(y_pred)))
    plt.plot([low, high], [low, high], color="red", linewidth=1)
    plt.xlabel("Actual demand")
    plt.ylabel("Predicted demand")
    plt.title("Predicted vs Actual Demand")
    plt.tight_layout()
    plt.savefig(plot_dir / filename, dpi=160)
    plt.close()


def run_baseline(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    plot_dir: Path,
) -> tuple[LinearRegression, Metrics, np.ndarray]:
    print("\nPHASE 3 - BASELINE")
    model = LinearRegression()
    model.fit(x_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        preds = model.predict(x_valid)
    metrics = compute_metrics(y_valid, preds)
    print_metrics("LinearRegression", metrics)
    plot_predicted_vs_actual(y_valid, preds, plot_dir, "05_baseline_predicted_vs_actual.png")
    return model, metrics, preds


def sampled_xgb_params(n_iter: int) -> list[dict]:
    param_dist = {
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.02, 0.03, 0.05, 0.07, 0.1],
        "n_estimators": [300, 600, 900],
        "subsample": [0.75, 0.85, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
    }
    sampler = ParameterSampler(param_dist, n_iter=n_iter, random_state=RANDOM_STATE)
    return list(sampler)


def default_xgb_params() -> dict:
    return {
        "subsample": 0.75,
        "n_estimators": 600,
        "max_depth": 4,
        "learning_rate": 0.05,
        "colsample_bytree": 0.75,
    }


def build_xgb(params: dict, early_stopping: bool = True) -> XGBRegressor:
    model_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "random_state": RANDOM_STATE,
        "n_jobs": 4,
        "tree_method": "hist",
        **params,
    }
    if early_stopping:
        model_params["early_stopping_rounds"] = 50
    return XGBRegressor(
        **model_params,
    )


def build_lgbm(params: dict | None = None) -> lgb.LGBMRegressor:
    model_params = {
        "objective": "regression",
        "n_estimators": 1800,
        "learning_rate": 0.025,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.01,
        "reg_lambda": 1.0,
        "random_state": RANDOM_STATE,
        "n_jobs": 4,
        "verbose": -1,
    }
    if params:
        model_params.update(params)
    return lgb.LGBMRegressor(**model_params)


def run_xgb_cv(
    df: pd.DataFrame, n_splits: int, tune_iter: int
) -> tuple[dict, list[float], float]:
    print("\nPHASE 4 - MAIN MODEL")
    print(f"XGBoost tuning: {tune_iter} sampled parameter sets, {n_splits}-fold KFold")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y = df[TARGET].copy()
    best_params: dict | None = None
    best_scores: list[float] | None = None
    best_mean = np.inf
    start = time.perf_counter()

    for i, params in enumerate(sampled_xgb_params(tune_iter), start=1):
        fold_scores: list[float] = []
        for train_idx, valid_idx in kf.split(df):
            train_fold = df.iloc[train_idx]
            valid_fold = df.iloc[valid_idx]
            fe = TrafficFeatureEngineer()
            x_train = fe.fit_transform(train_fold)
            x_valid = fe.transform(valid_fold)
            model = build_xgb(params)
            model.fit(
                x_train,
                y.iloc[train_idx],
                eval_set=[(x_valid, y.iloc[valid_idx])],
                verbose=False,
            )
            preds = model.predict(x_valid)
            fold_scores.append(float(np.sqrt(mean_squared_error(y.iloc[valid_idx], preds))))

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        print(f"  XGB set {i:02d}: RMSE {mean_score:.6f} +/- {std_score:.6f} | {params}")
        if mean_score < best_mean:
            best_mean = mean_score
            best_params = params
            best_scores = fold_scores

    elapsed = time.perf_counter() - start
    if best_params is None or best_scores is None:
        raise RuntimeError("No XGBoost parameter set was evaluated.")

    print(f"Best XGB params: {best_params}")
    print(f"CV RMSE mean +/- std: {np.mean(best_scores):.6f} +/- {np.std(best_scores):.6f}")
    print(f"XGBoost CV time: {elapsed:.1f}s")
    return best_params, best_scores, elapsed


def compare_lightgbm_speed(
    df: pd.DataFrame, xgb_scores: Iterable[float], n_splits: int
) -> tuple[list[float], float]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    y = df[TARGET].copy()
    scores: list[float] = []
    start = time.perf_counter()

    for train_idx, valid_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        valid_fold = df.iloc[valid_idx]
        fe = TrafficFeatureEngineer()
        x_train = fe.fit_transform(train_fold)
        x_valid = fe.transform(valid_fold)
        model = build_lgbm(
            {
                "n_estimators": 800,
                "learning_rate": 0.05,
                "max_depth": -1,
                "num_leaves": 63,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            }
        )
        model.fit(
            x_train,
            y.iloc[train_idx],
            eval_set=[(x_valid, y.iloc[valid_idx])],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(x_valid)
        scores.append(float(np.sqrt(mean_squared_error(y.iloc[valid_idx], preds))))

    elapsed = time.perf_counter() - start
    xgb_mean = float(np.mean(list(xgb_scores)))
    print(
        f"LGBM CV RMSE mean +/- std: {np.mean(scores):.6f} +/- {np.std(scores):.6f}"
    )
    print(f"LGBM CV time: {elapsed:.1f}s; XGB best CV RMSE was {xgb_mean:.6f}")
    return scores, elapsed


def train_final_xgb(
    x_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    params: dict,
) -> tuple[XGBRegressor, Metrics, np.ndarray]:
    model = build_xgb(params)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    preds = model.predict(x_valid)
    metrics = compute_metrics(y_valid, preds)
    return model, metrics, preds


def run_evaluation(
    model: XGBRegressor,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    preds: np.ndarray,
    baseline_metrics: Metrics,
    final_metrics: Metrics,
    plot_dir: Path,
    shap_sample: int,
) -> None:
    print("\nPHASE 5 - EVALUATION")
    print_metrics("Baseline LinearRegression", baseline_metrics)
    print_metrics("Final XGBRegressor", final_metrics)

    improvement = baseline_metrics.rmse - final_metrics.rmse
    print(f"RMSE improvement vs baseline: {improvement:.6f}")

    importances = pd.Series(model.feature_importances_, index=x_valid.columns)
    importances = importances.sort_values(ascending=False).head(25)
    plt.figure(figsize=(10, 7))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Top XGBoost Feature Importances")
    plt.tight_layout()
    plt.savefig(plot_dir / "06_xgb_feature_importances.png", dpi=160)
    plt.close()

    residuals = y_valid.to_numpy() - preds
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=preds, y=residuals, alpha=0.35, edgecolor=None)
    plt.axhline(0, color="red", linewidth=1)
    plt.xlabel("Predicted demand")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted Demand")
    plt.tight_layout()
    plt.savefig(plot_dir / "07_residuals_vs_predicted.png", dpi=160)
    plt.close()

    sample = x_valid.sample(min(shap_sample, len(x_valid)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(plot_dir / "08_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Evaluation plots saved to {plot_dir}")


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    lagged = df.copy()
    lagged[["hour", "minute"]] = lagged["timestamp"].apply(parse_timestamp).tolist()
    lagged = lagged.sort_values(["geohash", "day", "hour", "minute"])
    group = lagged.groupby(["geohash", "day"])[TARGET]
    lagged["demand_lag_1"] = group.shift(1)
    lagged["demand_lag_2"] = group.shift(2)
    lagged["demand_roll_mean_3"] = group.shift(1).rolling(3, min_periods=1).mean()
    for col in ["demand_lag_1", "demand_lag_2", "demand_roll_mean_3"]:
        lagged[col] = lagged[col].fillna(lagged[col].median())
    return lagged.sort_index()


class LagTrafficFeatureEngineer(TrafficFeatureEngineer):
    def _encode(self, df: pd.DataFrame, fit_columns: bool) -> pd.DataFrame:
        features = super()._encode(df, fit_columns)
        for col in ["demand_lag_1", "demand_lag_2", "demand_roll_mean_3"]:
            if col in df.columns:
                features[col] = df[col].to_numpy()
        return features


def run_optional_lag_model(
    df: pd.DataFrame, params: dict, reference_rmse: float
) -> Metrics:
    print("\nPHASE 6 - OPTIONAL LAG FEATURES")
    lagged = add_lag_features(df)
    train_df, valid_df = train_test_split(lagged, test_size=0.2, random_state=RANDOM_STATE)
    fe = LagTrafficFeatureEngineer()
    x_train = fe.fit_transform(train_df)
    x_valid = fe.transform(valid_df)
    y_train = train_df[TARGET].copy()
    y_valid = valid_df[TARGET].copy()

    model = build_xgb(params)
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    preds = model.predict(x_valid)
    metrics = compute_metrics(y_valid, preds)
    print_metrics("XGB with lag features", metrics)
    print(f"RMSE delta vs final XGB: {metrics.rmse - reference_rmse:+.6f}")
    return metrics


def run_local_score(
    df: pd.DataFrame,
    model_name: str,
    plot_dir: Path,
) -> Metrics:
    print("\nLOCAL 80/20 VALIDATION SCORE")
    print("Split: 80% train / 20% validation, random_state=42")
    x_train, x_valid, y_train, y_valid = split_features(df)

    if model_name == "linear":
        model = LinearRegression()
        model.fit(x_train, y_train)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            preds = model.predict(x_valid)
    elif model_name == "xgb":
        model = build_xgb(default_xgb_params())
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
        preds = model.predict(x_valid)
    elif model_name == "lgbm":
        model = build_lgbm()
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        preds = model.predict(x_valid)
    else:
        raise ValueError(f"Unsupported local score model: {model_name}")

    preds = np.clip(preds, 0.0, 1.0)
    metrics = compute_metrics(y_valid, preds)
    print_metrics(model_name.upper(), metrics)
    print(f"Score out of 100: {score_out_of_100(metrics):.2f}")
    plot_predicted_vs_actual(
        y_valid, preds, plot_dir, f"local_score_{model_name}_predicted_vs_actual.png"
    )
    return metrics


def write_test_predictions(
    train_df: pd.DataFrame,
    test_path: Path,
    output_path: Path,
    params: dict,
    model_name: str,
) -> pd.DataFrame:
    print("\nPREDICT TEST")
    test_df = load_data(test_path)
    if TARGET in test_df.columns:
        raise ValueError(f"Test file should not contain target column '{TARGET}'.")

    fe = TrafficFeatureEngineer()
    x_train_full = fe.fit_transform(train_df)
    x_test = fe.transform(test_df)
    y_train_full = train_df[TARGET].copy()

    if model_name == "lgbm":
        model = build_lgbm()
        model.fit(x_train_full, y_train_full)
    elif model_name == "xgb":
        model = build_xgb(params, early_stopping=False)
        model.fit(x_train_full, y_train_full, verbose=False)
    else:
        raise ValueError(f"Unsupported submission model: {model_name}")
    predictions = np.clip(model.predict(x_test), 0.0, 1.0)

    submission = pd.DataFrame(
        {
            "Index": test_df["Index"].to_numpy()
            if "Index" in test_df.columns
            else np.arange(len(test_df)),
            TARGET: predictions,
        }
    )
    ensure_dir(output_path.parent)
    submission.to_csv(output_path, index=False)
    print(f"Submission model: {model_name.upper()}")
    print(f"Saved {len(submission)} predictions to {output_path}")
    print(submission.head().to_string(index=False))
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Traffic demand prediction pipeline")
    parser.add_argument("--train", type=Path, default=Path("dataset/train.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--xgb-tune-iter", type=int, default=8)
    parser.add_argument("--shap-sample", type=int, default=2000)
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--run-lag", action="store_true")
    parser.add_argument("--predict-test", type=Path, help="Path to test.csv for prediction")
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Skip validation phases and train the submission model on full train.csv",
    )
    parser.add_argument(
        "--local-score-only",
        action="store_true",
        help="Train on 80% of train.csv and print a 20% validation score out of 100",
    )
    parser.add_argument(
        "--local-model",
        choices=["linear", "xgb", "lgbm"],
        default="lgbm",
        help="Model to use with --local-score-only",
    )
    parser.add_argument(
        "--submission-output",
        type=Path,
        default=Path("artifacts/submission.csv"),
        help="Where to write predictions when --predict-test is provided",
    )
    parser.add_argument(
        "--submission-model",
        choices=["lgbm", "xgb"],
        default="lgbm",
        help="Model to train on full train.csv for --predict-test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_dir = args.artifacts / "plots"
    ensure_dir(plot_dir)

    df = load_data(args.train)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in {args.train}")

    if args.local_score_only:
        run_local_score(df, args.local_model, plot_dir)
        return

    if args.predict_only:
        if not args.predict_test:
            raise ValueError("--predict-only requires --predict-test")
        write_test_predictions(
            df,
            args.predict_test,
            args.submission_output,
            default_xgb_params(),
            args.submission_model,
        )
        return

    if not args.skip_eda:
        run_eda(df, plot_dir)

    print("\nPHASE 2 - FEATURE ENGINEERING")
    x_train, x_valid, y_train, y_valid = split_features(df)
    print(f"Training feature matrix: {x_train.shape}")
    print(f"Validation feature matrix: {x_valid.shape}")
    print(f"Feature columns: {x_train.columns.tolist()}")

    _, baseline_metrics, _ = run_baseline(x_train, x_valid, y_train, y_valid, plot_dir)

    best_params, xgb_scores, _ = run_xgb_cv(
        df, n_splits=args.n_splits, tune_iter=args.xgb_tune_iter
    )
    compare_lightgbm_speed(df, xgb_scores, n_splits=args.n_splits)

    final_model, final_metrics, final_preds = train_final_xgb(
        x_train, x_valid, y_train, y_valid, best_params
    )

    if args.skip_shap:
        print("\nPHASE 5 - EVALUATION")
        print_metrics("Baseline LinearRegression", baseline_metrics)
        print_metrics("Final XGBRegressor", final_metrics)
        print("SHAP skipped by --skip-shap")
    else:
        run_evaluation(
            final_model,
            x_valid,
            y_valid,
            final_preds,
            baseline_metrics,
            final_metrics,
            plot_dir,
            args.shap_sample,
        )

    if args.run_lag:
        run_optional_lag_model(df, best_params, final_metrics.rmse)

    if args.predict_test:
        write_test_predictions(
            df,
            args.predict_test,
            args.submission_output,
            best_params,
            args.submission_model,
        )


if __name__ == "__main__":
    main()
