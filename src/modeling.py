from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from eda import add_geohash_features, add_time_features


DATA_DIR = Path("dataset")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"
OUTPUT_DIR = Path("outputs")
MODEL_DIR = OUTPUT_DIR / "models"

RANDOM_STATE = 42
TARGET = "demand"


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Leakage-safe target mean encoder when used inside a sklearn Pipeline."""

    def __init__(self, columns: list[str] | None = None, smoothing: float = 20.0):
        self.columns = columns
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SmoothedTargetEncoder":
        frame = self._to_frame(X)
        self.columns_ = list(self.columns or frame.columns)
        self.global_mean_ = float(pd.Series(y).mean())
        self.mappings_: dict[str, pd.Series] = {}

        target = pd.Series(y).reset_index(drop=True)
        for column in self.columns_:
            values = frame[column].reset_index(drop=True).fillna("__MISSING__").astype(str)
            stats = target.groupby(values).agg(["count", "mean"])
            encoded = (
                stats["count"] * stats["mean"] + self.smoothing * self.global_mean_
            ) / (stats["count"] + self.smoothing)
            self.mappings_[column] = encoded
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        frame = self._to_frame(X)
        encoded_columns = []
        for column in self.columns_:
            values = frame[column].fillna("__MISSING__").astype(str)
            encoded = values.map(self.mappings_[column]).fillna(self.global_mean_)
            encoded_columns.append(encoded.to_numpy(dtype=float))
        return np.column_stack(encoded_columns)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        return np.array([f"{column}_target_mean" for column in self.columns_], dtype=object)

    def _to_frame(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        columns = self.columns or [f"col_{idx}" for idx in range(np.asarray(X).shape[1])]
        return pd.DataFrame(X, columns=columns)


@dataclass
class FeatureConfig:
    numeric: list[str]
    one_hot: list[str]
    target_encoded: list[str]


def parse_timestamp_to_minutes(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = add_geohash_features(add_time_features(df))

    enriched["time_sin"] = np.sin(2 * np.pi * enriched["time_minutes"] / (24 * 60))
    enriched["time_cos"] = np.cos(2 * np.pi * enriched["time_minutes"] / (24 * 60))
    enriched["minute_sin"] = np.sin(2 * np.pi * enriched["minute"] / 60)
    enriched["minute_cos"] = np.cos(2 * np.pi * enriched["minute"] / 60)

    enriched["Temperature_missing"] = enriched["Temperature"].isna().astype(int)
    enriched["RoadType_missing"] = enriched["RoadType"].isna().astype(int)
    enriched["Weather_missing"] = enriched["Weather"].isna().astype(int)

    enriched["LargeVehicles_allowed"] = (enriched["LargeVehicles"] == "Allowed").astype(int)
    enriched["Landmarks_yes"] = (enriched["Landmarks"] == "Yes").astype(int)
    enriched["lanes_x_large_allowed"] = enriched["NumberofLanes"] * enriched["LargeVehicles_allowed"]

    road = enriched["RoadType"].fillna("Missing").astype(str)
    weather = enriched["Weather"].fillna("Missing").astype(str)
    hour = enriched["hour"].astype(str)
    lanes = enriched["NumberofLanes"].astype(str)

    enriched["road_weather"] = road + "__" + weather
    enriched["weather_hour"] = weather + "__h" + hour
    enriched["road_lanes"] = road + "__lanes" + lanes
    enriched["geo5_hour"] = enriched["geohash_prefix_5"].astype(str) + "__h" + hour
    enriched["geo4_hour"] = enriched["geohash_prefix_4"].astype(str) + "__h" + hour
    return enriched


def get_feature_config() -> FeatureConfig:
    return FeatureConfig(
        numeric=[
            "time_minutes",
            "hour",
            "minute",
            "slot_15m",
            "hour_sin",
            "hour_cos",
            "time_sin",
            "time_cos",
            "minute_sin",
            "minute_cos",
            "NumberofLanes",
            "Temperature",
            "latitude",
            "longitude",
            "Temperature_missing",
            "RoadType_missing",
            "Weather_missing",
            "LargeVehicles_allowed",
            "Landmarks_yes",
            "lanes_x_large_allowed",
        ],
        one_hot=[
            "RoadType",
            "LargeVehicles",
            "Landmarks",
            "Weather",
            "geohash_prefix_4",
        ],
        target_encoded=[
            "geohash",
            "geohash_prefix_5",
            "timestamp",
            "road_weather",
            "weather_hour",
            "road_lanes",
            "geo5_hour",
            "geo4_hour",
        ],
    )


def make_preprocessor(config: FeatureConfig, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), config.numeric),
            (
                "one_hot",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                config.one_hot,
            ),
            (
                "target_mean",
                SmoothedTargetEncoder(columns=config.target_encoded, smoothing=30.0),
                config.target_encoded,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model: Any, config: FeatureConfig, scale_numeric: bool = False) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(config, scale_numeric=scale_numeric)),
            ("model", model),
        ]
    )


def make_models(config: FeatureConfig) -> dict[str, Pipeline]:
    return {
        "LinearRegression": make_pipeline(LinearRegression(), config, scale_numeric=True),
        "RandomForest": make_pipeline(
            RandomForestRegressor(
                n_estimators=140,
                max_depth=18,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            config,
        ),
        "XGBoost": make_pipeline(
            XGBRegressor(
                n_estimators=420,
                max_depth=6,
                learning_rate=0.045,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                eval_metric="rmse",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            config,
        ),
        "LightGBM": make_pipeline(
            LGBMRegressor(
                n_estimators=520,
                learning_rate=0.045,
                num_leaves=63,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_samples=25,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            config,
        ),
        "CatBoost": make_pipeline(
            CatBoostRegressor(
                iterations=520,
                depth=6,
                learning_rate=0.045,
                loss_function="RMSE",
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            ),
            config,
        ),
    }


def make_time_folds(df: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    sequence = (df["day"] - df["day"].min()) * 96 + df["slot_15m"]
    fold_specs = [
        ("day48_midday_to_afternoon", 47, 48, 63),
        ("day48_afternoon_to_evening", 63, 64, 79),
        ("day48_evening_to_night", 79, 80, 95),
        ("day49_early_future_holdout", 95, 96, 104),
    ]

    folds = []
    for name, train_end, valid_start, valid_end in fold_specs:
        train_idx = df.index[sequence <= train_end].to_numpy()
        valid_idx = df.index[(sequence >= valid_start) & (sequence <= valid_end)].to_numpy()
        if len(train_idx) and len(valid_idx):
            folds.append((name, train_idx, valid_idx))
    return folds


def metric_dict(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    clipped = np.clip(y_pred, 0.0, 1.0)
    return {
        "r2": r2_score(y_true, clipped),
        "mae": mean_absolute_error(y_true, clipped),
        "rmse": float(np.sqrt(mean_squared_error(y_true, clipped))),
    }


def evaluate_models(
    models: dict[str, Pipeline],
    train_df: pd.DataFrame,
    config: FeatureConfig,
    folds: list[tuple[str, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    feature_columns = config.numeric + config.one_hot + config.target_encoded
    X = train_df[feature_columns]
    y = train_df[TARGET]

    records = []
    for model_name, pipeline in models.items():
        print(f"Benchmarking {model_name}...")
        for fold_name, train_idx, valid_idx in folds:
            fitted = clone(pipeline)
            fitted.fit(X.loc[train_idx], y.loc[train_idx])
            predictions = fitted.predict(X.loc[valid_idx])
            metrics = metric_dict(y.loc[valid_idx], predictions)
            records.append(
                {
                    "model": model_name,
                    "fold": fold_name,
                    "train_rows": len(train_idx),
                    "valid_rows": len(valid_idx),
                    **metrics,
                }
            )

    return pd.DataFrame(records)


def summarize_cv(cv_results: pd.DataFrame) -> pd.DataFrame:
    return (
        cv_results.groupby("model", as_index=False)
        .agg(
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
        )
        .sort_values("mean_rmse")
    )


def tuning_grid(model_name: str) -> list[dict[str, Any]]:
    if model_name == "LightGBM":
        return [
            {"n_estimators": 700, "learning_rate": 0.035, "num_leaves": 63, "min_child_samples": 20, "reg_lambda": 1.0},
            {"n_estimators": 900, "learning_rate": 0.025, "num_leaves": 95, "min_child_samples": 30, "reg_lambda": 2.0},
            {"n_estimators": 650, "learning_rate": 0.04, "num_leaves": 47, "min_child_samples": 15, "reg_lambda": 0.5},
            {"n_estimators": 800, "learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 40, "reg_lambda": 3.0},
        ]
    if model_name == "XGBoost":
        return [
            {"n_estimators": 600, "learning_rate": 0.035, "max_depth": 5, "subsample": 0.9, "colsample_bytree": 0.85, "reg_lambda": 1.0},
            {"n_estimators": 750, "learning_rate": 0.025, "max_depth": 6, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0},
            {"n_estimators": 500, "learning_rate": 0.045, "max_depth": 7, "subsample": 0.8, "colsample_bytree": 0.9, "reg_lambda": 1.5},
        ]
    if model_name == "CatBoost":
        return [
            {"iterations": 700, "learning_rate": 0.035, "depth": 6, "l2_leaf_reg": 3.0},
            {"iterations": 850, "learning_rate": 0.025, "depth": 7, "l2_leaf_reg": 5.0},
            {"iterations": 650, "learning_rate": 0.04, "depth": 8, "l2_leaf_reg": 8.0},
        ]
    if model_name == "RandomForest":
        return [
            {"n_estimators": 220, "max_depth": 20, "min_samples_leaf": 3},
            {"n_estimators": 220, "max_depth": None, "min_samples_leaf": 5},
        ]
    return [{}]


def tune_model(
    base_model_name: str,
    base_pipeline: Pipeline,
    train_df: pd.DataFrame,
    config: FeatureConfig,
    latest_fold: tuple[str, np.ndarray, np.ndarray],
) -> tuple[Pipeline, pd.DataFrame]:
    feature_columns = config.numeric + config.one_hot + config.target_encoded
    X = train_df[feature_columns]
    y = train_df[TARGET]
    fold_name, train_idx, valid_idx = latest_fold

    records = []
    best_pipeline = None
    best_rmse = np.inf

    for trial_id, params in enumerate(tuning_grid(base_model_name), start=1):
        candidate = clone(base_pipeline)
        if params:
            candidate.set_params(**{f"model__{key}": value for key, value in params.items()})
        print(f"Tuning {base_model_name} trial {trial_id}: {params}")
        candidate.fit(X.loc[train_idx], y.loc[train_idx])
        predictions = candidate.predict(X.loc[valid_idx])
        metrics = metric_dict(y.loc[valid_idx], predictions)
        records.append(
            {
                "model": base_model_name,
                "trial": trial_id,
                "fold": fold_name,
                "params": json.dumps(params, sort_keys=True),
                **metrics,
            }
        )
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_pipeline = candidate

    assert best_pipeline is not None
    return best_pipeline, pd.DataFrame(records).sort_values("rmse")


def get_feature_names(pipeline: Pipeline) -> np.ndarray:
    preprocessor = pipeline.named_steps["preprocess"]
    return preprocessor.get_feature_names_out()


def save_feature_importance(pipeline: Pipeline, model_name: str) -> pd.DataFrame:
    feature_names = get_feature_names(pipeline)
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
        importance_type = "importance"
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
        importance_type = "abs_coefficient"
    else:
        return pd.DataFrame()

    importance = (
        pd.DataFrame({"feature": feature_names, importance_type: values})
        .sort_values(importance_type, ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORT_DIR / "model_feature_importance.csv", index=False)

    top = importance.head(30)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top, y="feature", x=importance_type, color="#4c78a8")
    plt.title(f"Top Feature Importance: {model_name}")
    plt.xlabel(importance_type.replace("_", " ").title())
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "09_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close()
    return importance


def error_analysis(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    config: FeatureConfig,
    latest_fold: tuple[str, np.ndarray, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_columns = config.numeric + config.one_hot + config.target_encoded
    _, _, valid_idx = latest_fold
    valid = train_df.loc[valid_idx].copy()
    predictions = np.clip(pipeline.predict(valid[feature_columns]), 0.0, 1.0)
    valid["prediction"] = predictions
    valid["residual"] = valid[TARGET] - valid["prediction"]
    valid["absolute_error"] = valid["residual"].abs()
    valid["demand_band"] = pd.qcut(valid[TARGET], q=5, duplicates="drop")

    grouped_parts = []
    for column in ["hour", "RoadType", "Weather", "demand_band"]:
        summary = (
            valid.groupby(column, dropna=False, observed=True)
            .agg(
                rows=(TARGET, "size"),
                mean_actual=(TARGET, "mean"),
                mean_prediction=("prediction", "mean"),
                mae=("absolute_error", "mean"),
                bias=("residual", "mean"),
            )
            .reset_index()
        )
        summary.insert(0, "segment", column)
        summary = summary.rename(columns={column: "segment_value"})
        grouped_parts.append(summary)

    error_summary = pd.concat(grouped_parts, ignore_index=True)
    error_summary.to_csv(REPORT_DIR / "model_error_analysis.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.scatterplot(data=valid.sample(min(len(valid), 6000), random_state=RANDOM_STATE), x=TARGET, y="prediction", alpha=0.35, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], color="black", linewidth=1)
    axes[0].set_title("Actual vs Prediction: Latest Holdout")
    axes[0].set_xlabel("Actual demand")
    axes[0].set_ylabel("Predicted demand")

    hourly = valid.groupby("hour", as_index=False)["absolute_error"].mean()
    sns.lineplot(data=hourly, x="hour", y="absolute_error", marker="o", ax=axes[1])
    axes[1].set_title("MAE By Hour: Latest Holdout")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("MAE")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "10_error_analysis.png", dpi=160, bbox_inches="tight")
    plt.close()
    return valid, error_summary


def feature_selection_experiment(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    config: FeatureConfig,
    latest_fold: tuple[str, np.ndarray, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        empty_metrics = pd.DataFrame(
            [{"experiment": "select_from_model_median", "status": "skipped_no_feature_importance"}]
        )
        return empty_metrics, pd.DataFrame()

    feature_columns = config.numeric + config.one_hot + config.target_encoded
    X = train_df[feature_columns]
    y = train_df[TARGET]
    fold_name, train_idx, valid_idx = latest_fold

    selected_pipeline = Pipeline(
        steps=[
            ("preprocess", clone(pipeline.named_steps["preprocess"])),
            (
                "select",
                SelectFromModel(
                    estimator=clone(model),
                    threshold="median",
                ),
            ),
            ("model", clone(model)),
        ]
    )
    selected_pipeline.fit(X.loc[train_idx], y.loc[train_idx])
    predictions = selected_pipeline.predict(X.loc[valid_idx])
    metrics = metric_dict(y.loc[valid_idx], predictions)

    preprocessor = selected_pipeline.named_steps["preprocess"]
    selector = selected_pipeline.named_steps["select"]
    feature_names = preprocessor.get_feature_names_out()
    selected_features = pd.DataFrame(
        {
            "feature": feature_names,
            "selected": selector.get_support(),
        }
    )
    selected_features.to_csv(REPORT_DIR / "model_feature_selection.csv", index=False)

    selection_metrics = pd.DataFrame(
        [
            {
                "experiment": "select_from_model_median",
                "fold": fold_name,
                "selected_features": int(selected_features["selected"].sum()),
                "total_features": len(selected_features),
                **metrics,
            }
        ]
    )
    selection_metrics.to_csv(REPORT_DIR / "model_feature_selection_metrics.csv", index=False)
    return selection_metrics, selected_features


def write_model_report(
    cv_results: pd.DataFrame,
    cv_summary: pd.DataFrame,
    tuning_results: pd.DataFrame,
    feature_selection_metrics: pd.DataFrame,
    selected_features: pd.DataFrame,
    best_model_name: str,
    feature_importance: pd.DataFrame,
    error_summary: pd.DataFrame,
    submission_path: Path,
) -> Path:
    def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
        shown = df if max_rows is None else df.head(max_rows)
        escaped = shown.astype(object).where(pd.notna(shown), "")
        headers = [str(column) for column in escaped.columns]
        rows = [[str(value) for value in row] for row in escaped.to_numpy()]
        widths = [
            max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
            for idx in range(len(headers))
        ]

        def format_row(values: list[str]) -> str:
            cells = [value.ljust(widths[idx]) for idx, value in enumerate(values)]
            return "| " + " | ".join(cells) + " |"

        divider = "| " + " | ".join("-" * width for width in widths) + " |"
        return "\n".join([format_row(headers), divider, *(format_row(row) for row in rows)])

    lines = [
        "# Traffic Demand Forecasting: Modeling Report",
        "",
        "## Validation Strategy",
        "",
        "The test set is a chronological continuation of the training set, so model selection uses expanding time folds instead of a random split.",
        "Target encodings are fitted inside each fold through the sklearn pipeline, which prevents validation targets from leaking into aggregate features.",
        "",
        "## Cross-Validation Summary",
        "",
        markdown_table(cv_summary),
        "",
        "## Fold-Level Metrics",
        "",
        markdown_table(cv_results.sort_values(["model", "fold"])),
        "",
        "## Hyperparameter Tuning",
        "",
        markdown_table(tuning_results),
        "",
        "## Feature Selection Experiment",
        "",
        markdown_table(feature_selection_metrics),
        "",
        "The final model keeps the full feature set unless this experiment clearly improves the latest chronological holdout. This preserves useful weak signals while still documenting which transformed features can be pruned for a leaner deployment.",
        "",
        "## Selected Model",
        "",
        f"The selected model is **{best_model_name}**, chosen primarily by lowest chronological-validation RMSE, with MAE and R2 used as supporting checks.",
        "",
    ]

    if not feature_importance.empty:
        selected_top = pd.DataFrame()
        if not selected_features.empty and "selected" in selected_features:
            selected_top = selected_features[selected_features["selected"]].head(30)
        lines.extend(
            [
                "## Feature Importance",
                "",
                "![feature_importance](figures/09_feature_importance.png)",
                "",
                markdown_table(feature_importance, max_rows=20),
                "",
            ]
        )
        if not selected_top.empty:
            lines.extend(
                [
                    "### Selected Feature Sample",
                    "",
                    markdown_table(selected_top, max_rows=30),
                    "",
                ]
            )

    lines.extend(
        [
            "## Error Analysis",
            "",
            "![error_analysis](figures/10_error_analysis.png)",
            "",
            markdown_table(error_summary, max_rows=40),
            "",
            "## Final Artifacts",
            "",
            f"- Submission: `{submission_path}`",
            f"- Final model: `{MODEL_DIR / 'final_model.joblib'}`",
            "- Metrics and diagnostics are saved under `reports/`.",
        ]
    )

    path = REPORT_DIR / "modeling_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_train, raw_test = load_raw_data()
    train = engineer_features(raw_train)
    test = engineer_features(raw_test)
    config = get_feature_config()
    feature_columns = config.numeric + config.one_hot + config.target_encoded

    folds = make_time_folds(train)
    fold_manifest = pd.DataFrame(
        [
            {"fold": name, "train_rows": len(train_idx), "valid_rows": len(valid_idx)}
            for name, train_idx, valid_idx in folds
        ]
    )
    fold_manifest.to_csv(REPORT_DIR / "validation_folds.csv", index=False)

    models = make_models(config)
    cv_results = evaluate_models(models, train, config, folds)
    cv_summary = summarize_cv(cv_results)
    cv_results.to_csv(REPORT_DIR / "model_cv_results.csv", index=False)
    cv_summary.to_csv(REPORT_DIR / "model_cv_summary.csv", index=False)

    best_model_name = cv_summary.iloc[0]["model"]
    best_base_pipeline = models[best_model_name]
    latest_fold = folds[-1]
    tuned_pipeline, tuning_results = tune_model(best_model_name, best_base_pipeline, train, config, latest_fold)
    tuning_results.to_csv(REPORT_DIR / "model_tuning_results.csv", index=False)

    feature_importance = save_feature_importance(tuned_pipeline, best_model_name)
    feature_selection_metrics, selected_features = feature_selection_experiment(
        tuned_pipeline,
        train,
        config,
        latest_fold,
    )
    _, error_summary = error_analysis(tuned_pipeline, train, config, latest_fold)

    final_pipeline = clone(tuned_pipeline)
    final_pipeline.fit(train[feature_columns], train[TARGET])
    predictions = np.clip(final_pipeline.predict(test[feature_columns]), 0.0, 1.0)
    submission = pd.DataFrame({"Index": raw_test["Index"], "demand": predictions})
    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)
    joblib.dump(final_pipeline, MODEL_DIR / "final_model.joblib")

    report_path = write_model_report(
        cv_results=cv_results,
        cv_summary=cv_summary,
        tuning_results=tuning_results,
        feature_selection_metrics=feature_selection_metrics,
        selected_features=selected_features,
        best_model_name=best_model_name,
        feature_importance=feature_importance,
        error_summary=error_summary,
        submission_path=submission_path,
    )

    print(f"Best model: {best_model_name}")
    print(f"Wrote CV results to {REPORT_DIR / 'model_cv_summary.csv'}")
    print(f"Wrote modeling report to {report_path}")
    print(f"Wrote submission to {submission_path}")
    print(f"Wrote final model to {MODEL_DIR / 'final_model.joblib'}")


if __name__ == "__main__":
    main()
