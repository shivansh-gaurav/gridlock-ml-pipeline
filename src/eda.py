from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / "matplotlib-cache"))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DATA_DIR = Path("dataset")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"


GEOHASH_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def decode_geohash(geohash: str) -> tuple[float, float]:
    """Decode a geohash into approximate latitude and longitude."""
    lat_interval = [-90.0, 90.0]
    lon_interval = [-180.0, 180.0]
    even_bit = True

    for char in str(geohash).lower():
        value = GEOHASH_BASE32.index(char)
        for mask in [16, 8, 4, 2, 1]:
            if even_bit:
                midpoint = sum(lon_interval) / 2
                if value & mask:
                    lon_interval[0] = midpoint
                else:
                    lon_interval[1] = midpoint
            else:
                midpoint = sum(lat_interval) / 2
                if value & mask:
                    lat_interval[0] = midpoint
                else:
                    lat_interval[1] = midpoint
            even_bit = not even_bit

    latitude = sum(lat_interval) / 2
    longitude = sum(lon_interval) / 2
    return latitude, longitude


def parse_timestamp_to_minutes(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["day"] = enriched["day"].astype(int)
    enriched["time_minutes"] = enriched["timestamp"].map(parse_timestamp_to_minutes)
    enriched["hour"] = enriched["time_minutes"] // 60
    enriched["minute"] = enriched["time_minutes"] % 60
    enriched["slot_15m"] = enriched["time_minutes"] // 15
    enriched["absolute_slot"] = enriched["day"] * 96 + enriched["slot_15m"]
    enriched["hour_sin"] = np.sin(2 * np.pi * enriched["hour"] / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * enriched["hour"] / 24)
    return enriched


def add_geohash_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    decoded = enriched["geohash"].map(decode_geohash)
    enriched["latitude"] = decoded.map(lambda pair: pair[0])
    enriched["longitude"] = decoded.map(lambda pair: pair[1])
    enriched["geohash_prefix_4"] = enriched["geohash"].str[:4]
    enriched["geohash_prefix_5"] = enriched["geohash"].str[:5]
    return enriched


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_train = pd.read_csv(TRAIN_PATH)
    raw_test = pd.read_csv(TEST_PATH)
    sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    train = add_geohash_features(add_time_features(raw_train))
    test = add_geohash_features(add_time_features(raw_test))
    train.attrs["raw_shape"] = raw_train.shape
    test.attrs["raw_shape"] = raw_test.shape
    return train, test, sample_submission


def save_current_figure(filename: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()
    return path


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "missing_count": df.isna().sum(),
                "missing_pct": df.isna().mean().mul(100),
                "dtype": df.dtypes.astype(str),
                "unique_values": df.nunique(dropna=True),
            }
        )
        .sort_values(["missing_pct", "unique_values"], ascending=[False, False])
        .reset_index(names="column")
    )


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T


def category_overlap(train: pd.DataFrame, test: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        train_values = set(train[column].dropna().unique())
        test_values = set(test[column].dropna().unique())
        rows.append(
            {
                "column": column,
                "train_unique": len(train_values),
                "test_unique": len(test_values),
                "test_not_in_train": len(test_values - train_values),
                "train_not_in_test": len(train_values - test_values),
            }
        )
    return pd.DataFrame(rows)


def iqr_outlier_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[column] < lower) | (df[column] > upper)
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": mask.mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def make_dataset_assessment(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame) -> dict[str, pd.DataFrame]:
    categorical_columns = [
        "geohash",
        "RoadType",
        "NumberofLanes",
        "LargeVehicles",
        "Landmarks",
        "Weather",
        "day",
        "timestamp",
    ]

    key_columns = ["geohash", "day", "timestamp"]
    assessment = {
        "train_missing": missing_summary(train),
        "test_missing": missing_summary(test),
        "train_numeric_summary": numeric_summary(train),
        "test_numeric_summary": numeric_summary(test),
        "category_overlap": category_overlap(train, test, categorical_columns),
        "outliers": iqr_outlier_summary(train, ["demand", "Temperature"]),
        "train_duplicate_keys": pd.DataFrame(
            {
                "metric": ["full_row_duplicates", "duplicate_geohash_day_timestamp", "duplicate_index"],
                "value": [
                    int(train.duplicated().sum()),
                    int(train.duplicated(subset=key_columns).sum()),
                    int(train["Index"].duplicated().sum()),
                ],
            }
        ),
        "test_duplicate_keys": pd.DataFrame(
            {
                "metric": ["full_row_duplicates", "duplicate_geohash_day_timestamp", "duplicate_index"],
                "value": [
                    int(test.duplicated().sum()),
                    int(test.duplicated(subset=key_columns).sum()),
                    int(test["Index"].duplicated().sum()),
                ],
            }
        ),
        "file_shapes": pd.DataFrame(
            {
                "dataset": ["train", "test", "sample_submission"],
                "rows": [len(train), len(test), len(sample_submission)],
                "original_columns": [
                    train.attrs.get("raw_shape", train.shape)[1],
                    test.attrs.get("raw_shape", test.shape)[1],
                    sample_submission.shape[1],
                ],
                "columns_after_eda_features": [train.shape[1], test.shape[1], sample_submission.shape[1]],
            }
        ),
    }
    return assessment


def plot_target_distribution(train: pd.DataFrame) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(train["demand"], bins=60, kde=True, ax=axes[0], color="#3366aa")
    axes[0].set_title("Demand Distribution")
    axes[0].set_xlabel("Demand")
    axes[0].set_ylabel("Record count")

    sns.histplot(np.log1p(train["demand"]), bins=60, kde=True, ax=axes[1], color="#2a9d8f")
    axes[1].set_title("Log1p Demand Distribution")
    axes[1].set_xlabel("log1p(Demand)")
    axes[1].set_ylabel("Record count")

    path = save_current_figure("01_target_distribution.png")
    high_tail_pct = (train["demand"] > train["demand"].quantile(0.95)).mean() * 100
    return {
        "path": str(path),
        "purpose": "Understand target scale, skewness, and whether transformation or robust metrics are needed.",
        "observations": (
            f"Demand is strictly positive and right-skewed. Median={train['demand'].median():.4f}, "
            f"mean={train['demand'].mean():.4f}, 95th percentile={train['demand'].quantile(0.95):.4f}; "
            f"the top tail above p95 contains {high_tail_pct:.1f}% of records by definition but contributes disproportionate error risk."
        ),
        "business_implications": (
            "Most road-time observations have low demand, while a smaller set of hotspots or peak intervals may dominate service-level risk. "
            "Optimization should avoid a model that only predicts the central tendency."
        ),
    }


def plot_time_trend(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, str]:
    trend = (
        train.groupby(["day", "time_minutes"], as_index=False)
        .agg(mean_demand=("demand", "mean"), median_demand=("demand", "median"), record_count=("demand", "size"))
        .assign(time_label=lambda df: df["time_minutes"].map(lambda x: f"{x // 60:02d}:{x % 60:02d}"))
    )

    plt.figure(figsize=(15, 5))
    sns.lineplot(data=trend, x="time_minutes", y="mean_demand", hue="day", marker="o", linewidth=1.8)
    plt.title("Mean Demand By 15-Minute Slot")
    plt.xlabel("Minute of day")
    plt.ylabel("Mean demand")
    plt.xticks(np.arange(0, 24 * 60 + 1, 120))
    path = save_current_figure("02_time_trend.png")

    train_min = train.groupby("day")["time_minutes"].min().to_dict()
    train_max = train.groupby("day")["time_minutes"].max().to_dict()
    test_min = test.groupby("day")["time_minutes"].min().to_dict()
    test_max = test.groupby("day")["time_minutes"].max().to_dict()

    return {
        "path": str(path),
        "purpose": "Inspect intraday seasonality and confirm the forecasting timeline.",
        "observations": (
            f"Train time coverage by day has min slots {train_min} and max slots {train_max}. "
            f"Test coverage has min slots {test_min} and max slots {test_max}. "
            "The test set continues day 49 after the training cutoff."
        ),
        "business_implications": (
            "Validation must mimic forecasting future time slots. Random validation would overstate performance because nearby time and geohash patterns leak across folds."
        ),
    }


def plot_hour_weather_heatmap(train: pd.DataFrame) -> dict[str, str]:
    pivot = train.pivot_table(index="Weather", columns="hour", values="demand", aggfunc="mean")
    plt.figure(figsize=(15, 4.8))
    sns.heatmap(pivot, cmap="viridis", linewidths=0.25, linecolor="white")
    plt.title("Mean Demand By Weather And Hour")
    plt.xlabel("Hour")
    plt.ylabel("Weather")
    path = save_current_figure("03_weather_hour_heatmap.png")

    best_pair = (
        train.groupby(["Weather", "hour"])["demand"]
        .mean()
        .sort_values(ascending=False)
        .head(1)
        .reset_index()
        .iloc[0]
    )
    return {
        "path": str(path),
        "purpose": "Find interaction between weather conditions and time of day.",
        "observations": (
            f"The highest observed Weather x hour average is Weather={best_pair['Weather']} at hour={int(best_pair['hour'])}, "
            f"mean demand={best_pair['demand']:.4f}."
        ),
        "business_implications": (
            "Dispatch or traffic-management plans can benefit from condition-specific time windows instead of one generic daily schedule."
        ),
    }


def plot_categorical_target(train: pd.DataFrame) -> dict[str, str]:
    features = ["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks", "Weather"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        grouped = (
            train.groupby(feature, dropna=False)["demand"]
            .agg(["mean", "median", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )
        sns.barplot(data=grouped, x=feature, y="mean", ax=axes[idx], color="#4c78a8")
        axes[idx].set_title(f"Mean Demand By {feature}")
        axes[idx].set_xlabel(feature)
        axes[idx].set_ylabel("Mean demand")
        axes[idx].tick_params(axis="x", rotation=25)

    axes[-1].axis("off")
    path = save_current_figure("04_categorical_target_relationships.png")

    top_lines = []
    for feature in features:
        grouped = train.groupby(feature, dropna=False)["demand"].mean().sort_values(ascending=False)
        top_lines.append(f"{feature}: top={grouped.index[0]} ({grouped.iloc[0]:.4f})")

    return {
        "path": str(path),
        "purpose": "Compare average demand across operational categorical features.",
        "observations": "Highest mean-demand categories: " + "; ".join(top_lines) + ".",
        "business_implications": (
            "These categories can inform infrastructure-specific baselines and help explain demand differences to operations teams."
        ),
    }


def plot_temperature_relationship(train: pd.DataFrame) -> dict[str, str]:
    valid = train.dropna(subset=["Temperature", "demand"]).copy()
    valid["temperature_bin"] = pd.qcut(valid["Temperature"], q=20, duplicates="drop")
    binned = (
        valid.groupby("temperature_bin", observed=True)
        .agg(
            temperature_mid=("Temperature", "mean"),
            mean_demand=("demand", "mean"),
            record_count=("demand", "size"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(valid["Temperature"], bins=50, kde=True, ax=axes[0], color="#6c757d")
    axes[0].set_title("Temperature Distribution")
    axes[0].set_xlabel("Temperature")
    axes[0].set_ylabel("Record count")

    sns.lineplot(data=binned, x="temperature_mid", y="mean_demand", marker="o", ax=axes[1], color="#b55d60")
    axes[1].set_title("Mean Demand By Temperature Bin")
    axes[1].set_xlabel("Temperature bin midpoint")
    axes[1].set_ylabel("Mean demand")
    path = save_current_figure("05_temperature_relationship.png")

    corr = valid[["Temperature", "demand"]].corr().iloc[0, 1]
    return {
        "path": str(path),
        "purpose": "Assess whether temperature is predictive and whether the relationship is nonlinear.",
        "observations": (
            f"Temperature ranges from {valid['Temperature'].min():.2f} to {valid['Temperature'].max():.2f}; "
            f"Pearson correlation with demand is {corr:.4f}. The binned line should be used to judge nonlinear effects."
        ),
        "business_implications": (
            "Temperature may act as a contextual modifier rather than a dominant driver. Tree-based models can capture any nonlinear relationship without forcing a straight-line assumption."
        ),
    }


def plot_correlation(train: pd.DataFrame) -> dict[str, str]:
    numeric_columns = [
        "demand",
        "day",
        "time_minutes",
        "hour",
        "minute",
        "slot_15m",
        "NumberofLanes",
        "Temperature",
        "latitude",
        "longitude",
    ]
    corr = train[numeric_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", linewidths=0.25)
    plt.title("Numeric Feature Correlation Matrix")
    path = save_current_figure("06_correlation_matrix.png")

    target_corr = corr["demand"].drop("demand").abs().sort_values(ascending=False)
    return {
        "path": str(path),
        "purpose": "Identify linear relationships, multicollinearity, and simple target associations.",
        "observations": (
            "Largest absolute numeric correlations with demand: "
            + ", ".join([f"{idx}={value:.3f}" for idx, value in target_corr.head(5).items()])
            + "."
        ),
        "business_implications": (
            "Correlation is useful for sanity checks, but low linear correlation does not rule out operational value because traffic patterns are often nonlinear and interaction-heavy."
        ),
    }


def plot_geographical_hotspots(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, str]:
    geo_summary = (
        train.groupby(["geohash", "latitude", "longitude"], as_index=False)
        .agg(mean_demand=("demand", "mean"), median_demand=("demand", "median"), record_count=("demand", "size"))
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    scatter = axes[0].scatter(
        geo_summary["longitude"],
        geo_summary["latitude"],
        c=geo_summary["mean_demand"],
        s=np.clip(geo_summary["record_count"], 10, 80),
        cmap="plasma",
        alpha=0.8,
    )
    axes[0].set_title("Geohash Demand Hotspots")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    fig.colorbar(scatter, ax=axes[0], label="Mean demand")

    train_geo = set(train["geohash"].unique())
    test_geo = set(test["geohash"].unique())
    coverage = pd.DataFrame(
        {
            "coverage": ["Shared", "Train only", "Test only"],
            "count": [
                len(train_geo & test_geo),
                len(train_geo - test_geo),
                len(test_geo - train_geo),
            ],
        }
    )
    sns.barplot(
        data=coverage,
        x="coverage",
        y="count",
        hue="coverage",
        ax=axes[1],
        palette=["#4c78a8", "#f58518", "#54a24b"],
        legend=False,
    )
    axes[1].set_title("Train/Test Geohash Coverage")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Unique geohashes")

    path = save_current_figure("07_geographical_hotspots.png")
    top_hotspots = geo_summary.sort_values("mean_demand", ascending=False).head(5)
    return {
        "path": str(path),
        "purpose": "Visualize spatial demand concentration and train/test geographic coverage.",
        "observations": (
            f"Train has {len(train_geo)} unique geohashes; test has {len(test_geo)}; "
            f"{len(test_geo - train_geo)} test geohashes are unseen in training. "
            "Top mean-demand geohashes: "
            + ", ".join(top_hotspots["geohash"].tolist())
            + "."
        ),
        "business_implications": (
            "Location is likely one of the strongest demand drivers. The model needs fallback spatial features such as geohash prefixes and decoded coordinates for unseen geohashes."
        ),
    }


def plot_feature_target_interactions(train: pd.DataFrame) -> dict[str, str]:
    pivot = train.pivot_table(index="RoadType", columns="Weather", values="demand", aggfunc="mean")
    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako", linewidths=0.25)
    plt.title("Mean Demand By Road Type And Weather")
    plt.xlabel("Weather")
    plt.ylabel("Road type")
    path = save_current_figure("08_road_weather_interaction.png")

    stacked = pivot.stack().sort_values(ascending=False)
    top_index = stacked.index[0]
    return {
        "path": str(path),
        "purpose": "Check whether road infrastructure and weather jointly shift expected demand.",
        "observations": (
            f"The highest RoadType x Weather mean demand is RoadType={top_index[0]}, Weather={top_index[1]}, "
            f"mean demand={stacked.iloc[0]:.4f}."
        ),
        "business_implications": (
            "Interaction features can improve forecasts and also support targeted traffic-control policies under adverse weather."
        ),
    }


def write_assessment_tables(assessment: dict[str, pd.DataFrame]) -> None:
    tables_dir = REPORT_DIR / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, table in assessment.items():
        table.to_csv(tables_dir / f"{name}.csv", index=True)


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str:
    shown = df.head(max_rows)
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


def write_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
    assessment: dict[str, pd.DataFrame],
    visuals: list[dict[str, str]],
) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Traffic Demand Forecasting: EDA Report",
        "",
        "## Executive Summary",
        "",
        f"- Raw train shape: `{train.attrs.get('raw_shape', train.shape)}`; after EDA feature enrichment: `{train.shape}`.",
        f"- Raw test shape: `{test.attrs.get('raw_shape', test.shape)}`; after EDA feature enrichment: `{test.shape}`.",
        f"- Sample submission shape: `{sample_submission.shape}`.",
        f"- Target mean: `{train['demand'].mean():.6f}`; median: `{train['demand'].median():.6f}`; p95: `{train['demand'].quantile(0.95):.6f}`; max: `{train['demand'].max():.6f}`.",
        "- The target is right-skewed and capped at 1.0, so high-demand errors deserve explicit attention.",
        "- The test set is a chronological continuation of day 49, so validation should be time-aware rather than random.",
        "- Geohash is high-cardinality and has unseen test values, so exact geohash memorization is risky.",
        "",
        "## Dataset Assessment",
        "",
        "### File Shapes",
        "",
        dataframe_to_markdown(assessment["file_shapes"]),
        "",
        "### Train Missingness, Dtypes, And Cardinality",
        "",
        dataframe_to_markdown(assessment["train_missing"]),
        "",
        "### Test Missingness, Dtypes, And Cardinality",
        "",
        dataframe_to_markdown(assessment["test_missing"]),
        "",
        "### Train/Test Category Overlap",
        "",
        dataframe_to_markdown(assessment["category_overlap"]),
        "",
        "### Duplicate Checks",
        "",
        "**Train**",
        "",
        dataframe_to_markdown(assessment["train_duplicate_keys"]),
        "",
        "**Test**",
        "",
        dataframe_to_markdown(assessment["test_duplicate_keys"]),
        "",
        "### IQR Outlier Summary",
        "",
        dataframe_to_markdown(assessment["outliers"]),
        "",
        "## Visualization Walkthrough",
        "",
    ]

    for idx, visual in enumerate(visuals, start=1):
        lines.extend(
            [
                f"### {idx}. {Path(visual['path']).name}",
                "",
                f"![{Path(visual['path']).stem}]({Path(visual['path']).relative_to(REPORT_DIR).as_posix()})",
                "",
                f"**Purpose:** {visual['purpose']}",
                "",
                f"**Key observations:** {visual['observations']}",
                "",
                f"**Business implications:** {visual['business_implications']}",
                "",
            ]
        )

    lines.extend(
        [
            "## Recommended Next Steps",
            "",
            "1. Build leakage-safe feature engineering using timestamp decomposition, cyclical hour encodings, geohash prefixes, decoded coordinates, and train-fold-only aggregate encodings.",
            "2. Use chronological validation that forecasts later slots from earlier slots.",
            "3. Benchmark Linear Regression, Random Forest, XGBoost, LightGBM, and CatBoost using R2, MAE, RMSE, and cross-validation.",
            "4. Tune the strongest gradient boosting models and perform residual analysis by hour, geohash, weather, and demand percentile.",
        ]
    )

    report_path = REPORT_DIR / "eda_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    train, test, sample_submission = load_data()
    assessment = make_dataset_assessment(train, test, sample_submission)
    write_assessment_tables(assessment)

    visuals = [
        plot_target_distribution(train),
        plot_time_trend(train, test),
        plot_hour_weather_heatmap(train),
        plot_categorical_target(train),
        plot_temperature_relationship(train),
        plot_correlation(train),
        plot_geographical_hotspots(train, test),
        plot_feature_target_interactions(train),
    ]

    report_path = write_report(train, test, sample_submission, assessment, visuals)
    print(f"Wrote EDA report to {report_path}")
    print(f"Wrote figures to {FIGURE_DIR}")
    print(f"Wrote assessment tables to {REPORT_DIR / 'tables'}")


if __name__ == "__main__":
    main()
