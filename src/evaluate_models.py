"""Reproducible temporal evaluation for regional wine-production models.

Run from the repository root with:
    python src/evaluate_models.py

The script writes model metrics, predictions, and a comparison figure under
``outputs/``. Model choices are made on the validation year only; the final
test years remain untouched until the last evaluation.
"""

# prompt: Review the PML project and implement a leakage-resistant temporal
# model comparison with transparent baselines, a validation set, reproducible
# outputs, and concise comments. The generated implementation is below.
# Modifications: paths, split years, metrics, candidate models, and output names
# were adapted to this repository and its IVV datasets.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"

FEATURES = ["region", "year_start", "vineyard_area_ha"]
TARGET = "total_production_hl"
VALIDATION_YEARS = [2018, 2019, 2020, 2021, 2022]
TEST_YEARS = [2023, 2024, 2025]
RANDOM_STATE = 42


@dataclass(frozen=True)
class Candidate:
    name: str
    estimator: Pipeline


def load_data() -> pd.DataFrame:
    production = pd.read_csv(DATA_DIR / "wine_production_by_region_clean.csv")
    area = pd.read_csv(DATA_DIR / "vineyard_area_by_region_clean.csv")
    data = production.merge(area, on=["region", "year_start"], how="left")
    return data.sort_values(["year_start", "region"]).reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, ["year_start", "vineyard_area_ha"]),
            ("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
        ]
    )


def candidate_models() -> list[Candidate]:
    candidates: list[Candidate] = []
    specifications = [
        ("Linear regression", LinearRegression()),
        ("Ridge (alpha=0.1)", Ridge(alpha=0.1)),
        ("Ridge (alpha=1)", Ridge(alpha=1.0)),
        ("Ridge (alpha=10)", Ridge(alpha=10.0)),
        ("Ridge (alpha=100)", Ridge(alpha=100.0)),
    ]
    for name, estimator in specifications:
        candidates.append(
            Candidate(
                name,
                Pipeline(
                    [("preprocess", make_preprocessor()), ("regressor", estimator)]
                ),
            )
        )

    for leaf_size in (1, 2, 4):
        candidates.append(
            Candidate(
                f"Random forest (min_leaf={leaf_size})",
                Pipeline(
                    [
                        ("preprocess", make_preprocessor()),
                        (
                            "regressor",
                            RandomForestRegressor(
                                n_estimators=500,
                                min_samples_leaf=leaf_size,
                                max_features=0.8,
                                random_state=RANDOM_STATE,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                ),
            )
        )
    return candidates


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def persistence_predictions(train: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    latest = (
        train.sort_values("year_start")
        .groupby("region", observed=True)
        .tail(1)
        .set_index("region")[TARGET]
    )
    fallback = float(train[TARGET].mean())
    return evaluation["region"].map(latest).fillna(fallback).to_numpy()


def regional_mean_predictions(train: pd.DataFrame, evaluation: pd.DataFrame) -> np.ndarray:
    regional_means = train.groupby("region", observed=True)[TARGET].mean()
    fallback = float(train[TARGET].mean())
    return evaluation["region"].map(regional_means).fillna(fallback).to_numpy()


def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = load_data()
    final_training = data[data["year_start"] < min(TEST_YEARS)].copy()
    test = data[data["year_start"].isin(TEST_YEARS)].copy()

    if test.empty:
        raise ValueError("The configured test period has no observations.")

    validation_rows: list[dict[str, float | str]] = []
    for candidate in candidate_models():
        observed_parts = []
        prediction_parts = []
        for validation_year in VALIDATION_YEARS:
            development = data[data["year_start"] < validation_year]
            validation = data[data["year_start"] == validation_year]
            if development.empty or validation.empty:
                continue
            candidate.estimator.fit(development[FEATURES], development[TARGET])
            observed_parts.append(validation[TARGET].to_numpy())
            prediction_parts.append(candidate.estimator.predict(validation[FEATURES]))
        observed = np.concatenate(observed_parts)
        prediction = np.concatenate(prediction_parts)
        validation_rows.append(
            {
                "model": candidate.name,
                "split": "rolling_validation_2018_2022",
                **metrics(pd.Series(observed), prediction),
            }
        )

    for baseline_name, baseline_function in [
        ("Persistence (last region value)", persistence_predictions),
        ("Historical region mean", regional_mean_predictions),
    ]:
        observed_parts = []
        prediction_parts = []
        for validation_year in VALIDATION_YEARS:
            development = data[data["year_start"] < validation_year]
            validation = data[data["year_start"] == validation_year]
            if development.empty or validation.empty:
                continue
            observed_parts.append(validation[TARGET].to_numpy())
            prediction_parts.append(baseline_function(development, validation))
        validation_rows.append(
            {
                "model": baseline_name,
                "split": "rolling_validation_2018_2022",
                **metrics(pd.Series(np.concatenate(observed_parts)), np.concatenate(prediction_parts)),
            }
        )

    validation_metrics = pd.DataFrame(validation_rows).sort_values("MAE")
    selected_name = str(validation_metrics.iloc[0]["model"])

    test_predictions = {
        "Persistence (last region value)": persistence_predictions(final_training, test),
        "Historical region mean": regional_mean_predictions(final_training, test),
    }
    for candidate in candidate_models():
        candidate.estimator.fit(final_training[FEATURES], final_training[TARGET])
        test_predictions[candidate.name] = candidate.estimator.predict(test[FEATURES])

    test_rows = []
    prediction_rows = []
    for name, prediction in test_predictions.items():
        test_rows.append(
            {
                "model": name,
                "split": "test_2023_2025",
                **metrics(test[TARGET], prediction),
            }
        )
        model_predictions = test[["region", "year_start", TARGET]].copy()
        model_predictions["model"] = name
        model_predictions["predicted_production_hl"] = prediction
        model_predictions["absolute_error_hl"] = np.abs(
            model_predictions[TARGET] - model_predictions["predicted_production_hl"]
        )
        prediction_rows.append(model_predictions)

    all_metrics = pd.concat(
        [validation_metrics, pd.DataFrame(test_rows).sort_values("MAE")],
        ignore_index=True,
    )
    all_metrics["selected_on_validation"] = all_metrics["model"].eq(selected_name)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return all_metrics, predictions


def save_outputs(metric_table: pd.DataFrame, predictions: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metric_table.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    predictions.to_csv(TABLES_DIR / "test_predictions.csv", index=False)

    test_metrics = metric_table[metric_table["split"] == "test_2023_2025"].sort_values("MAE")
    fig, axis = plt.subplots(figsize=(9, 4.8))
    axis.barh(test_metrics["model"], test_metrics["MAE"], color="#7a0019")
    axis.invert_yaxis()
    axis.set_xlabel("Mean absolute error (hectolitres)")
    axis.set_title("Temporal test performance: 2023-2025")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison_mae.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metric_table, predictions = evaluate()
    save_outputs(metric_table, predictions)
    print(metric_table.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))


if __name__ == "__main__":
    main()
