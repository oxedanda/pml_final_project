"""Experimental one-year-ahead wine-production forecasts for 2026/27.

The forecasting features are built exclusively from observations preceding the
target campaign. Model selection uses rolling-origin validation over 2018-2022;
2023-2025 remains a final one-step-ahead test period.
"""

# prompt: Create a genuine future wine-production simulator without removing
# the existing notebook. Use lagged production, rolling temporal validation,
# explicit persistence comparisons, a 2026/27 forecast, and honest uncertainty.
# Modifications: feature definitions, candidate models, split years, file paths,
# and the empirical interval were adapted to the IVV regional panel.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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

TARGET = "total_production_hl"
FEATURES = [
    "region",
    "year_start",
    "vineyard_area_ha",
    "production_lag_1",
    "production_lag_2",
    "production_mean_3",
    "production_trend_1",
]
VALIDATION_YEARS = [2018, 2019, 2020, 2021, 2022]
TEST_YEARS = [2023, 2024, 2025]
FUTURE_YEAR = 2026
RANDOM_STATE = 42


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_model: str
    interval_half_width: float


def load_history() -> pd.DataFrame:
    production = pd.read_csv(DATA_DIR / "wine_production_by_region_clean.csv")
    area = pd.read_csv(DATA_DIR / "vineyard_area_by_region_clean.csv")
    history = production.merge(area, on=["region", "year_start"], how="left")
    return history.sort_values(["region", "year_start"]).reset_index(drop=True)


def build_supervised_data(history: pd.DataFrame) -> pd.DataFrame:
    """Create features using only values earlier than each target year."""
    data = history.sort_values(["region", "year_start"]).copy()
    grouped_target = data.groupby("region", observed=True)[TARGET]
    data["production_lag_1"] = grouped_target.shift(1)
    data["production_lag_2"] = grouped_target.shift(2)
    data["production_mean_3"] = grouped_target.transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).mean()
    )
    data["production_trend_1"] = (
        data["production_lag_1"] - data["production_lag_2"]
    )
    return data.dropna(subset=["production_lag_1", "production_lag_2"]).reset_index(drop=True)


def make_preprocessor() -> ColumnTransformer:
    numeric_features = [feature for feature in FEATURES if feature != "region"]
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("region", OneHotEncoder(handle_unknown="ignore"), ["region"]),
        ]
    )


def make_pipeline(regressor: object) -> Pipeline:
    return Pipeline([("preprocess", make_preprocessor()), ("regressor", regressor)])


def candidate_factories() -> dict[str, Callable[[], Pipeline]]:
    return {
        "Linear regression with lags": lambda: make_pipeline(LinearRegression()),
        "Ridge with lags (alpha=0.1)": lambda: make_pipeline(Ridge(alpha=0.1)),
        "Ridge with lags (alpha=1)": lambda: make_pipeline(Ridge(alpha=1.0)),
        "Ridge with lags (alpha=10)": lambda: make_pipeline(Ridge(alpha=10.0)),
        "Random forest with lags": lambda: make_pipeline(
            RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                max_features=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "R2": r2_score(y_true, y_pred),
    }


def predict_named_model(
    name: str, training: pd.DataFrame, evaluation: pd.DataFrame
) -> np.ndarray:
    if name == "Persistence (previous campaign)":
        return evaluation["production_lag_1"].to_numpy()
    model = candidate_factories()[name]()
    model.fit(training[FEATURES], training[TARGET])
    return model.predict(evaluation[FEATURES])


def evaluate_forecasters() -> EvaluationResult:
    data = build_supervised_data(load_history())
    names = ["Persistence (previous campaign)", *candidate_factories().keys()]
    metric_rows: list[dict[str, float | str | bool]] = []
    prediction_rows: list[pd.DataFrame] = []
    validation_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name in names:
        observed_parts = []
        predicted_parts = []
        for validation_year in VALIDATION_YEARS:
            training = data[data["year_start"] < validation_year]
            validation = data[data["year_start"] == validation_year]
            prediction = predict_named_model(name, training, validation)
            observed_parts.append(validation[TARGET].to_numpy())
            predicted_parts.append(prediction)
        observed = np.concatenate(observed_parts)
        predicted = np.concatenate(predicted_parts)
        validation_predictions[name] = (observed, predicted)
        metric_rows.append(
            {
                "model": name,
                "split": "rolling_validation_2018_2022",
                **regression_metrics(observed, predicted),
            }
        )

    validation_metrics = pd.DataFrame(metric_rows).sort_values("MAE")
    selected_model = str(validation_metrics.iloc[0]["model"])

    final_training = data[data["year_start"] < min(TEST_YEARS)]
    test = data[data["year_start"].isin(TEST_YEARS)]
    for name in names:
        prediction = predict_named_model(name, final_training, test)
        metric_rows.append(
            {
                "model": name,
                "split": "test_2023_2025",
                **regression_metrics(test[TARGET].to_numpy(), prediction),
            }
        )
        detail = test[["region", "year_start", TARGET]].copy()
        detail["model"] = name
        detail["predicted_production_hl"] = prediction
        detail["absolute_error_hl"] = np.abs(detail[TARGET] - prediction)
        prediction_rows.append(detail)

    observed, predicted = validation_predictions[selected_model]
    interval_half_width = float(np.quantile(np.abs(observed - predicted), 0.90))
    metrics = pd.DataFrame(metric_rows)
    metrics["selected_on_validation"] = metrics["model"].eq(selected_model)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return EvaluationResult(metrics, predictions, selected_model, interval_half_width)


def make_future_features(
    history: pd.DataFrame,
    future_year: int = FUTURE_YEAR,
    area_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows = []
    overrides = area_overrides or {}
    for region, region_history in history.groupby("region", observed=True):
        region_history = region_history.sort_values("year_start")
        recent = region_history.tail(3)
        if len(recent) < 2:
            continue
        latest_area = region_history["vineyard_area_ha"].dropna().iloc[-1]
        lag_1 = float(recent[TARGET].iloc[-1])
        lag_2 = float(recent[TARGET].iloc[-2])
        rows.append(
            {
                "region": region,
                "year_start": future_year,
                "vineyard_area_ha": float(overrides.get(region, latest_area)),
                "production_lag_1": lag_1,
                "production_lag_2": lag_2,
                "production_mean_3": float(recent[TARGET].mean()),
                "production_trend_1": lag_1 - lag_2,
            }
        )
    return pd.DataFrame(rows).sort_values("region").reset_index(drop=True)


def forecast_2026(
    evaluation: EvaluationResult | None = None,
    area_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    evaluation = evaluation or evaluate_forecasters()
    history = load_history()
    supervised = build_supervised_data(history)
    future = make_future_features(history, area_overrides=area_overrides)

    if evaluation.selected_model == "Persistence (previous campaign)":
        prediction = future["production_lag_1"].to_numpy()
    else:
        model = candidate_factories()[evaluation.selected_model]()
        model.fit(supervised[FEATURES], supervised[TARGET])
        prediction = model.predict(future[FEATURES])

    result = future[["region", "year_start", "vineyard_area_ha"]].copy()
    result["selected_model"] = evaluation.selected_model
    result["predicted_production_hl"] = np.maximum(0, prediction).round().astype(int)
    result["lower_90_hl"] = np.maximum(
        0, prediction - evaluation.interval_half_width
    ).round().astype(int)
    result["upper_90_hl"] = (
        prediction + evaluation.interval_half_width
    ).round().astype(int)
    return result


def save_outputs() -> tuple[EvaluationResult, pd.DataFrame]:
    evaluation = evaluate_forecasters()
    future = forecast_2026(evaluation)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    evaluation.metrics.to_csv(TABLES_DIR / "future_model_comparison.csv", index=False)
    evaluation.predictions.to_csv(TABLES_DIR / "future_model_test_predictions.csv", index=False)
    future.to_csv(TABLES_DIR / "forecast_2026_27.csv", index=False)
    return evaluation, future


def main() -> None:
    evaluation, future = save_outputs()
    print("Selected on rolling validation:", evaluation.selected_model)
    print("Empirical 90% interval half-width:", f"{evaluation.interval_half_width:,.0f} hl")
    print("\nMetrics")
    print(evaluation.metrics.sort_values(["split", "MAE"]).to_string(index=False))
    print("\n2026/27 experimental forecasts")
    print(future.to_string(index=False))


if __name__ == "__main__":
    main()
