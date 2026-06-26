"""Future wine-production forecasts with regional uncertainty intervals.

The model predicts one campaign ahead using only information available before
the target year. Forecast intervals are calibrated from rolling-origin errors
and use a historical lower floor per region so small regions do not show a
misleading zero lower bound when their historical production has been positive.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
LOWER_FLOOR_SHARE_OF_HISTORICAL_MIN = 0.30


def find_project_root() -> Path:
    """Return the project root whether this file is run from src/ or app root."""
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "data" / "processed").exists():
            return candidate
    return here.parent


ROOT = find_project_root()
DATA_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "outputs" / "tables"


@dataclass
class EvaluationResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    selected_model: str
    interval_half_width: float
    region_intervals: dict[str, float] = field(default_factory=dict)
    region_lower_floors: dict[str, float] = field(default_factory=dict)


def load_history() -> pd.DataFrame:
    production_path = DATA_DIR / "wine_production_by_region_clean.csv"
    area_path = DATA_DIR / "vineyard_area_by_region_clean.csv"

    if not production_path.exists():
        raise FileNotFoundError(f"Missing processed production file: {production_path}")
    if not area_path.exists():
        raise FileNotFoundError(f"Missing processed vineyard-area file: {area_path}")

    production = pd.read_csv(production_path)
    area = pd.read_csv(area_path)
    history = production.merge(area, on=["region", "year_start"], how="left")

    numeric_columns = [TARGET, "year_start", "vineyard_area_ha"]
    for column in numeric_columns:
        history[column] = pd.to_numeric(history[column], errors="coerce")

    history = history.dropna(subset=["region", "year_start", TARGET])
    history["year_start"] = history["year_start"].astype(int)
    return history.sort_values(["region", "year_start"]).reset_index(drop=True)


def build_supervised_data(history: pd.DataFrame) -> pd.DataFrame:
    data = history.sort_values(["region", "year_start"]).copy()
    grouped_target = data.groupby("region", observed=True)[TARGET]

    data["production_lag_1"] = grouped_target.shift(1)
    data["production_lag_2"] = grouped_target.shift(2)
    data["production_mean_3"] = grouped_target.transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).mean()
    )
    data["production_trend_1"] = data["production_lag_1"] - data["production_lag_2"]

    return data.dropna(subset=["production_lag_1", "production_lag_2"]).reset_index(
        drop=True
    )


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
    return Pipeline(
        [
            ("preprocess", make_preprocessor()),
            ("regressor", regressor),
        ]
    )


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
    name: str,
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> np.ndarray:
    if name == "Persistence (previous campaign)":
        return evaluation["production_lag_1"].to_numpy()

    model = candidate_factories()[name]()
    model.fit(training[FEATURES], training[TARGET])
    return model.predict(evaluation[FEATURES])


def rolling_predictions(
    data: pd.DataFrame,
    model_name: str,
    years: list[int],
    split_name: str,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    observed_parts: list[np.ndarray] = []
    predicted_parts: list[np.ndarray] = []
    detail_parts: list[pd.DataFrame] = []

    for year in years:
        training = data[data["year_start"] < year]
        evaluation = data[data["year_start"] == year]
        if training.empty or evaluation.empty:
            continue

        prediction = predict_named_model(model_name, training, evaluation)
        observed = evaluation[TARGET].to_numpy()

        observed_parts.append(observed)
        predicted_parts.append(prediction)

        detail = evaluation[["region", "year_start", TARGET]].copy()
        detail["model"] = model_name
        detail["split"] = split_name
        detail["predicted_production_hl"] = prediction
        detail["absolute_error_hl"] = np.abs(observed - prediction)
        detail_parts.append(detail)

    if not observed_parts:
        raise ValueError(f"No evaluation rows were available for {split_name}.")

    observed_all = np.concatenate(observed_parts)
    predicted_all = np.concatenate(predicted_parts)
    metric_row = {
        "model": model_name,
        "split": split_name,
        **regression_metrics(observed_all, predicted_all),
    }
    detail_all = pd.concat(detail_parts, ignore_index=True)
    return metric_row, detail_all


def calculate_region_intervals(
    validation_predictions: pd.DataFrame,
    global_half_width: float,
) -> dict[str, float]:
    intervals: dict[str, float] = {}
    for region, group in validation_predictions.groupby("region", observed=True):
        errors = group["absolute_error_hl"].dropna()
        if errors.empty:
            intervals[str(region)] = float(global_half_width)
        else:
            intervals[str(region)] = float(errors.quantile(0.90))
    return intervals


def calculate_region_lower_floors(history: pd.DataFrame) -> dict[str, float]:
    floors: dict[str, float] = {}
    for region, group in history.groupby("region", observed=True):
        positive = group.loc[group[TARGET] > 0, TARGET]
        if positive.empty:
            floors[str(region)] = 0.0
        else:
            floors[str(region)] = float(
                positive.min() * LOWER_FLOOR_SHARE_OF_HISTORICAL_MIN
            )
    return floors


def evaluate_forecasters() -> EvaluationResult:
    history = load_history()
    data = build_supervised_data(history)
    model_names = ["Persistence (previous campaign)", *candidate_factories().keys()]

    metric_rows: list[dict[str, float | str]] = []
    prediction_parts: list[pd.DataFrame] = []
    validation_prediction_by_model: dict[str, pd.DataFrame] = {}

    for model_name in model_names:
        metrics, predictions = rolling_predictions(
            data,
            model_name=model_name,
            years=VALIDATION_YEARS,
            split_name="rolling_validation_2018_2022",
        )
        metric_rows.append(metrics)
        prediction_parts.append(predictions)
        validation_prediction_by_model[model_name] = predictions

    validation_metrics = pd.DataFrame(metric_rows).sort_values("MAE")
    selected_model = str(validation_metrics.iloc[0]["model"])

    final_training = data[data["year_start"] < min(TEST_YEARS)]
    test = data[data["year_start"].isin(TEST_YEARS)]
    for model_name in model_names:
        prediction = predict_named_model(model_name, final_training, test)
        metrics = {
            "model": model_name,
            "split": "test_2023_2025",
            **regression_metrics(test[TARGET].to_numpy(), prediction),
        }
        metric_rows.append(metrics)

        detail = test[["region", "year_start", TARGET]].copy()
        detail["model"] = model_name
        detail["split"] = "test_2023_2025"
        detail["predicted_production_hl"] = prediction
        detail["absolute_error_hl"] = np.abs(test[TARGET].to_numpy() - prediction)
        prediction_parts.append(detail)

    selected_validation = validation_prediction_by_model[selected_model]
    interval_half_width = float(selected_validation["absolute_error_hl"].quantile(0.90))
    region_intervals = calculate_region_intervals(
        selected_validation,
        global_half_width=interval_half_width,
    )
    region_lower_floors = calculate_region_lower_floors(history)

    metrics = pd.DataFrame(metric_rows)
    metrics["selected_on_validation"] = metrics["model"].eq(selected_model)
    predictions = pd.concat(prediction_parts, ignore_index=True)

    return EvaluationResult(
        metrics=metrics,
        predictions=predictions,
        selected_model=selected_model,
        interval_half_width=interval_half_width,
        region_intervals=region_intervals,
        region_lower_floors=region_lower_floors,
    )


def make_future_features(
    history: pd.DataFrame,
    future_year: int = FUTURE_YEAR,
    area_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    overrides = area_overrides or {}

    for region, region_history in history.groupby("region", observed=True):
        region_history = region_history.sort_values("year_start")
        recent = region_history.tail(3)
        latest_area_series = region_history["vineyard_area_ha"].dropna()

        if len(recent) < 2 or latest_area_series.empty:
            continue

        latest_area = float(latest_area_series.iloc[-1])
        lag_1 = float(recent[TARGET].iloc[-1])
        lag_2 = float(recent[TARGET].iloc[-2])

        rows.append(
            {
                "region": str(region),
                "year_start": int(future_year),
                "vineyard_area_ha": float(overrides.get(str(region), latest_area)),
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

    lower_bounds: list[int] = []
    upper_bounds: list[int] = []
    interval_widths: list[int] = []
    lower_floors: list[int] = []

    for _, row in result.iterrows():
        region = str(row["region"])
        predicted = float(row["predicted_production_hl"])
        half_width = float(
            evaluation.region_intervals.get(region, evaluation.interval_half_width)
        )
        historical_floor = float(evaluation.region_lower_floors.get(region, 0.0))

        raw_lower = max(0.0, predicted - half_width)
        if historical_floor > 0:
            lower = max(raw_lower, historical_floor)
        elif predicted > 0:
            lower = max(raw_lower, predicted * 0.05)
        else:
            lower = 0.0

        if predicted > 0:
            lower = min(lower, predicted)

        upper = predicted + half_width
        lower_bounds.append(int(round(lower)))
        upper_bounds.append(int(round(upper)))
        interval_widths.append(int(round(half_width)))
        lower_floors.append(int(round(historical_floor)))

    result["lower_90_hl"] = lower_bounds
    result["upper_90_hl"] = upper_bounds
    result["region_interval_half_width_hl"] = interval_widths
    result["historical_lower_floor_hl"] = lower_floors
    return result


def save_outputs() -> tuple[EvaluationResult, pd.DataFrame]:
    evaluation = evaluate_forecasters()
    future = forecast_2026(evaluation=evaluation)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    evaluation.metrics.to_csv(TABLES_DIR / "future_model_comparison.csv", index=False)
    evaluation.predictions.to_csv(
        TABLES_DIR / "future_model_predictions.csv",
        index=False,
    )
    evaluation.predictions.query("split == 'test_2023_2025'").to_csv(
        TABLES_DIR / "future_model_test_predictions.csv",
        index=False,
    )
    future.to_csv(TABLES_DIR / "forecast_2026_27.csv", index=False)

    return evaluation, future


def main() -> None:
    evaluation, future = save_outputs()
    print("Selected on rolling validation:", evaluation.selected_model)
    print(
        "Global empirical 90% interval half-width:",
        f"{evaluation.interval_half_width:,.0f} hl",
    )
    print("\nMetrics")
    print(evaluation.metrics.sort_values(["split", "MAE"]).to_string(index=False))
    print("\n2026/27 experimental forecasts")
    print(future.to_string(index=False))


if __name__ == "__main__":
    main()
