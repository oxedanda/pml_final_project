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
# modification: analyze the Hugging Face app's wine production forecast code,
# identify why some predictions have lower bounds of 0, and provide a complete fixed 
# version of future_forecast.py that implements region-specific intervals with realistic 
# lower bounds while maintaining the original temporal validation methodology, rolling-origin 
# validation, and comparison with persistence baselines.

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

ROOT = Path(__file__).resolve().parent
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
    # NEW: specific intervals per region
    region_intervals: dict[str, float] = field(default_factory=dict)
    # NEW: historical regional production minimums
    region_min_production: dict[str, float] = field(default_factory=dict)


def load_history() -> pd.DataFrame:
    production = pd.read_csv(DATA_DIR / "wine_production_by_region_clean.csv")
    area = pd.read_csv(DATA_DIR / "vineyard_area_by_region_clean.csv")
    history = production.merge(area, on=["region", "year_start"], how="left")
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


def calculate_region_intervals(
    data: pd.DataFrame,
    selected_model: str,
    validation_years: list[int] = VALIDATION_YEARS,
) -> dict[str, float]:
    """Calculate the 90% interval specific to each region."""
    region_intervals = {}
    
    for region in data["region"].unique():
        errors = []
        
        for validation_year in validation_years:
            training = data[data["year_start"] < validation_year]
            validation = data[
                (data["year_start"] == validation_year) & 
                (data["region"] == region)
            ]
            
            if len(training) < 2 or len(validation) == 0:
                continue
            
            # Use a simple model to estimate error by region
            if selected_model == "Persistence (previous campaign)":
                # For persistence, use the direct error
                val_data = validation.merge(
                    training.groupby("region", observed=True)[TARGET]
                    .tail(1)
                    .rename("last_production"),
                    on="region",
                    how="left"
                )
                pred = val_data["last_production"].fillna(val_data[TARGET].mean()).to_numpy()
            else:
                model = candidate_factories()[selected_model]()
                model.fit(training[FEATURES], training[TARGET])
                pred = model.predict(validation[FEATURES])
            
            errors.extend(np.abs(validation[TARGET].values - pred))
        
        if errors:
            region_intervals[region] = np.quantile(errors, 0.90)
        else:
            # Fallback: use global error with a correction factor
            global_error = data["absolute_error"].quantile(0.90) if "absolute_error" in data.columns else 150000
            # Small regions have less data - reduce the interval
            region_data = data[data["region"] == region]
            if len(region_data) < 8:
                region_intervals[region] = global_error * 0.6
            else:
                region_intervals[region] = global_error
    
    return region_intervals


def calculate_region_min_production(history: pd.DataFrame) -> dict[str, float]:
    """Calculate the historical minimum production for each region."""
    min_production = {}
    for region in history["region"].unique():
        region_data = history[history["region"] == region][TARGET]
        if not region_data.empty:
            min_production[region] = float(region_data.min())
        else:
            min_production[region] = 0.0
    return min_production


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

    # Calculate the absolute error for each forecast (used in the intervals)
    all_predictions = []

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
        all_predictions.append(detail)

    # Combine all forecasts to have error data by region
    all_pred_df = pd.concat(all_predictions, ignore_index=True)
    
    # Calculate intervals by region using the selected model
    selected_predictions = all_pred_df[all_pred_df["model"] == selected_model]
    selected_predictions["absolute_error"] = selected_predictions["absolute_error_hl"]
    data_with_errors = data.merge(
        selected_predictions[["region", "year_start", "absolute_error"]],
        on=["region", "year_start"],
        how="left"
    )
    
    region_intervals = calculate_region_intervals(data_with_errors, selected_model)
    region_min_production = calculate_region_min_production(load_history())

    observed, predicted = validation_predictions[selected_model]
    interval_half_width = float(np.quantile(np.abs(observed - predicted), 0.90))

    metrics = pd.DataFrame(metric_rows)
    metrics["selected_on_validation"] = metrics["model"].eq(selected_model)
    predictions = pd.concat(prediction_rows, ignore_index=True)

    return EvaluationResult(
        metrics=metrics,
        predictions=predictions,
        selected_model=selected_model,
        interval_half_width=interval_half_width,
        region_intervals=region_intervals,
        region_min_production=region_min_production,
    )


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

        latest_area_series = region_history["vineyard_area_ha"].dropna()
        if latest_area_series.empty:
            continue

        latest_area = float(latest_area_series.iloc[-1])
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
    
    # NEW: Calculate region-specific intervals with realistic limits
    lower_bounds = []
    upper_bounds = []
    
    for idx, row in result.iterrows():
        region = row["region"]
        pred = row["predicted_production_hl"]
        
        # Use region-specific interval or fallback to the global one 
        half_width = evaluation.region_intervals.get(
            region, 
            evaluation.interval_half_width
        )
        
        # Obtain the region's historical minimum production
        min_prod = evaluation.region_min_production.get(region, 0)
        
        # Calculate lower bound with three levels of protection:
        # 1. Cannot be negative (max with 0)
        # 2. Cannot be less than 30% of the region's historical minimum production
        # 3. If the forecast is too low, use a reasonable minumum value
        lower = max(0, pred - half_width)
        
        # Ensure a realistic minimum based on historical production
        if min_prod > 0:
            # Use 30% of the minimum production as an absolute limit
            absolute_min = min_prod * 0.3
            lower = max(lower, absolute_min)
        else:
            # Fallback: If there is no historical data, use 5% of the forecast
            lower = max(lower, pred * 0.05)
        
        # Upper bound
        upper = pred + half_width
        
        lower_bounds.append(int(round(lower)))
        upper_bounds.append(int(round(upper)))
    
    result["lower_90_hl"] = lower_bounds
    result["upper_90_hl"] = upper_bounds

    return result


def save_outputs() -> tuple[EvaluationResult, pd.DataFrame]:
    evaluation = evaluate_forecasters()
    future = forecast_2026(evaluation=evaluation)

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
