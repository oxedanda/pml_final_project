"""Interactive Gradio app for the experimental wine-production forecast.

Run from the repository root with:
    python src/app.py

The app is intentionally a visual prototype. It uses the same forecasting logic
as ``src/future_forecast.py`` and does not make causal claims about vineyard
area changes.
"""

from __future__ import annotations

import pandas as pd
import gradio as gr

try:
    from future_forecast import evaluate_forecasters, forecast_2026, load_history
except ModuleNotFoundError:
    from src.future_forecast import evaluate_forecasters, forecast_2026, load_history


def number(value: float | int, decimals: int = 0) -> str:
    """Format numbers for cards and tables."""
    if pd.isna(value):
        return ""
    return f"{value:,.{decimals}f}"


def format_forecast_table(data: pd.DataFrame) -> pd.DataFrame:
    """Return a human-readable forecast table for display in Gradio."""
    view = data.copy()
    numeric_columns = [
        "vineyard_area_ha",
        "predicted_production_hl",
        "lower_90_hl",
        "upper_90_hl",
    ]
    for column in numeric_columns:
        if column in view.columns:
            view[column] = view[column].map(lambda value: number(value))

    return view.rename(
        columns={
            "region": "Region",
            "year_start": "Year",
            "vineyard_area_ha": "Area (ha)",
            "selected_model": "Selected model",
            "predicted_production_hl": "Predicted production (hl)",
            "lower_90_hl": "Lower empirical 90% (hl)",
            "upper_90_hl": "Upper empirical 90% (hl)",
        }
    )


def format_metrics_table(data: pd.DataFrame) -> pd.DataFrame:
    """Round and rename metric columns for a cleaner dashboard table."""
    view = data[["model", "MAE", "RMSE", "R2"]].copy()
    view["MAE"] = view["MAE"].map(lambda value: number(value))
    view["RMSE"] = view["RMSE"].map(lambda value: number(value))
    view["R2"] = view["R2"].map(lambda value: f"{value:.3f}")
    return view.rename(
        columns={
            "model": "Model",
            "MAE": "MAE (hl)",
            "RMSE": "RMSE (hl)",
            "R2": "R²",
        }
    )


history = load_history()
evaluation = evaluate_forecasters()
base_forecast = forecast_2026(evaluation=evaluation)

latest_area = (
    history.sort_values(["region", "year_start"])
    .groupby("region", as_index=False)
    .tail(1)[["region", "vineyard_area_ha"]]
)

region_choices = sorted(base_forecast["region"].astype(str).unique().tolist())
default_region = region_choices[0]
area_map = dict(zip(latest_area["region"].astype(str), latest_area["vineyard_area_ha"]))

validation_metrics = (
    evaluation.metrics.query("split == 'rolling_validation_2018_2022'")
    .sort_values("MAE")
    .reset_index(drop=True)
)
test_metrics = (
    evaluation.metrics.query("split == 'test_2023_2025'")
    .sort_values("MAE")
    .reset_index(drop=True)
)


def update_area(region: str):
    """Reset the area input to the latest observed area for the chosen region."""
    return gr.update(value=round(float(area_map.get(region, 0.0)), 2))


def simulate(region: str, area_ha: float):
    """Forecast one region under the selected vineyard-area scenario."""
    if not region:
        raise gr.Error("Please choose a region.")
    if area_ha is None or area_ha <= 0:
        raise gr.Error("Area (ha) must be greater than 0.")

    result = forecast_2026(
        evaluation=evaluation,
        area_overrides={region: float(area_ha)},
    )
    selected = result[result["region"] == region].copy().reset_index(drop=True)
    if selected.empty:
        raise gr.Error(f"No forecast returned for region: {region}")

    row = selected.iloc[0]
    summary = f"""
<div class="forecast-card">
    <div class="forecast-eyebrow">2026/27 forecast</div>
    <div class="forecast-title">{row['region']}</div>
    <div class="forecast-grid">
        <div class="big-stat">
            <span>Predicted production</span>
            <strong>{number(row['predicted_production_hl'])} hl</strong>
        </div>
        <div class="big-stat">
            <span>Empirical 90% interval</span>
            <strong>{number(row['lower_90_hl'])} – {number(row['upper_90_hl'])} hl</strong>
        </div>
        <div class="big-stat">
            <span>Vineyard-area scenario</span>
            <strong>{number(row['vineyard_area_ha'], 2)} ha</strong>
        </div>
        <div class="big-stat">
            <span>Selected model</span>
            <strong>{row['selected_model']}</strong>
        </div>
    </div>
    <p class="card-note">
        Scenario output only: changing vineyard area does not prove causality.
    </p>
</div>
"""
    return summary, format_forecast_table(selected)


custom_css = """
:root {
    --wine: #6f1232;
    --wine-dark: #3d071a;
    --rose: #fff4f6;
    --rose-2: #fbe7eb;
    --gold: #c29348;
    --ink: #24151a;
    --muted: #76676b;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background:
        radial-gradient(circle at 10% 0%, rgba(194, 147, 72, 0.18), transparent 26%),
        radial-gradient(circle at 95% 8%, rgba(111, 18, 50, 0.16), transparent 24%),
        linear-gradient(180deg, #fffafa 0%, #fbf5f1 100%) !important;
}

.hero {
    border-radius: 28px;
    padding: 34px 38px;
    margin: 8px 0 22px;
    color: white;
    background:
        linear-gradient(135deg, rgba(61, 7, 26, 0.98), rgba(111, 18, 50, 0.94)),
        radial-gradient(circle at 85% 20%, rgba(194, 147, 72, 0.45), transparent 28%);
    box-shadow: 0 22px 55px rgba(61, 7, 26, 0.22);
}

.hero-kicker {
    display: inline-block;
    padding: 6px 12px;
    border: 1px solid rgba(255, 255, 255, 0.28);
    border-radius: 999px;
    color: #ffe6b8;
    font-size: 0.84rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 14px 0 8px;
    font-size: clamp(2rem, 4vw, 3.5rem);
    line-height: 1.02;
}

.hero p {
    max-width: 780px;
    color: rgba(255, 255, 255, 0.86);
    font-size: 1.04rem;
}

.metric-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin: 0 0 20px;
}

.metric-card {
    border: 1px solid rgba(111, 18, 50, 0.10);
    border-radius: 20px;
    padding: 18px 20px;
    background: rgba(255, 255, 255, 0.78);
    box-shadow: 0 12px 34px rgba(61, 7, 26, 0.08);
}

.metric-card span {
    display: block;
    color: var(--muted);
    font-size: 0.83rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.metric-card strong {
    display: block;
    margin-top: 6px;
    color: var(--wine-dark);
    font-size: 1.25rem;
}

.interpretation {
    border-left: 5px solid var(--gold);
    border-radius: 18px;
    padding: 16px 18px;
    margin-bottom: 22px;
    background: rgba(255, 250, 239, 0.86);
    color: var(--ink);
}

.panel-card {
    border: 1px solid rgba(111, 18, 50, 0.11) !important;
    border-radius: 24px !important;
    padding: 18px !important;
    background: rgba(255, 255, 255, 0.84) !important;
    box-shadow: 0 16px 38px rgba(61, 7, 26, 0.07) !important;
}

.section-title {
    color: var(--wine-dark);
    margin-bottom: 6px;
}

.forecast-card {
    border: 1px solid rgba(111, 18, 50, 0.14);
    border-radius: 26px;
    padding: 26px;
    background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(255, 244, 246, 0.98));
    box-shadow: 0 18px 46px rgba(61, 7, 26, 0.12);
}

.forecast-eyebrow {
    color: var(--gold);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.78rem;
}

.forecast-title {
    margin-top: 4px;
    color: var(--wine-dark);
    font-size: 2rem;
    font-weight: 800;
}

.forecast-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 14px;
    margin-top: 20px;
}

.big-stat {
    border-radius: 18px;
    padding: 15px 16px;
    background: white;
    border: 1px solid rgba(111, 18, 50, 0.08);
}

.big-stat span {
    display: block;
    color: var(--muted);
    font-size: 0.82rem;
}

.big-stat strong {
    display: block;
    margin-top: 4px;
    color: var(--wine);
    font-size: 1.12rem;
}

.card-note {
    margin: 18px 0 0;
    color: var(--muted);
    font-size: 0.92rem;
}

button.primary {
    background: linear-gradient(135deg, var(--wine), #e64267) !important;
    border: none !important;
    box-shadow: 0 12px 24px rgba(111, 18, 50, 0.24) !important;
}

@media (max-width: 850px) {
    .metric-row,
    .forecast-grid {
        grid-template-columns: 1fr;
    }
}
"""

theme = gr.themes.Soft(primary_hue="rose", secondary_hue="orange", neutral_hue="stone")

with gr.Blocks(
    title="Wine Production Predictor",
    theme=theme,
    css=custom_css,
) as demo:
    gr.HTML(
        """
        <section class="hero">
            <span class="hero-kicker">Experimental forecast prototype</span>
            <h1>Wine Production Predictor</h1>
            <p>
                Forecast 2026/27 wine production for Portuguese viticultural
                regions using lagged production, vineyard area scenarios, and
                rolling-origin model validation.
            </p>
        </section>
        """
    )

    gr.HTML(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <span>Selected model</span>
                <strong>{evaluation.selected_model}</strong>
            </div>
            <div class="metric-card">
                <span>Empirical interval</span>
                <strong>± {number(evaluation.interval_half_width)} hl</strong>
            </div>
            <div class="metric-card">
                <span>Regions covered</span>
                <strong>{len(region_choices)}</strong>
            </div>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="interpretation">
            <b>Important interpretation.</b> The vineyard-area input creates a
            scenario, not a causal estimate. The 90% interval is empirical,
            based on validation errors, and should not be read as a formal
            statistical confidence interval.
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1, elem_classes=["panel-card"]):
            gr.Markdown("### Simulation inputs", elem_classes=["section-title"])
            region = gr.Dropdown(
                choices=region_choices,
                value=default_region,
                label="Region",
                info="Choose a Portuguese wine region.",
            )
            area = gr.Number(
                value=round(float(area_map.get(default_region, 0.0)), 2),
                label="Vineyard area (ha)",
                precision=2,
                info=(
                    "Adjust vineyard area as a scenario input; this is not a "
                    "causal effect estimate."
                ),
            )
            run_button = gr.Button("Forecast 2026/27", variant="primary")

        with gr.Column(scale=2, elem_classes=["panel-card"]):
            gr.Markdown("### Prediction result", elem_classes=["section-title"])
            summary = gr.HTML()
            result_table = gr.Dataframe(
                label="Forecast result",
                interactive=False,
                wrap=True,
            )

    with gr.Accordion("Validation metrics", open=False):
        gr.Dataframe(
            value=format_metrics_table(validation_metrics),
            label="Rolling validation 2018-2022",
            interactive=False,
            wrap=True,
        )

    with gr.Accordion("Final one-step-ahead test metrics", open=False):
        gr.Dataframe(
            value=format_metrics_table(test_metrics),
            label="Test years 2023-2025",
            interactive=False,
            wrap=True,
        )

    with gr.Accordion("Forecast table for all regions", open=False):
        gr.Dataframe(
            value=format_forecast_table(base_forecast),
            label="Default 2026/27 forecasts for all regions",
            interactive=False,
            wrap=True,
        )

    region.change(fn=update_area, inputs=region, outputs=area)
    run_button.click(fn=simulate, inputs=[region, area], outputs=[summary, result_table])
    demo.load(fn=simulate, inputs=[region, area], outputs=[summary, result_table])


if __name__ == "__main__":
    demo.launch()
