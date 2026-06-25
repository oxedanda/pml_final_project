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
            view[column] = view[column].map(
                lambda value: f"{value:,.0f}" if pd.notna(value) else ""
            )

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
<div class="summary-card">
    <h3>Forecast summary</h3>
    <p><b>Region:</b> {row['region']}</p>
    <p><b>Campaign:</b> 2026/27</p>
    <p><b>Selected model:</b> {row['selected_model']}</p>
    <p><b>Vineyard area scenario:</b> {row['vineyard_area_ha']:,.2f} ha</p>
    <p><b>Predicted production:</b> {row['predicted_production_hl']:,.0f} hl</p>
    <p><b>Empirical 90% interval:</b> {row['lower_90_hl']:,.0f}
       to {row['upper_90_hl']:,.0f} hl</p>
</div>
"""
    return summary, format_forecast_table(selected)


custom_css = """
.gradio-container {
    max-width: 1120px !important;
}
.summary-card {
    border: 1px solid #ead7d7;
    border-radius: 14px;
    padding: 18px;
    background: #fff8f8;
}
"""

theme = gr.themes.Soft(primary_hue="rose", secondary_hue="orange")

with gr.Blocks(
    title="Wine Production Predictor",
    theme=theme,
    css=custom_css,
) as demo:
    gr.Markdown(
        """
        # Wine Production Predictor

        Experimental 2026/27 wine-production forecast by Portuguese
        viticultural region.

        **Important interpretation:** the vineyard-area input creates a
        scenario, not a causal estimate. The empirical 90% interval is based on
        observed validation errors and should not be read as a formal
        statistical confidence interval.
        """
    )

    gr.Markdown(
        f"""
        **Selected one-step-ahead model:** `{evaluation.selected_model}`

        **Empirical 90% interval half-width:** `{evaluation.interval_half_width:,.0f} hl`
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Simulation inputs")
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

        with gr.Column(scale=2):
            gr.Markdown("### Prediction result")
            summary = gr.HTML()
            result_table = gr.Dataframe(label="Forecast result", interactive=False)

    with gr.Accordion("Validation metrics", open=False):
        gr.Dataframe(
            value=validation_metrics[["model", "MAE", "RMSE", "R2"]],
            label="Rolling validation 2018-2022",
            interactive=False,
        )

    with gr.Accordion("Final one-step-ahead test metrics", open=False):
        gr.Dataframe(
            value=test_metrics[["model", "MAE", "RMSE", "R2"]],
            label="Test years 2023-2025",
            interactive=False,
        )

    with gr.Accordion("Forecast table for all regions", open=False):
        gr.Dataframe(
            value=format_forecast_table(base_forecast),
            label="Default 2026/27 forecasts for all regions",
            interactive=False,
        )

    region.change(fn=update_area, inputs=region, outputs=area)
    run_button.click(fn=simulate, inputs=[region, area], outputs=[summary, result_table])
    demo.load(fn=simulate, inputs=[region, area], outputs=[summary, result_table])


if __name__ == "__main__":
    demo.launch()
