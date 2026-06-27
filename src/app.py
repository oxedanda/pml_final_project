# prompt: Build a Gradio app to forecast Portuguese wine production for 2026/27 using
# local CSV data, lagged features, rolling temporal validation, a persistence
# baseline, and a 90% empirical uncertainty interval, with logic in
# `future_forecast.py` and a burgundy wine-themed UI in `app.py`

"""Interactive Gradio app for the experimental wine-production forecast.

Run from the repository root with:
    python src/app.py

The app is intentionally a visual prototype. It uses the same forecasting logic
as ``src/future_forecast.py`` and does not make causal claims about vineyard
area changes.
"""

from __future__ import annotations

import gradio as gr
import pandas as pd

from future_forecast import evaluate_forecasters, forecast_2026, load_history


def format_forecast_table(df: pd.DataFrame) -> pd.DataFrame:
    view = df.copy()

    numeric_cols = [
        "vineyard_area_ha",
        "predicted_production_hl",
        "lower_90_hl",
        "upper_90_hl",
    ]
    for col in numeric_cols:
        if col in view.columns:
            view[col] = view[col].map(lambda x: f"{x:,.0f}" if pd.notna(x) else "")

    return view.rename(
        columns={
            "region": "Region",
            "year_start": "Year",
            "vineyard_area_ha": "Area (ha)",
            "selected_model": "Selected model",
            "predicted_production_hl": "Predicted production (hl)",
            "lower_90_hl": "Lower 90% (hl)",
            "upper_90_hl": "Upper 90% (hl)",
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

area_map = dict(
    zip(latest_area["region"].astype(str), latest_area["vineyard_area_ha"])
)

validation_metrics = (
    evaluation.metrics
    .query("split == 'rolling_validation_2018_2022'")
    .sort_values("MAE")
    .reset_index(drop=True)
)

test_metrics = (
    evaluation.metrics
    .query("split == 'test_2023_2025'")
    .sort_values("MAE")
    .reset_index(drop=True)
)


def update_area(region: str):
    return gr.update(value=round(float(area_map.get(region, 0.0)), 2))


def simulate(region: str, area_ha: float):
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
    <div class="summary-title">🍷 Forecast summary</div>
    <div class="summary-grid">
        <div><span>Region</span><strong>{row['region']}</strong></div>
        <div><span>Campaign</span><strong>2026/27</strong></div>
        <div><span>Selected model</span><strong>{row['selected_model']}</strong></div>
        <div><span>Vineyard area</span><strong>{row['vineyard_area_ha']:,.2f} ha</strong></div>
        <div><span>Predicted production</span><strong>{row['predicted_production_hl']:,.0f} hl</strong></div>
        <div><span>90% interval</span><strong>{row['lower_90_hl']:,.0f} to {row['upper_90_hl']:,.0f} hl</strong></div>
    </div>
</div>
"""

    return summary, format_forecast_table(selected)


custom_css = """
.gradio-container {
    max-width: 1120px !important;
    background:
        radial-gradient(circle at top left, rgba(122, 24, 52, 0.14), transparent 28%),
        radial-gradient(circle at top right, rgba(177, 137, 74, 0.10), transparent 22%),
        linear-gradient(180deg, #fcf8f5 0%, #f7f1ec 100%);
}
.dark .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(122, 24, 52, 0.22), transparent 28%),
        radial-gradient(circle at top right, rgba(177, 137, 74, 0.10), transparent 22%),
        linear-gradient(180deg, #1a1115 0%, #140d10 100%);
}
#hero {
    text-align: center;
    padding: 22px 0 8px 0;
}
#hero h1 {
    margin-bottom: 0.35rem;
    font-size: 2.3rem;
    color: #6f1232;
    letter-spacing: -0.02em;
}
#hero p {
    margin: 0;
    color: #6e5a57;
    font-size: 1rem;
}
.dark #hero h1 {
    color: #f0d9df;
}
.dark #hero p {
    color: #c8b6b2;
}
.metric-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    margin: 8px 0 18px 0;
}
.metric-card {
    border-radius: 20px;
    padding: 16px 18px;
    background: linear-gradient(180deg, #fff7f8 0%, #fffaf5 100%);
    border: 1px solid #ead5db;
    box-shadow: 0 8px 24px rgba(111, 18, 50, 0.08);
}
.dark .metric-card {
    background: linear-gradient(180deg, #2a161d 0%, #231317 100%);
    border: 1px solid #55303b;
    box-shadow: none;
}
.metric-card .label {
    display: block;
    font-size: 0.92rem;
    color: #7a5461;
    margin-bottom: 6px;
}
.metric-card .value {
    display: block;
    font-size: 1.1rem;
    font-weight: 700;
    color: #6f1232;
}
.dark .metric-card .label {
    color: #d7b5bf;
}
.dark .metric-card .value {
    color: #f7e3e8;
}
.main-card {
    border-radius: 24px;
    padding: 18px;
    background: linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(255,250,246,0.95) 100%);
    border: 1px solid #e8d9d0;
    box-shadow: 0 12px 32px rgba(94, 36, 49, 0.08);
    margin-bottom: 18px;
}
.dark .main-card {
    background: linear-gradient(180deg, rgba(36,20,25,0.94) 0%, rgba(28,16,20,0.98) 100%);
    border: 1px solid #4c3138;
    box-shadow: none;
}
.section-title {
    color: #6f1232;
    margin-bottom: 6px;
}
.dark .section-title {
    color: #f2dbe2;
}
.summary-card {
    border-radius: 20px;
    padding: 18px;
    background: linear-gradient(180deg, #7a1834 0%, #5e1028 100%);
    color: #fff7f2;
    border: 1px solid #8b2c4a;
    box-shadow: 0 12px 28px rgba(94, 16, 40, 0.24);
}
.summary-title {
    font-size: 1.08rem;
    font-weight: 700;
    margin-bottom: 14px;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
}
.summary-grid div {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 12px 14px;
}
.summary-grid span {
    display: block;
    font-size: 0.85rem;
    color: #f3d9c9;
    margin-bottom: 4px;
}
.summary-grid strong {
    display: block;
    font-size: 1rem;
    color: #fffaf6;
}
button.primary {
    background: linear-gradient(180deg, #7a1834 0%, #5e1028 100%) !important;
    border: 1px solid #8d3250 !important;
}
button.primary:hover {
    filter: brightness(1.05);
}
@media (max-width: 768px) {
    .metric-row,
    .summary-grid {
        grid-template-columns: 1fr;
    }
    #hero h1 {
        font-size: 1.8rem;
    }
}
"""

theme = gr.themes.Soft(
    primary_hue="rose",
    secondary_hue="orange",
    neutral_hue="stone",
)

with gr.Blocks(
    title="🍷 Wine Production Predictor",
) as demo:
    gr.HTML(
        """
        <div id="hero">
            <h1>🍷 Wine Production Predictor</h1>
            <p>Experimental 2026/27 forecast by Portuguese viticultural region</p>
        </div>
        """
    )

    gr.HTML(
        f"""
        <div class="metric-row">
            <div class="metric-card">
                <span class="label">🏆 Selected model</span>
                <span class="value">{evaluation.selected_model}</span>
            </div>
            <div class="metric-card">
                <span class="label">📏 Empirical 90% interval half-width</span>
                <span class="value">{evaluation.interval_half_width:,.0f} hl</span>
            </div>
        </div>
        """
    )

    gr.HTML('<div class="main-card">')
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🍇 Simulation inputs", elem_classes=["section-title"])
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
                info="Adjust the vineyard area to simulate a different scenario.",
            )
            run_button = gr.Button("🍷 Forecast 2026/27", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### 📋 Prediction result", elem_classes=["section-title"])
            summary = gr.HTML()
            result_table = gr.Dataframe(
                label="Forecast result",
                interactive=False,
            )
    gr.HTML("</div>")

    with gr.Accordion("📊 Validation metrics", open=False):
        gr.Dataframe(
            value=validation_metrics[["model", "MAE", "RMSE", "R2"]],
            label="Rolling validation 2018–2022",
            interactive=False,
        )

    with gr.Accordion("🧪 Final test metrics", open=False):
        gr.Dataframe(
            value=test_metrics[["model", "MAE", "RMSE", "R2"]],
            label="Final one-step-ahead test 2023–2025",
            interactive=False,
        )

    with gr.Accordion("🗂️ Baseline forecast table", open=False):
        gr.Dataframe(
            value=format_forecast_table(base_forecast),
            label="Forecasts for all regions",
            interactive=False,
        )

    region.change(fn=update_area, inputs=region, outputs=area)
    run_button.click(fn=simulate, inputs=[region, area], outputs=[summary, result_table])
    demo.load(fn=simulate, inputs=[region, area], outputs=[summary, result_table])

if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css)
