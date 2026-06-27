# Predicting Annual Wine Production by Viticultural Region in Portugal

Final project for Practical Machine Learning, Master's in Green Data Science,
Instituto Superior de Agronomia, Universidade de Lisboa (2025/2026).

The project studies annual wine-production forecasting for 14 Portuguese
viticultural regions using official Instituto da Vinha e do Vinho (IVV) data.
It compares transparent temporal baselines with linear, regularized linear, and
random-forest regression models.

## Team

- No. 27119 - Andrea Dombe
- No. 27916 - Dandara França
- No. 26298 - Fernanda Chácara

## Main findings

Model selection used rolling-origin validation over 2018-2022. The years
2023-2025 were kept as a final chronological test set. Linear regression was the
best machine-learning candidate during validation, but a simple persistence
baseline performed better on the untouched test period:

| Model | Test MAE (hl) | Test RMSE (hl) | Test R² |
|---|---:|---:|---:|
| Persistence: last regional value | 55,139 | 98,771 | 0.961 |
| Historical regional mean | 70,377 | 117,842 | 0.945 |
| Linear regression | 78,982 | 115,420 | 0.947 |
| Random forest, min. leaf 1 | 102,482 | 237,583 | 0.777 |

The result is intentionally reported without claiming that the ML model beats
the baseline. Region and vineyard area explain much of the cross-sectional
variation, but the available predictors do not consistently capture annual
changes.

## Repository structure

```text
data/raw/                 Original IVV workbooks
data/processed/           Clean model-ready CSV files
notebooks/                Data cleaning, EDA, and modelling notebooks
notebooks/03_forecast_simulator.ipynb
                          Combined fixed-test baseline evaluation and experimental
                          2026/27 forecast simulator
outputs/figures/          Generated figures
outputs/tables/           Metrics and test predictions
project_proposal/         Submitted proposal
report/pml_wine_production_report.pdf
                          Final complete project report
src/evaluate_models.py    Reproducible temporal evaluation
src/app.py                Optional interactive Gradio forecast app
requirements.txt          Python dependencies
```

## Reproduce the final evaluation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/evaluate_models.py
```

The script recreates:

- `outputs/tables/model_comparison.csv`
- `outputs/tables/test_predictions.csv`
- `outputs/figures/model_comparison_mae.png`

The experimental future-forecast workflow can be reproduced with:

```bash
python src/future_forecast.py
```

It creates `outputs/tables/forecast_2026_27.csv` and the associated lag-model
validation and test tables. The forecast interval uses region-specific
validation errors plus a historical regional lower floor, so regions with
positive historical production are not shown with misleading zero lower bounds.
The notebook `03_forecast_simulator.ipynb`
consolidates the former simulator notebooks: it first reproduces the strict
fixed 2023-2025 baseline evaluation and then presents the experimental
one-step-ahead 2026/27 forecast simulator with an interactive region and
vineyard-area scenario interface.

The optional local Gradio app can be launched with:

```bash
python src/app.py
```

It uses the same `forecast_2026()` function as the notebook and shows the
predicted 2026/27 production by region. Vineyard area is treated only as a
scenario input, and the 90% band is an empirical validation-error band rather
than a formal confidence interval.

## Hugging Face / Gradio Space

An interactive deployment of the experimental forecast workflow is available on
Hugging Face Spaces, built with Gradio and using the same methodology as
`src/future_forecast.py` and the consolidated simulator notebook.

The app allows users to explore the 2026/27 production forecast for each
viticultural region, with the option to adjust vineyard area as a scenario
input. It loads the same processed datasets and uses the selected lagged random
forest model to generate predictions. The displayed uncertainty band
corresponds to empirical validation errors, not formal confidence intervals,
and follows the same lower-bound safeguard described in the report and
notebooks.

The app is intended for demonstration and exploratory use only, not as a formal
decision-support tool.

🔗 **Live app:** [https://huggingface.co/spaces/oxedanda/wine-production-predictor](https://huggingface.co/spaces/oxedanda/wine-production-predictor)

## Data sources

- [IVV wine production statistics](https://www.ivv.gov.pt/np4/163.html)
- [IVV vineyard-area statistics](https://www.ivv.gov.pt/np4/10586.html)

The original and processed datasets required for the submitted analysis are
included in this repository.

## Colab notebooks

Each principal notebook contains an “Open in Colab” badge. For the final model
comparison, the repository scripts are the authoritative reproducible workflows.
Notebook 03 contains the consolidated forecast simulator and reports an
empirical 90% error interval rather than presenting the point forecast as
certain.
