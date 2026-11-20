---
title: Air Quality AQI Explainer
emoji: 🌬️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Air Quality — EPA AQI Explainer

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/waleedarif1011/air-quality-prediction)
[![CI](https://github.com/waleedarif1011/air-quality-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/waleedarif1011/air-quality-prediction/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**GitHub description (suggested):** EPA AQI category mapper + Gradio explainer — pollutants → AQI (derived target; high R² expected).

Interactive **EPA Air Quality Index (AQI) explainer**: enter pollutant concentrations (CO, NO₂, SO₂, O₃, PM2.5, PM10), see a mapped **EPA health category** (Good → Hazardous), and compare a small regression model that recovers AQI from those same inputs.

> **What this is (and isn’t):** Official AQI is largely a **deterministic function of pollutant concentrations**. The features here *are* the pollutants that define AQI, so an R² near **0.99** is **expected** — a model-check / teaching demo, **not** a forecasting breakthrough or next-hour prediction from meteorology/lags.

> **Live demo:** after the Hugging Face Space is created (see [Deploy](#deploy-hugging-face-space)), open the badge URL above. Until then, run locally with `python app.py`.

## What you get

1. **Category mapper** — AQI score → EPA band + short health advisory.
2. **Model check** — HistGradientBoosting (and baselines) fitted on the same pollutant → AQI relationship; useful for verifying pipeline/metrics, not for claiming novel air-quality forecast skill.
3. **Presets** — Clean coastal day, rush-hour smog, wildfire smoke.

## Results (held-out 20%)

| Model | R² | MAE | RMSE |
|-------|----|-----|------|
| Linear Regression | 0.991 | 3.77 | 4.71 |
| Random Forest | 0.992 | 3.55 | 4.39 |
| **HistGradientBoosting** | **0.992** | **3.52** | **4.34** |

High R² is expected because the target is derived from the feature pollutants. Full numbers: [`artifacts/metrics.json`](artifacts/metrics.json).

### Feature importance & actual vs predicted

![Feature importance](assets/feature_importance.png)

![Actual vs predicted](assets/actual_vs_predicted.png)

## Quickstart

```bash
python -m pip install -r requirements.txt
python -m src.train          # train + write artifacts/ and assets/
python app.py                # Gradio demo at http://127.0.0.1:7860
```

### CLI inference

```bash
python -m src.predict --co 120 --no2 8 --so2 2 --o3 35 --pm25 6 --pm10 12
python -m src.predict --co 900 --no2 95 --so2 40 --o3 70 --pm25 180 --pm10 320
```

Returns JSON with `aqi`, EPA `category`, `color`, and `advisory`.

## Method (short)

- **Data:** Bundled `data/air_quality_sample.csv` (4k rows, six cities / pollutants + AQI). Optional full Kaggle pull via `scripts/fetch_data.py` when credentials exist.
- **Features → target:** CO, NO₂, SO₂, O₃, PM2.5, PM10 → AQI (derived from those pollutants).
- **Split:** 80/20 train/test (`random_state=42`).
- **Models:** Linear Regression (scaled), Random Forest, HistGradientBoosting — best saved to `artifacts/aqi_model.joblib`.
- **EPA mapping:** Good (0–50) → Hazardous (301+) with health advisory text in `src/predict.py`.
- **Demo presets:** Clean coastal day, Rush-hour smog, Wildfire smoke event.

## Tests & CI

```bash
python -m pytest -q
```

GitHub Actions runs the same suite on every push/PR (`.github/workflows/ci.yml`).

## Deploy (Hugging Face Space)

This repo is Space-ready (`app.py` + Gradio README frontmatter). Create and push:

```bash
# once: huggingface-cli login   # needs a write token with Spaces scope
huggingface-cli repo create air-quality-prediction --type space --space_sdk gradio -y
git remote add space https://huggingface.co/spaces/waleedarif1011/air-quality-prediction
git push space HEAD:main
```

Step-by-step: [`docs/hf-space-deploy.md`](docs/hf-space-deploy.md).

## Layout

```
app.py                 Gradio EPA AQI explainer (3 presets)
src/train.py           multi-model train + metrics/plots
src/predict.py         CLI + EPA category mapping
data/air_quality_sample.csv
scripts/fetch_data.py  optional Kaggle download
artifacts/             model + metrics.json
assets/                feature importance + actual-vs-predicted
notebooks/             original exploratory notebook
tests/                 pytest suite
```

## Topics (suggested for GitHub)

`air-quality` · `aqi` · `epa` · `gradio` · `scikit-learn` · `portfolio`

## License

MIT — see [`LICENSE`](LICENSE).
