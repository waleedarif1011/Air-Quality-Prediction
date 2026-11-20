"""Gradio live demo — EPA AQI explainer / category mapper from pollutant inputs."""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from src.predict import load_model, predict
from src.train import IMPORTANCE_PATH, METRICS_PATH, MODEL_PATH

ROOT = Path(__file__).resolve().parent

PRESETS = [
    {
        "label": "Clean coastal day",
        "values": {
            "CO": 120.0,
            "NO2": 8.0,
            "SO2": 2.0,
            "O3": 35.0,
            "PM2.5": 6.0,
            "PM10": 12.0,
        },
    },
    {
        "label": "Rush-hour smog",
        "values": {
            "CO": 520.0,
            "NO2": 75.0,
            "SO2": 28.0,
            "O3": 55.0,
            "PM2.5": 55.0,
            "PM10": 110.0,
        },
    },
    {
        "label": "Wildfire smoke event",
        "values": {
            "CO": 900.0,
            "NO2": 95.0,
            "SO2": 40.0,
            "O3": 70.0,
            "PM2.5": 180.0,
            "PM10": 320.0,
        },
    },
]

_ARTIFACT = None


def _get_artifact():
    global _ARTIFACT
    if _ARTIFACT is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                "artifacts/aqi_model.joblib missing — run `python -m src.train` first."
            )
        _ARTIFACT = load_model(MODEL_PATH)
    return _ARTIFACT


def _metrics_blurb() -> str:
    if not METRICS_PATH.is_file():
        return "_Train with `python -m src.train` to populate metrics._"
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    best = m.get("best", {})
    rows = "\n".join(
        f"| `{r['name']}` | {r['r2']:.3f} | {r['mae']:.2f} | {r['rmse']:.2f} |"
        for r in m.get("models", [])
    )
    return (
        "_AQI is derived from these pollutant features, so R² near 0.99 is "
        "expected — this is a model check / teaching demo, not a forecasting "
        "breakthrough._\n\n"
        f"**Best model:** `{best.get('name', '—')}` "
        f"(R²={best.get('r2', float('nan')):.3f}, "
        f"MAE={best.get('mae', float('nan')):.2f})\n\n"
        "| Model | R² | MAE | RMSE |\n"
        "|-------|----|-----|------|\n"
        f"{rows}"
    )


def _card_html(result: dict) -> str:
    color = result["color"]
    return f"""
<div style="border-radius:12px;padding:1.1rem 1.25rem;background:{color}22;
            border:2px solid {color};font-family:system-ui,sans-serif;">
  <div style="font-size:0.85rem;letter-spacing:0.04em;text-transform:uppercase;
              opacity:0.75;">Mapped AQI</div>
  <div style="font-size:2.6rem;font-weight:700;line-height:1.1;color:#111;">
    {result['aqi']:.0f}
  </div>
  <div style="margin-top:0.35rem;display:inline-block;padding:0.25rem 0.7rem;
              border-radius:999px;background:{color};color:#111;font-weight:600;">
    {result['category']}
  </div>
  <p style="margin:0.85rem 0 0;color:#222;line-height:1.45;">{result['advisory']}</p>
</div>
"""


def estimate(co: float, no2: float, so2: float, o3: float, pm25: float, pm10: float):
    features = {
        "CO": float(co),
        "NO2": float(no2),
        "SO2": float(so2),
        "O3": float(o3),
        "PM2.5": float(pm25),
        "PM10": float(pm10),
    }
    result = predict(features, artifact=_get_artifact())
    summary = (
        f"### Mapped AQI **{result['aqi']:.1f}** — {result['category']}\n\n"
        f"{result['advisory']}"
    )
    plot = str(IMPORTANCE_PATH) if IMPORTANCE_PATH.is_file() else None
    return summary, _card_html(result), plot


def apply_preset(name: str):
    match = next((p for p in PRESETS if p["label"] == name), PRESETS[0])
    v = match["values"]
    return v["CO"], v["NO2"], v["SO2"], v["O3"], v["PM2.5"], v["PM10"]


def build_ui() -> gr.Blocks:
    defaults = PRESETS[0]["values"]
    with gr.Blocks(title="EPA AQI Explainer") as demo:
        gr.Markdown(
            "# EPA AQI Explainer / Category Mapper\n\n"
            "Enter pollutant concentrations (CO, NO₂, SO₂, O₃, PM2.5, PM10) to "
            "map an **AQI** score and the matching **EPA health category** "
            "(Good → Hazardous).\n\n"
            "Official AQI is largely determined by these pollutants — the model "
            "recovers that relationship as a check, not as novel forecasting.\n\n"
            + _metrics_blurb()
        )
        with gr.Row():
            with gr.Column(scale=3):
                preset = gr.Dropdown(
                    choices=[p["label"] for p in PRESETS],
                    value=PRESETS[0]["label"],
                    label="City / event presets",
                )
                co = gr.Slider(40, 2000, value=defaults["CO"], step=1, label="CO")
                no2 = gr.Slider(0, 250, value=defaults["NO2"], step=0.1, label="NO₂")
                so2 = gr.Slider(0, 120, value=defaults["SO2"], step=0.1, label="SO₂")
                o3 = gr.Slider(0, 200, value=defaults["O3"], step=0.1, label="O₃")
                pm25 = gr.Slider(0, 300, value=defaults["PM2.5"], step=0.1, label="PM2.5")
                pm10 = gr.Slider(0, 450, value=defaults["PM10"], step=0.1, label="PM10")
                run = gr.Button("Map AQI category", variant="primary")
            with gr.Column(scale=2):
                summary = gr.Markdown()
                card = gr.HTML()
                importance = gr.Image(label="Feature importance", type="filepath")

        inputs = [co, no2, so2, o3, pm25, pm10]
        outputs = [summary, card, importance]
        preset.change(fn=apply_preset, inputs=preset, outputs=inputs)
        run.click(fn=estimate, inputs=inputs, outputs=outputs)
        demo.load(fn=estimate, inputs=inputs, outputs=outputs)
    return demo


def main() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    main()
