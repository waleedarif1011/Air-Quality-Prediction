"""CLI and programmatic AQI prediction with EPA health categories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.train import FEATURE_COLS, MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]

# EPA AQI breakpoints (inclusive upper bound of each band except Hazardous).
EPA_CATEGORIES: list[dict[str, Any]] = [
    {
        "name": "Good",
        "min": 0,
        "max": 50,
        "color": "#00E400",
        "advisory": "Air quality is satisfactory; little or no risk.",
    },
    {
        "name": "Moderate",
        "min": 51,
        "max": 100,
        "color": "#FFFF00",
        "advisory": (
            "Acceptable; unusually sensitive people should consider limiting "
            "prolonged outdoor exertion."
        ),
    },
    {
        "name": "Unhealthy for Sensitive Groups",
        "min": 101,
        "max": 150,
        "color": "#FF7E00",
        "advisory": (
            "Members of sensitive groups may experience health effects; "
            "the general public is less likely to be affected."
        ),
    },
    {
        "name": "Unhealthy",
        "min": 151,
        "max": 200,
        "color": "#FF0000",
        "advisory": (
            "Everyone may begin to experience health effects; "
            "sensitive groups may experience more serious effects."
        ),
    },
    {
        "name": "Very Unhealthy",
        "min": 201,
        "max": 300,
        "color": "#8F3F97",
        "advisory": (
            "Health alert: everyone may experience more serious health effects. "
            "Avoid outdoor activity."
        ),
    },
    {
        "name": "Hazardous",
        "min": 301,
        "max": 500,
        "color": "#7E0023",
        "advisory": (
            "Emergency conditions: the entire population is more likely to be affected. "
            "Remain indoors and follow local guidance."
        ),
    },
]


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing model at {path}. Run: python -m src.train"
        )
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Unexpected artifact format in {path}")
    return artifact


def epa_category(aqi: float) -> dict[str, Any]:
    """Map a numeric AQI to the official EPA bucket + advisory text."""
    value = float(aqi)
    if value < 0:
        value = 0.0
    for band in EPA_CATEGORIES:
        if value <= band["max"]:
            return {
                "category": band["name"],
                "color": band["color"],
                "advisory": band["advisory"],
                "range": [band["min"], band["max"]],
            }
    # Above 500 — still Hazardous.
    last = EPA_CATEGORIES[-1]
    return {
        "category": last["name"],
        "color": last["color"],
        "advisory": last["advisory"],
        "range": [last["min"], last["max"]],
    }


def _as_feature_frame(features: dict[str, float], columns: list[str]) -> pd.DataFrame:
    missing = [c for c in columns if c not in features]
    if missing:
        raise ValueError(f"Missing pollutant features: {missing}")
    row = {c: float(features[c]) for c in columns}
    return pd.DataFrame([row], columns=columns)


def predict(
    features: dict[str, float],
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Predict numeric AQI and attach EPA category metadata."""
    art = artifact or load_model()
    columns = list(art.get("features") or FEATURE_COLS)
    frame = _as_feature_frame(features, columns)
    raw = float(np.asarray(art["model"].predict(frame), dtype=float).ravel()[0])
    aqi = float(np.clip(raw, 0.0, 500.0))
    bucket = epa_category(aqi)
    return {
        "aqi": round(aqi, 2),
        "aqi_raw": raw,
        "category": bucket["category"],
        "color": bucket["color"],
        "advisory": bucket["advisory"],
        "range": bucket["range"],
        "model_name": art.get("model_name"),
        "features": {c: float(features[c]) for c in columns},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AQI from pollutant concentrations.")
    parser.add_argument("--co", type=float, required=True, help="CO concentration")
    parser.add_argument("--no2", type=float, required=True, help="NO2 concentration")
    parser.add_argument("--so2", type=float, required=True, help="SO2 concentration")
    parser.add_argument("--o3", type=float, required=True, help="O3 concentration")
    parser.add_argument("--pm25", type=float, required=True, dest="pm25", help="PM2.5")
    parser.add_argument("--pm10", type=float, required=True, help="PM10")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to aqi_model.joblib",
    )
    args = parser.parse_args()

    features = {
        "CO": args.co,
        "NO2": args.no2,
        "SO2": args.so2,
        "O3": args.o3,
        "PM2.5": args.pm25,
        "PM10": args.pm10,
    }
    result = predict(features, artifact=load_model(args.model))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
