"""Tests for EPA bucketing, model load, and prediction sanity."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.predict import EPA_CATEGORIES, epa_category, load_model, predict
from src.train import METRICS_PATH, MODEL_PATH, load_air_quality, train

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "air_quality_sample.csv"

CLEAN_AIR = {
    "CO": 120.0,
    "NO2": 8.0,
    "SO2": 2.0,
    "O3": 35.0,
    "PM2.5": 6.0,
    "PM10": 12.0,
}
SMOKY_AIR = {
    "CO": 900.0,
    "NO2": 95.0,
    "SO2": 40.0,
    "O3": 70.0,
    "PM2.5": 180.0,
    "PM10": 320.0,
}


@pytest.fixture(scope="module")
def trained_metrics():
    assert DATA.is_file(), "data/air_quality_sample.csv must ship with the repo"
    metrics = train(DATA)
    assert MODEL_PATH.is_file()
    assert METRICS_PATH.is_file()
    assert metrics["best"]["r2"] >= 0.85
    return metrics


@pytest.fixture(scope="module")
def artifact(trained_metrics):
    return load_model(MODEL_PATH)


def test_dataset_shape():
    df = load_air_quality(DATA)
    assert len(df) >= 500
    for col in ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10", "AQI"]:
        assert col in df.columns
    assert df["AQI"].min() >= 0


@pytest.mark.parametrize(
    "aqi,expected",
    [
        (0, "Good"),
        (42, "Good"),
        (50, "Good"),
        (51, "Moderate"),
        (100, "Moderate"),
        (101, "Unhealthy for Sensitive Groups"),
        (150, "Unhealthy for Sensitive Groups"),
        (151, "Unhealthy"),
        (200, "Unhealthy"),
        (201, "Very Unhealthy"),
        (300, "Very Unhealthy"),
        (301, "Hazardous"),
        (450, "Hazardous"),
    ],
)
def test_epa_category_buckets(aqi, expected):
    result = epa_category(aqi)
    assert result["category"] == expected
    assert result["advisory"]
    assert result["color"].startswith("#")


def test_epa_categories_cover_full_scale():
    names = [c["name"] for c in EPA_CATEGORIES]
    assert names == [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    ]


def test_model_file_loads(artifact):
    assert artifact is not None
    assert "model" in artifact
    assert hasattr(artifact["model"], "predict")


def test_high_pm25_yields_higher_aqi(artifact):
    clean = predict(CLEAN_AIR, artifact=artifact)
    smoky = predict(SMOKY_AIR, artifact=artifact)
    assert smoky["aqi"] > clean["aqi"]
    assert clean["aqi"] < 80
    assert smoky["aqi"] > clean["aqi"] + 50
    assert clean["category"] in {"Good", "Moderate"}
    assert smoky["category"] in {
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous",
    }


def test_prediction_bounds(artifact):
    for features in (CLEAN_AIR, SMOKY_AIR):
        result = predict(features, artifact=artifact)
        assert 0.0 <= result["aqi"] <= 500.0
        assert result["category"] in {c["name"] for c in EPA_CATEGORIES}


def test_metrics_include_all_models(trained_metrics):
    names = {m["name"] for m in trained_metrics["models"]}
    assert names == {
        "linear_regression",
        "random_forest",
        "hist_gradient_boosting",
    }
    assert trained_metrics["best_model"] in names
