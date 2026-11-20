"""Train and compare AQI regressors; export best model + metrics/plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "air_quality_sample.csv"
ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
MODEL_PATH = ARTIFACTS / "aqi_model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"
IMPORTANCE_PATH = ASSETS / "feature_importance.png"
ACTUAL_VS_PRED_PATH = ASSETS / "actual_vs_predicted.png"

FEATURE_COLS = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10"]
TARGET_COL = "AQI"


def load_air_quality(path: str | Path = DEFAULT_DATA) -> pd.DataFrame:
    """Load pollutant + AQI CSV; keep numeric feature columns and target."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing dataset at {path}. Bundled sample should live at "
            f"{DEFAULT_DATA.relative_to(ROOT)}."
        )
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    out = df[FEATURE_COLS + [TARGET_COL]].apply(pd.to_numeric, errors="coerce")
    out = out.dropna().reset_index(drop=True)
    if len(out) < 50:
        raise ValueError(f"Too few usable rows in {path}: {len(out)}")
    return out


def build_models() -> dict[str, Any]:
    """Linear baseline + two tree ensembles (RF, HistGBDT)."""
    return {
        "linear_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=16,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.08,
            max_iter=250,
            random_state=42,
        ),
    }


def _evaluate(name: str, model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> dict[str, Any]:
    preds = np.asarray(model.predict(X_test), dtype=float).ravel()
    mse = float(mean_squared_error(y_test, preds))
    return {
        "name": name,
        "r2": float(r2_score(y_test, preds)),
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "y_pred": preds.tolist(),
    }


def _feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    """Extract importances from tree models; |coef| for linear pipeline."""
    if isinstance(model, Pipeline):
        est = model.named_steps.get("model", model[-1])
        if hasattr(est, "coef_"):
            coef = np.abs(np.asarray(est.coef_, dtype=float).ravel())
            total = float(coef.sum()) or 1.0
            return {f: float(v / total) for f, v in zip(feature_names, coef)}
        model = est

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_, dtype=float).ravel()
        return {f: float(v) for f, v in zip(feature_names, imp)}

    # HistGradientBoosting: permutation-free fallback via absolute prediction sensitivity.
    return {f: 0.0 for f in feature_names}


def _hgb_importance(model: HistGradientBoostingRegressor, X: pd.DataFrame) -> dict[str, float]:
    """Cheap leave-one-feature-shuffle importance for HistGBDT."""
    rng = np.random.default_rng(0)
    y_hat = np.asarray(model.predict(X), dtype=float)
    base_var = float(np.var(y_hat)) or 1.0
    scores: dict[str, float] = {}
    for col in X.columns:
        Xs = X.copy()
        Xs[col] = rng.permutation(Xs[col].to_numpy())
        shuffled = np.asarray(model.predict(Xs), dtype=float)
        scores[col] = float(np.mean((y_hat - shuffled) ** 2) / base_var)
    total = sum(scores.values()) or 1.0
    return {k: v / total for k, v in scores.items()}


def _save_importance(importances: dict[str, float], path: Path) -> None:
    names = list(importances.keys())
    values = [importances[n] for n in names]
    order = np.argsort(values)[::-1]
    names = [names[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=values, y=names, ax=ax, color="#2a6f97", orient="h")
    ax.set_xlabel("Relative importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature importance (best model)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _save_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=12, alpha=0.45, color="#1b4965", edgecolors="none")
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color="#c1121f", lw=1.5, linestyle="--", label="Ideal")
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title("Actual vs predicted AQI")
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(
    data_path: str | Path = DEFAULT_DATA,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    df = load_air_quality(data_path)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        fitted[name] = model
        results.append(_evaluate(name, model, X_test, y_test))

    best = max(results, key=lambda r: (r["r2"], -r["rmse"], -r["mae"]))
    best_model = fitted[best["name"]]

    if best["name"] == "hist_gradient_boosting":
        importances = _hgb_importance(best_model, X_test)
    else:
        importances = _feature_importance(best_model, FEATURE_COLS)
        if sum(importances.values()) == 0 and best["name"] == "hist_gradient_boosting":
            importances = _hgb_importance(best_model, X_test)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": best_model,
        "model_name": best["name"],
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "feature_importance": importances,
    }
    joblib.dump(artifact, MODEL_PATH)

    y_pred = np.asarray(best["y_pred"], dtype=float)
    _save_importance(importances, IMPORTANCE_PATH)
    _save_actual_vs_predicted(y_test, y_pred, ACTUAL_VS_PRED_PATH)

    models_summary = [
        {"name": r["name"], "r2": r["r2"], "mae": r["mae"], "rmse": r["rmse"]}
        for r in results
    ]
    metrics: dict[str, Any] = {
        "n_samples": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": FEATURE_COLS,
        "test_size": test_size,
        "random_state": random_state,
        "best_model": best["name"],
        "best": {
            "name": best["name"],
            "r2": best["r2"],
            "mae": best["mae"],
            "rmse": best["rmse"],
        },
        "models": models_summary,
        "feature_importance": importances,
        "artifacts": {
            "model": str(MODEL_PATH.relative_to(ROOT)),
            "feature_importance": str(IMPORTANCE_PATH.relative_to(ROOT)),
            "actual_vs_predicted": str(ACTUAL_VS_PRED_PATH.relative_to(ROOT)),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    metrics = train()
    best = metrics["best"]
    print(
        f"Best={best['name']} | "
        f"R²={best['r2']:.4f} MAE={best['mae']:.4f} RMSE={best['rmse']:.4f}"
    )
    print(f"Saved {MODEL_PATH} and {METRICS_PATH}")
    for m in metrics["models"]:
        print(
            f"  {m['name']}: R²={m['r2']:.4f} "
            f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f}"
        )


if __name__ == "__main__":
    main()
