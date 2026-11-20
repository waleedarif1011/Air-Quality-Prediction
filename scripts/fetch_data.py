#!/usr/bin/env python3
"""Optionally download the full Kaggle air-quality dataset when credentials exist.

Falls back to documenting that `data/air_quality_sample.csv` already ships in-repo.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SAMPLE_PATH = DATA_DIR / "air_quality_sample.csv"
RAW_DIR = DATA_DIR / "raw"
KAGGLE_SLUG = "youssefelebiary/global-air-quality-2023-6-cities"


def fetch(dest: Path = RAW_DIR) -> Path:
    """Download via kagglehub and copy CSVs into data/raw/."""
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "kagglehub is not installed. pip install kagglehub  "
            "— or use the bundled data/air_quality_sample.csv."
        ) from exc

    dest.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = Path(kagglehub.dataset_download(KAGGLE_SLUG))
    except Exception as exc:  # pragma: no cover - depends on live Kaggle auth
        raise SystemExit(
            f"Kaggle download failed ({exc}).\n"
            "Authenticate with Kaggle credentials, or rely on "
            f"{SAMPLE_PATH.relative_to(ROOT)} which is already committed."
        ) from exc

    copied: list[Path] = []
    for csv_path in downloaded.rglob("*.csv"):
        target = dest / csv_path.name
        shutil.copy2(csv_path, target)
        copied.append(target)

    if not copied:
        raise SystemExit(f"No CSV files found under {downloaded}")

    print(f"Downloaded to {downloaded}")
    for path in copied:
        print(f"  copied → {path.relative_to(ROOT)}")
    return copied[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=RAW_DIR,
        help="Directory for full CSV copies (default: data/raw/)",
    )
    args = parser.parse_args()

    if SAMPLE_PATH.is_file():
        print(
            f"Bundled sample ready: {SAMPLE_PATH.relative_to(ROOT)} "
            "(training/tests/demo need no Kaggle auth)."
        )

    fetch(args.dest)


if __name__ == "__main__":
    main()
