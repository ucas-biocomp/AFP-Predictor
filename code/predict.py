#!/usr/bin/env python3
"""Predict AFP labels with the explicit or implicit SVM ensemble."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "SVM_implicit_ensemble_models.joblib"


def load_bundle(feature: str) -> dict:
    path = MODEL_PATH
    if not path.is_file():
        raise FileNotFoundError(f"SVM model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not bundle.get("models") or "ensemble_threshold" not in bundle:
        raise ValueError(f"Invalid model bundle: {path}")
    return bundle


def load_explicit_csv(path: Path, id_column: str | None) -> tuple[list[str], np.ndarray]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("The input CSV contains no feature rows")
    if id_column:
        if id_column not in frame.columns:
            raise ValueError(f"ID column not found: {id_column}")
        ids = frame.pop(id_column).astype(str).tolist()
    else:
        ids = [str(index) for index in range(len(frame))]
    try:
        matrix = frame.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("The explicit CSV must contain numeric feature columns and an optional ID column") from exc
    return ids, matrix


def load_implicit_embeddings(path: Path) -> tuple[list[str], np.ndarray]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Implicit prediction requires PyTorch") from exc

    files = [path] if path.is_file() else sorted(path.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt embedding files found: {path}")

    ids = []
    vectors = []
    for file in files:
        data = torch.load(file, map_location="cpu", weights_only=False)
        if isinstance(data, dict) and "mean_representations" in data:
            representations = data["mean_representations"]
            vector = representations[33] if 33 in representations else representations["33"]
            sample_id = str(data.get("label", file.stem))
        elif isinstance(data, dict) and "features" in data:
            vector = data["features"]
            sample_id = str(data.get("label", file.stem))
        else:
            raise ValueError(f"Unsupported embedding format: {file}")
        if hasattr(vector, "detach"):
            vector = vector.detach().cpu().numpy()
        ids.append(sample_id)
        vectors.append(np.asarray(vector, dtype=np.float64).reshape(-1))
    return ids, np.stack(vectors)


def predict(input_path: Path, feature: str, id_column: str | None) -> pd.DataFrame:
    bundle = load_bundle(feature)
    if feature == "explicit":
        ids, matrix = load_explicit_csv(input_path, id_column)
    else:
        ids, matrix = load_implicit_embeddings(input_path)

    expected = getattr(bundle["models"][0], "n_features_in_", None)
    if expected is not None and matrix.shape[1] != expected:
        raise ValueError(f"Expected {expected} {feature} features, received {matrix.shape[1]}")
    if not np.isfinite(matrix).all():
        raise ValueError("Input features contain NaN or infinite values")

    probabilities = np.column_stack([model.predict_proba(matrix)[:, 1] for model in bundle["models"]])
    mean_probability = probabilities.mean(axis=1)
    mean_threshold = float(bundle["ensemble_threshold"])
    return pd.DataFrame({
        "id": ids,
        "score": mean_probability,
        "threshold": mean_threshold,
        "prediction": np.where(mean_probability >= mean_threshold, "AFP", "non-AFP"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict AFP labels with the SVM ensemble")
    parser.add_argument("input", type=Path, help="Implicit: one .pt file or directory; explicit: a feature CSV")
    parser.add_argument("--feature", choices=("explicit", "implicit"), required=True)
    parser.add_argument("--id-column", help="Optional ID column in an explicit feature CSV")
    parser.add_argument("--output", type=Path, help="Optional CSV output; otherwise print one label per row")
    args = parser.parse_args()

    result = predict(args.input, args.feature, args.id_column)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False, encoding="utf-8-sig", float_format="%.10g")
    else:
        for label in result["prediction"]:
            print(label)


if __name__ == "__main__":
    main()
