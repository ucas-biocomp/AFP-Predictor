#!/usr/bin/env python3
"""Train the five-fold SVM threshold ensemble from feature and split CSV files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 84


def create_model(params: dict, seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=params["C"],
            kernel=params["kernel"],
            gamma=params["gamma"],
            degree=params["degree"],
            coef0=params["coef0"],
            probability=True,
            class_weight="balanced",
            random_state=seed,
        )),
    ])


def suggest_params(trial: optuna.Trial) -> dict:
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly", "sigmoid"])
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "kernel": kernel,
        "degree": 3,
        "gamma": "scale",
        "coef0": 0.0,
    }
    if kernel == "poly":
        params["degree"] = trial.suggest_int("degree", 2, 5)
    if kernel in {"rbf", "poly", "sigmoid"}:
        params["gamma"] = trial.suggest_categorical("gamma", ["scale", "auto"])
    if kernel in {"poly", "sigmoid"}:
        params["coef0"] = trial.suggest_float("coef0", 0.0, 10.0)
    return params


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    false_positive, true_positive, thresholds = roc_curve(labels, probabilities)
    valid = np.isfinite(thresholds)
    objective = np.where(valid, true_positive - false_positive, -np.inf)
    return float(thresholds[int(np.argmax(objective))])


def load_inputs(features_path: Path, splits_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    features = pd.read_csv(features_path)
    splits = pd.read_csv(splits_path)
    required_features = {"sample_id", "label"}
    required_splits = {"sample_id", "base_model", "role"}
    if not required_features.issubset(features.columns):
        raise ValueError(f"Feature CSV must include columns: {sorted(required_features)}")
    if not required_splits.issubset(splits.columns):
        raise ValueError(f"Split CSV must include columns: {sorted(required_splits)}")
    if features["sample_id"].astype(str).duplicated().any():
        raise ValueError("Feature CSV contains duplicate sample_id values")
    features["sample_id"] = features["sample_id"].astype(str)
    splits["sample_id"] = splits["sample_id"].astype(str)
    feature_columns = [column for column in features.columns if column not in {"sample_id", "label"}]
    if not feature_columns:
        raise ValueError("No numeric feature columns found")
    features[feature_columns] = features[feature_columns].apply(pd.to_numeric, errors="raise")
    features["label"] = pd.to_numeric(features["label"], errors="raise").astype(int)
    if set(features["label"].unique()) != {0, 1}:
        raise ValueError("Training labels must contain both 0 and 1")
    if not np.isfinite(features[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Feature values contain NaN or infinite values")
    unknown = sorted(set(splits["sample_id"]) - set(features["sample_id"]))
    if unknown:
        raise ValueError(f"Split manifest IDs missing from feature CSV: {unknown[:10]}")
    return features, splits, feature_columns


def train(args: argparse.Namespace) -> Path:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    features, splits, feature_columns = load_inputs(args.features, args.splits)
    indexed = features.set_index("sample_id")
    models = []
    thresholds = []
    params_list = []

    for base in range(1, 6):
        fold = splits.loc[splits["base_model"] == base]
        train_ids = fold.loc[fold["role"].str.lower() == "train", "sample_id"].tolist()
        validation_ids = fold.loc[fold["role"].str.lower() == "validation", "sample_id"].tolist()
        if not train_ids or not validation_ids:
            raise ValueError(f"Base model {base} needs Train and Validation rows")
        train_frame = indexed.loc[train_ids]
        validation_frame = indexed.loc[validation_ids]
        x_train = train_frame[feature_columns].to_numpy(dtype=np.float64)
        y_train = train_frame["label"].to_numpy(dtype=int)
        x_validation = validation_frame[feature_columns].to_numpy(dtype=np.float64)
        y_validation = validation_frame["label"].to_numpy(dtype=int)
        if set(y_train) != {0, 1} or set(y_validation) != {0, 1}:
            raise ValueError(f"Base model {base} train and validation splits must each contain both labels")

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

        def objective(trial: optuna.Trial) -> float:
            params = suggest_params(trial)
            scores = []
            for train_index, validation_index in cv.split(x_train, y_train):
                model = create_model(params, SEED)
                model.fit(x_train[train_index], y_train[train_index])
                probabilities = model.predict_proba(x_train[validation_index])[:, 1]
                scores.append(roc_auc_score(y_train[validation_index], probabilities))
            return float(np.mean(scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED + base - 1),
        )
        study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
        params = suggest_best_params(study.best_params)
        model = create_model(params, SEED)
        model.fit(x_train, y_train)
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        threshold = select_threshold(y_validation, validation_probabilities)
        params_list.append({**params, "inference_threshold": threshold, "cv_best_auroc": float(study.best_value)})
        models.append(model)
        thresholds.append(threshold)
        print(f"Base {base}/5: CV AUROC={study.best_value:.4f}; threshold={threshold:.6f}")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "models": models,
        "thresholds": thresholds,
        "model_params_list": params_list,
        "scheme": "B",
        "feature": args.feature,
        "model_type": "svm",
        "objective": "3-fold cross-validation AUROC only",
        "random_seed": SEED,
        "feature_columns": feature_columns,
    }
    joblib.dump(bundle, output / "model_bundle.joblib")
    metadata = {
        "feature": args.feature,
        "model_type": "SVM",
        "base_models": len(models),
        "feature_dimension": len(feature_columns),
        "optuna_trials_per_base_model": args.trials,
        "random_seed": SEED,
        "thresholds": thresholds,
        "feature_columns": feature_columns,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output


def suggest_best_params(best_params: dict) -> dict:
    kernel = best_params["kernel"]
    return {
        "C": float(best_params["C"]),
        "kernel": kernel,
        "gamma": best_params.get("gamma", "scale"),
        "degree": int(best_params.get("degree", 3)),
        "coef0": float(best_params.get("coef0", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the five-fold SVM threshold ensemble")
    parser.add_argument("--feature", choices=("explicit", "implicit"), required=True)
    parser.add_argument("--features", type=Path, required=True, help="CSV with sample_id, label, and numeric feature columns")
    parser.add_argument("--splits", type=Path, required=True, help="CSV with sample_id, base_model, and role columns")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--trials", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output is None:
        args.output = Path(__file__).resolve().parents[1] / "mode" / args.feature / "SVM"
    train(args)
