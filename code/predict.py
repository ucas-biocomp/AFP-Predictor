#!/usr/bin/env python3
"""
Minimal AFP predictor: load ESM2 embeddings and predict with the MLP ensemble.

Usage:
    python predict.py /path/to/embedding.pt      # single .pt file
    python predict.py /path/to/embedding_dir     # directory of .pt files

The model bundle is loaded from ../model/mlp_ensemble_models.pkl (relative
to this script). Embeddings must contain "mean_representations" (layer 33),
i.e. the output of ESM2 extract.py with --include mean.
"""

import argparse
import os
import sys

import joblib
import numpy as np
import torch

ESM_LAYER = 33
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "model", "mlp_ensemble_models.pkl")


def patch_numpy_compat():
    """Shim for loading sklearn models pickled under a different numpy version."""
    if "numpy._core" not in sys.modules:
        import numpy.core as _nc
        sys.modules["numpy._core"] = _nc
    try:
        from numpy.random import _pickle

        def _patched(name="MT19937"):
            if isinstance(name, type):
                name = name.__name__
            if name in _pickle.BitGenerators:
                return _pickle.BitGenerators[name]()
            raise ValueError(f"{name} is not a known BitGenerator")

        _pickle.__bit_generator_ctor = _patched
        if hasattr(_pickle, "__generator_ctor"):
            _pickle.__generator_ctor.__defaults__ = ("MT19937", _patched)
        if hasattr(_pickle, "__randomstate_ctor"):
            _pickle.__randomstate_ctor.__defaults__ = ("MT19937", _patched)
    except Exception:
        pass


def load_bundle(model_path=MODEL_PATH):
    patch_numpy_compat()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    return joblib.load(model_path)


def load_embedding(pt_path):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    emb = data["mean_representations"][ESM_LAYER]
    if isinstance(emb, torch.Tensor):
        emb = emb.cpu().numpy()
    label = data.get("label", os.path.basename(pt_path))
    return label, emb


def list_pt_files(path):
    if os.path.isfile(path):
        return [path]
    return sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pt"))


def predict_embedding(embedding, bundle):
    models = bundle["models"]
    thresholds = [p["inference_threshold"] for p in bundle["model_params_list"]]
    avg_threshold = float(np.mean(thresholds))

    X = embedding.reshape(1, -1)
    proba_list = [float(m.predict_proba(X)[:, 1][0]) for m in models]
    avg_proba = float(np.mean(proba_list))
    return avg_proba >= avg_threshold


def main():
    parser = argparse.ArgumentParser(description="Minimal AFP predictor (ESM2 + MLP ensemble)")
    parser.add_argument("embedding", help="Path to a .pt file or a directory of .pt files")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model .pkl")
    args = parser.parse_args()

    pt_files = list_pt_files(args.embedding)
    if not pt_files:
        parser.error(f"No .pt files found in {args.embedding}")

    bundle = load_bundle(args.model)

    for pt in pt_files:
        _, emb = load_embedding(pt)
        print("AFP" if predict_embedding(emb, bundle) else "non-AFP")


if __name__ == "__main__":
    main()
