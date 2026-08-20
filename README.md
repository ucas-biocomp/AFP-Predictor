# AFP-Predictor

Ensemble of 5 MLP base models trained on ESM2 embeddings for antifreeze protein (AFP) prediction.

## Method

- Features: ESM2 `mean_representations`, layer 33
- Base models: 5 MLP classifiers, each on its own fold split
- Ensemble: probability averaging
- Hyperparameters: tuned with Optuna
- Evaluation: internal test set (28+28) and external test set (164+164)

## Files

- `code/ESM_train_model_MLP.ipynb` — training + evaluation notebook
- `model/mlp_ensemble_models.pkl` — trained model bundle

## Requirements

Install with `pip install -r requirements.txt`.

## ESM2 Embeddings

Input features are ESM-2 (`esm2_t33_650M_UR50D`) `mean_representations` (layer 33). ESM-2 is **not included** in this repository — download it yourself from the official [ESM repository](https://github.com/facebookresearch/esm), then extract embeddings with `extract.py`. Example:

```bash
python /path/to/esm-main/scripts/extract.py esm2_t33_650M_UR50D \
    /path/to/your_proteins.fasta \
    /path/to/output_embedding_dir \
    --repr_layers 0 32 33 --include mean
```

This writes `.pt` files containing `mean_representations`, which the notebook loads.
