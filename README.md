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
