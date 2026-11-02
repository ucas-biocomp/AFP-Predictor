# AFP-Predictor

Antifreeze protein (AFP) prediction from ESM2 embeddings with an ensemble of
five support vector machine (SVM) classifiers.

## Method

- **Features**: ESM-2 (`esm2_t33_650M_UR50D`) layer-33 mean embedding (1280 dimensions per protein)
- **Base models**: 5 `StandardScaler` + `SVC` pipelines, each trained on its own fold split
- **Ensemble**: arithmetic mean of the five positive-class probabilities
- **Decision**:  `AFP` if score ≥ 0.6722, otherwise `non-AFP` 
- **Hyperparameters**: tuned with Optuna (3-fold cross-validation AUROC), `class_weight="balanced"`, seed 84
- **Evaluation**: external test set of 164 positives + 164 negatives

## Project Structure

```text
AFP-Predictor/
├── code/
│   ├── predict.py                          # predict AFP/non-AFP from ESM2 embeddings
│   └── train.py                            # retrain the five-fold SVM ensemble
├── data/
│   ├── README.md                           # sequence file formats and usage
│   ├── positive_153.csv                    # training positives
│   ├── negative_653.csv                    # training negatives
│   ├── test_positive_164.csv               # external test positives
│   ├── test_negative_164.csv               # external test negatives
│   └── test_positive_6.csv                 # protein-level test positives
├── model/
│   └── SVM_implicit_ensemble_models.joblib # trained model bundle (5 pipelines)
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/ucas-biocomp/AFP-Predictor.git
cd AFP-Predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.9.1
```

## Usage

### Prediction

1. Extract ESM2 embeddings for your proteins (see [ESM2 Embeddings](#esm2-embeddings)).
2. Run prediction on a single .pt file or a directory containing .pt files:

```bash
python code/predict.py /path/to/embedding.pt --feature implicit    # single file
python code/predict.py /path/to/embedding_dir --feature implicit   # folder of .pt files
```

By default, the script prints AFP or non-AFP for each protein. Add --output result.csv to save a CSV containing the protein ID, prediction score, decision threshold, and predicted label:
```bash
python code/predict.py /path/to/embedding_dir --feature implicit --output result.csv
```
CSV columns: id, score, threshold, and prediction.

### Training

```bash
python code/train.py --feature implicit --features features.csv --splits splits.csv
```

Training requires a precomputed feature CSV and a five-fold split manifest. The feature CSV must contain `sample_id`, binary `label` (0 for non-AFP and 1 for AFP), and numeric feature columns. The split manifest must contain `sample_id`, `base_model` (1–5), and `role` (`Train` or `Validation`). IDs must match between the two files.

## Model Detail

Five `StandardScaler` + `SVC(probability=True, class_weight="balanced", random_state=84)`
pipelines. The values below are read from the model file.

| Model | Kernel | C | gamma | degree | coef0 | Support vectors (non-AFP / AFP) |
|---:|:---:|---:|:---:|---:|---:|---:|
| 1 | rbf | 1.955 | auto | 3 | 0 | 60 / 63 |
| 2 | rbf | 1.081 | scale | 3 | 0 | 56 / 58 |
| 3 | poly | 0.05222 | scale | 5 | 1.028 | 68 / 65 |
| 4 | rbf | 10.36 | scale | 3 | 0 | 53 / 56 |
| 5 | poly | 32.21 | scale | 2 | 0.5934 | 51 / 39 |

The bundle (`model/SVM_implicit_ensemble_models.joblib`) is a dict whose
`models` key holds the five pipelines and whose `ensemble_threshold` (0.6722)
is the threshold `predict.py` applies.

## ESM2 Embeddings

ESM-2 is **not included** in this repository. Download it from the official
[ESM repository](https://github.com/facebookresearch/esm) and extract embeddings
with its `scripts/extract.py`:

```bash
python /path/to/esm-main/scripts/extract.py esm2_t33_650M_UR50D \
    /path/to/your_proteins.fasta \
    /path/to/output_embedding_dir \
    --repr_layers 33 --include mean
```

This writes one `.pt` per protein, ready for `predict.py`. Use exactly this
model and layer: other ESM models or layers can also produce 1280-dim vectors,
but they do not match the training features.

## Citation

If you use AFP-R in your research, please cite:

Liu, W., Zhang, Y., Xiu, D., Liu, Y., Wang, T., Chai, X., Qu, H., Min, Y., & Zhang, Z. (2026). AFP-R: An Open Resource Dedicated to Antifreeze Proteins. bioRxiv. https://doi.org/10.64898/2026.08.21.746139
