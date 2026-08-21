# AFP-Predictor

Ensemble of 5 MLP base models trained on ESM2 embeddings for antifreeze protein (AFP) prediction.

## Method

- Features: ESM2 `mean_representations`, layer 33
- Base models: 5 MLP classifiers, each on its own fold split
- Ensemble: probability averaging
- Hyperparameters: tuned with Optuna
- Evaluation: internal test set (28+28) and external test set (164+164)

## Project Structure

```text
AFP_Ensemble_MLP_ESM2_20260408_173024/
├── code/
│   ├── ESM_train_model_MLP.ipynb    # training + evaluation notebook
│   ├── predict.py                   # predict AFP/non-AFP from ESM2 embeddings
│   ├── README.md
│   ├── requirements.txt
│   └── LICENSE
├── data/
│   ├── negative_653.csv             # negative set
│   ├── positive_153.csv             # positive set
│   ├── test_negative_164.csv        # test negative set
│   ├── test_positive_164.csv        # test positive set
│   └── test_positive_6.csv          # protein-level test positive set
└── model/
    └── mlp_ensemble_models.pkl      # trained model bundle
```

## Installation

Python 3.10 or 3.11 is recommended.

```bash
git clone https://github.com/ucas-biocomp/AFP-Predictor.git
cd AFP-Predictor
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install torch==2.9.1
```

## Usage

1. Extract ESM2 embeddings for your proteins (see below).
2. Predict:

```bash
python code/predict.py /path/to/embedding.pt    # single file
python code/predict.py /path/to/embedding_dir   # folder of .pt files
```

Prints `AFP` or `non-AFP` for each protein.

## ESM2 Embeddings

Input features are ESM-2 (`esm2_t33_650M_UR50D`) `mean_representations` (layer 33). ESM-2 is **not included** in this repository — download it yourself from the official [ESM repository](https://github.com/facebookresearch/esm), then extract embeddings with `extract.py`. Example:

```bash
python /path/to/esm-main/scripts/extract.py esm2_t33_650M_UR50D \
    /path/to/your_proteins.fasta \
    /path/to/output_embedding_dir \
    --repr_layers 0 32 33 --include mean
```

This writes one `.pt` per protein, ready for `predict.py`.

## Citation

If this repository supports a publication or thesis, cite that work and include its bibliographic details here when publicly available.
