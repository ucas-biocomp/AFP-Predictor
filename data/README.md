# Data

Each file has two columns, `id,sequence`, one protein per row, with no header
row.

| File | Role | Rows |
| --- | --- | ---: |
| `positive_153.csv` | Training positives (AFP) | 153 |
| `negative_653.csv` | Training negatives (non-AFP) | 653 |
| `test_positive_164.csv` | External test positives | 164 |
| `test_negative_164.csv` | External test negatives | 164 |
| `test_positive_6.csv` | Protein-level test positives (subset of `test_positive_164.csv`) | 6 |

With pandas, pass the column names explicitly:
`pd.read_csv(path, header=None, names=["id", "sequence"])`.
