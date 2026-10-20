# Heart Disease Classification

This project explores binary classification of the UCI Heart Disease Cleveland
dataset. It includes exploratory analysis, a reusable data-preparation pipeline,
hyperparameter experiments tracked with MLflow, final model training, and a
small FastAPI prediction endpoint.

The experiments compare Logistic Regression with a Random Forest. The target is
converted to a binary label: `0` means no disease, and any nonzero original
target value means disease is present. This is an educational project, not a
medical device; its outputs are not medical advice.

## Project layout

```text
.
├── notebooks/
│   ├── 01_eda.ipynb           # Dataset acquisition, exploration, and preparation
│   └── 02_model_dev.ipynb     # Notebook-based model development
├── src/
│   ├── app.py                 # FastAPI prediction endpoint
│   ├── data_loader.py         # Fetch and clean the UCI dataset
│   ├── data_pipeline.py       # Load, preprocess, and engineer features
│   ├── feature_engineering.py # Age bins and binary target
│   ├── mlflow_config.py       # Local MLflow tracking configuration
│   ├── preprocessing.py      # Numeric checks and feature scaling
│   ├── train.py               # Select best runs, evaluate, and save models
│   └── tune.py                # Cross-validated parameter experiments
├── tests/                     # Pytest coverage for data and model workflows
├── data/                      # Notebook output (df_final.csv)
├── models/                    # Serialized model files used for inference
├── mlflow_tracking/           # Local MLflow SQLite database and artifacts
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.12 (the version used by the GitHub Actions workflow) and run
commands from the repository root.

```bash
python -m venv .venv
```

Activate the environment (`.venv\Scripts\Activate.ps1` in Windows PowerShell,
or `source .venv/bin/activate` on macOS/Linux), then install dependencies:

```bash
python -m pip install -r requirements.txt
```

The data loader fetches dataset ID 45 from the UCI Machine Learning Repository
using `ucimlrepo`; running the pipeline or its integration test therefore
requires network access.

## Data and feature preparation

`src.data_pipeline.build_dataset()` loads the UCI Cleveland data, renames its
`num` target column to `target`, removes rows with missing values, and rejects
non-numeric columns. Preprocessing adds standardized versions of `chol` and
`thalach` after a `log1p` transform, and standardized `trestbps`. Feature
engineering adds one-hot encoded age bins and `target_binary` (zero versus
nonzero original target). The original columns are retained alongside the
engineered columns.

## Workflow

### 1. Explore and prepare data (notebooks)

Run `notebooks/01_eda.ipynb` to fetch and inspect the dataset, review missing
values and feature distributions, and explore the target and feature
correlations. The notebook prepares the binary target, age groups, and scaled
features, then writes `data/df_final.csv`.

Run `notebooks/02_model_dev.ipynb` after the first notebook for its
CSV-based model-development workflow. The Python pipeline in `src/` fetches
the UCI data directly and does not require the notebook-generated CSV.

### 2. Tune models and log experiments

```bash
python -m src.tune
```

This command builds the dataset, creates a stratified 80/20 train/test split
with random state 42, and runs five-fold stratified cross-validation for each
parameter combination. The search covers:

- **Logistic Regression:** `C` = 0.01, 0.1, 1, or 10; solver = `liblinear`
  or `saga`; `l1_ratio` = 0.
- **Random Forest:** 100 or 200 trees; `max_depth` = unlimited, 5, or 10;
  `max_features` = `sqrt` or `log2`.

The default experiment is named `heart-disease-model-selection`. MLflow logs
the parameters, mean cross-validation ROC-AUC, and held-out test accuracy,
precision, recall, F1, and ROC-AUC. The SQLite tracking database and artifacts
are stored under `mlflow_tracking/`; Random Forest tuning also logs a model
artifact.

### 3. Select and train final models

After tuning has produced runs for both model types, run:

```bash
python -m src.train
```

The training script selects each model's best run by cross-validation ROC-AUC
(breaking ties by shorter run duration), evaluates the selected configuration,
and saves models to `models/lr_model.pkl` and `models/rf_model.pkl`.
The current training functions cast Logistic Regression `C` and Random Forest
`max_depth` values to integers. If the selected runs use `C` below 1 or an
unlimited (`None`) depth, final training can fail; inspect the selected MLflow
parameters if that happens.

### 4. View experiment results

```bash
mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db
```

Open the URL printed by MLflow (normally `http://127.0.0.1:5000`) to compare
runs, parameters, metrics, and logged artifacts.

## Prediction API

The API loads `models/lr_model.pkl` when `src.app` is imported, so that model
file must exist before starting the service. Start it from the repository root:

```bash
uvicorn src.app:app --reload
```

Send a JSON array of 13 numeric values in this exact order:

1. `age`
2. `sex`
3. `cp`
4. `trestbps`
5. `chol`
6. `fbs`
7. `restecg`
8. `thalach`
9. `exang`
10. `oldpeak`
11. `slope`
12. `ca`
13. `thal`

For example, the following is a request shape (replace the sample values with
valid encoded measurements):

```json
{
  "features": [55, 1, 0, 140, 250, 0, 1, 150, 0, 1.2, 1, 0, 2]
}
```

Send it as a `POST` to `http://127.0.0.1:8000/predict`. A successful response
contains an integer `prediction` and a `confidence` value. Interactive API
documentation is available at `http://127.0.0.1:8000/docs`.

**Inference limitation:** preprocessing currently fits the feature scalers on
the incoming dataframe. Since the API handles one record at a time, scaled
values can differ from those used during batch training. Treat this endpoint
as a demonstration; consistent training-time and inference-time preprocessing
is needed before relying on predictions.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest tests/
```

The tests cover dataset validation, feature scaling and engineering, pipeline
behavior, training input validation, and MLflow tuning runs. The data-loader
test contacts the UCI repository and needs network access. GitHub Actions runs
the suite on Python 3.12 and publishes test reports.
