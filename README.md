## Heart Disease Prediction
### 📖 Project Overview
This project explores machine learning models for predicting heart disease.
We compare Logistic Regression (linear baseline) and Random Forest (non-linear ensemble) using cross-validation, hyperparameter tuning, and MLflow experiment tracking.

heart-disease-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── heart.csv
├── notebooks/
│   ├── baseline_models.ipynb
│   └── tuning_experiments.ipynb
├── src/
│   ├── model.py
│   ├── utils.py
│   └── __init__.py
├── models/
│   ├── log_reg_model.pkl
│   └── rf_model.pkl
├── mlruns/        # MLflow file store (optional)
└── mlflow.db      # SQLite backend (optional)

### Setup
pip install -r requirements.txt

### Running Experiments
python src/model.py

### Launch MLFlow UI
mlflow ui

* sqlite backend
mlflow ui --backend-store-uri sqlite:///mlflow.db

### Model Selection & Tuning
Logistic Regression: tuned C, penalty, solver.

Random Forest: tuned n_estimators, max_depth, max_features.

Cross-validation metrics: Accuracy, Precision, Recall, F1, ROC-AUC.

Final models trained on full training set and evaluated on held-out test set.

### 📊 Results
Logistic Regression: ROC-AUC ≈ 0.85 (best tuned model).

Random Forest: ROC-AUC ≈ 0.88 (best tuned model).

MLflow UI provides full comparison of metrics, parameters, and artifacts.

### 🔁 Reproducibility
Both mlruns/ and mlflow.db are included for reproducibility.

Clone the repo, run mlflow ui, and you’ll see the same experiment history.

Models are saved as .pkl files in models/ for direct loading.

### 📌 Notes
For assignments, committing mlruns/ and/or mlflow.db is fine.

For production projects, use a central MLflow server with PostgreSQL/MySQL and cloud artifact storage.