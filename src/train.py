
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import itertools

import mlflow.sklearn


import mlflow
import joblib
from sklearn.ensemble import RandomForestClassifier

# Get best run
runs_df = mlflow.search_runs(experiment_ids=["1"], output_format="pandas")
best_run = runs_df.sort_values(by="metrics.cv_roc_auc", ascending=False).iloc[0]

# Extract params
n_estimators = int(best_run["params.n_estimators"])
max_depth = None if best_run["params.max_depth"] == "None" else int(best_run["params.max_depth"])
max_features = best_run["params.max_features"]

# Train final model
rf = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    max_features=max_features,
    random_state=42
)
rf.fit(X_train, y_train)

# Save model
joblib.dump(rf, "../models/rf_model.pkl")
