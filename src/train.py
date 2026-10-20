
import mlflow
from mlflow.tracking import MlflowClient
from pathlib import Path

import logging
import pandas as pd
import logging

import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import recall_score, f1_score
from sklearn.metrics import roc_auc_score, confusion_matrix
import os
import joblib
from pathlib import Path


def retrieve_best_run(experiment_name: str, 
                      model_name: str) -> pd.Series:
    """
    Retrieve the best hyper parameter set from the mlflow experiment
    """
    logging.info(f"Experiment retrieval in progress for: {experiment_name}")

    # Always ensure tracking URI is set
    mlflow.set_tracking_uri("sqlite:///./mlflow_tracking/mlflow.db")

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    logging.info("Loading Experiment runs")
    # Filter runs by model_name tag or param
    runs_df = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"params.model_name = '{model_name}'",   # or params.model_name
        #filter_string = "",
        run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
        output_format="pandas"
    )

    if runs_df.empty:
        raise ValueError(f"No runs found for model '{model_name}' in experiment '{experiment_name}'.")
    
    # Ensure start_time and end_time are datetime objects
    runs_df["start_time"] = pd.to_datetime(runs_df["start_time"])
    runs_df["end_time"]   = pd.to_datetime(runs_df["end_time"])

    # Calculate duration in seconds
    runs_df["duration_sec"] = (runs_df["end_time"] - runs_df["start_time"]).dt.total_seconds()

    # Sort by best ROC-AUC and shortest duration
    best_run = runs_df.sort_values(
        by=["metrics.cv_roc_auc", "duration_sec"],
        ascending=[False, True]
    ).iloc[0]

    logging.info("Best run is extracted")
    return best_run

def train_lr(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series,  
        best_run_lr: pd.Series) -> str:
    """
    Logistic Regression model trained on Heart-Disease Cleveland dataset
    The Hyper-parameters considered are:
    - C,
    - solver
    - l1_ratio
    """

    if X_train.empty or X_test.empty:
            raise ValueError("Training and test feature sets must not be empty.")
    if y_train.empty or y_test.empty:
        raise ValueError("Training and test target sets must not be empty.")
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError("Train and test feature sets must have the same columns.")
    
    if best_run_lr.empty:
             raise ValueError("Parameters are empty")

    logging.info("Logistic Regression training: Input validation done!")  
    
    # Define model
    start_time = time.time()
    log_reg = LogisticRegression(
                C=int(best_run_lr["params.C"]),
                solver=best_run_lr["params.solver"],
                l1_ratio=int(best_run_lr["params.l1_ratio"]),   # <-- use l1_ratio instead of penalty
                max_iter=1000,
                random_state=42
            )

    # Step 1: Cross-validation with multiple metrics
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(log_reg, X_train, y_train, cv=5, scoring=scoring)

    # Print mean CV scores
    logging.info("Logistic Regression CV Results:")
    for metric in scoring:
        mean_score = cv_results[f"test_{metric}"].mean()
        logging.info(f"  {metric}: {mean_score:.3f}")

    # Step 2: Fit final model on full training set
    log_reg.fit(X_train, y_train)

    # Step 3: Evaluate on held-out test set
    y_pred = log_reg.predict(X_test)
    y_proba = log_reg.predict_proba(X_test)[:,1]
    runtime = time.time() - start_time
    logging.info("\nLogistic Regression Test Results:")
    logging.info("  Accuracy:", accuracy_score(y_test, y_pred))
    logging.info("  Precision:", precision_score(y_test, y_pred))
    logging.info("  Recall:", recall_score(y_test, y_pred))
    logging.info("  F1-score:", f1_score(y_test, y_pred))
    logging.info("  ROC-AUC:", roc_auc_score(y_test, y_proba))
    logging.info("  Runtime:", runtime)
    logging.info("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
   
    model_path = "./models/lr_model.pkl"

    os.makedirs("./models", exist_ok=True)
    joblib.dump(log_reg, model_path)

    logging.info(f"Trained model is available in {model_path}")
    return model_path

def train_rf(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series,  
        best_run_rf: pd.Series) -> str:

    """
    Random Forest training on Heart Disease - Cleveland dataset
    Hyper-parameters considered are:
    -    n_estimators
    -   max_depth
    -   max_features
    """
    
    if X_train.empty or X_test.empty:
        raise ValueError("Training and test feature sets must not be empty.")
    if y_train.empty or y_test.empty:
        raise ValueError("Training and test target sets must not be empty.")
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError("Train and test feature sets must have the same columns.")

    if best_run_rf.empty:
         raise ValueError("Parameters are empty")

    logging.info("Random Forest training: Input validation done!")

    # Run CV for evaluation
    from sklearn.ensemble import RandomForestClassifier
    start_time = time.time()
    rf = RandomForestClassifier(
                n_estimators=int(best_run_rf["params.n_estimators"]),
                max_depth=int(best_run_rf["params.max_depth"]),
                max_features=best_run_rf["params.max_features"],
                random_state=42
            )
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    cv_results = cross_validate(rf, X_train, y_train, cv=5, scoring=scoring)

    for metric in scoring:
        logging.info(f"CV {metric}: {cv_results[f'test_{metric}'].mean():.3f}")

    # Fit final model on full training set
    rf.fit(X_train, y_train)

    # Now you can safely evaluate on the test set
    logging.info("Test Accuracy:", rf.score(X_test, y_test))

    # Step 3: Evaluate on held-out test set
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:,1]
    runtime = time.time() - start_time
    logging.info(" Random Forest Test Results:")
    logging.info("  Accuracy:", accuracy_score(y_test, y_pred))
    logging.info("  Precision:", precision_score(y_test, y_pred))
    logging.info("  Recall:", recall_score(y_test, y_pred))
    logging.info("  F1-score:", f1_score(y_test, y_pred))
    logging.info("  ROC-AUC:", roc_auc_score(y_test, y_proba))
    logging.info("  Runtime:", runtime)
    logging.info("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    model_path = "./models/rf_model.pkl"

    os.makedirs("./models", exist_ok=True)
    joblib.dump(rf, model_path)

    logging.info(f"Trained model is available in {model_path}")
    return model_path

if __name__ == "__main__":

    experiment_name = "heart-disease-model-selection"
    model_name_lr = "LogisticRegression"
    model_name_rf = "RandomForest"

    best_run_lr = retrieve_best_run(experiment_name, model_name_lr)
    best_run_rf = retrieve_best_run(experiment_name, model_name_rf)

    from src.data_pipeline import build_dataset
    from sklearn.model_selection import train_test_split

    df = build_dataset()

    y = df["target_binary"]
    X = df.drop("target_binary", axis=1)

    logging.info("Dataset is loaded")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # <-- ensures balanced split
    )
    
    train_lr(X_train, X_test, y_train, y_test, best_run_lr)
    train_rf(X_train, X_test, y_train, y_test, best_run_rf)


