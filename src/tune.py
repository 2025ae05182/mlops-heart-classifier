import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import itertools
import logging
import pandas as pd
from src.mlflow_config import setup_mlflow
from src.data_pipeline import build_dataset

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# Logistic Regression

def logistic_regression_tuning(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        experiment_name: str) -> str:
    """
    Finetunes Logistic Regression hyperparamters for Heart-Disease dataset
    Expects mlflow experiment name to log the model performance
    Returns the model name string that can be used in mlflow experiment to filter the results
    """

    if X_train.empty or X_test.empty:
        raise ValueError("Training and test feature sets must not be empty.")
    if y_train.empty or y_test.empty:
        raise ValueError("Training and test target sets must not be empty.")
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError("Train and test feature sets must have the same columns.")

    model_name = "LogisticRegression"

    # Define parameter grid
    param_grid_lr = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "saga"],
        "l1_ratio": [0]   # equivalent to penalty="l2"
    }

    # Generate all combinations
    param_combinations = list(itertools.product(
        param_grid_lr["C"],
        param_grid_lr["solver"],
        param_grid_lr["l1_ratio"]
    ))

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logging.info("Hyper-parameter finetuning begins...")
    for C, solver, l1_ratio in param_combinations:
        logging.info("*")
        with mlflow.start_run(run_name=f"LogReg_C{C}_solver{solver}_l1ratio{l1_ratio}"):
            # Define model with current params
            log_reg = LogisticRegression(
                C=C,
                solver=solver,
                l1_ratio=l1_ratio,   # <-- use l1_ratio instead of penalty
                max_iter=5000,
                random_state=42
            )

            # Cross-validation ROC-AUC
            cv_scores = cross_val_score(log_reg, X_train, y_train, cv=cv, scoring="roc_auc")
            mean_cv_score = cv_scores.mean()

            # Log parameters and CV metric
            mlflow.log_params({
                "model_name": "LogisticRegression",
                "C": C,
                "solver": solver,
                "l1_ratio": l1_ratio
            })
            mlflow.log_metric("cv_roc_auc", float(mean_cv_score))

            log_reg.fit(X_train, y_train)
            y_pred = log_reg.predict(X_test)
            y_proba = log_reg.predict_proba(X_test)[:, 1]

            # Log test metrics
            mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, y_pred)))
            mlflow.log_metric("test_precision", float(precision_score(y_test, y_pred)))
            mlflow.log_metric("test_recall", float(recall_score(y_test, y_pred)))
            mlflow.log_metric("test_f1", float(f1_score(y_test, y_pred)))
            mlflow.log_metric("test_roc_auc", float(roc_auc_score(y_test, y_proba)))

    logging.info(f"{model_name} hyper-parameter finetuning is complete")
    logging.info(
    f"Results are logged in MLflow experiment '{experiment_name}' "
    f"under model name '{model_name}'"
    )
    return model_name

def random_forest_tuning(
        X_train: pd.DataFrame, 
        X_test: pd.DataFrame, 
        y_train: pd.Series, 
        y_test: pd.Series,
        experiment_name: str) -> str:

    """
    Random Forest hyper-parameter finetuning for Heart-Disease dataset
    """
    if X_train.empty or X_test.empty:
        raise ValueError("Training and test feature sets must not be empty.")
    if y_train.empty or y_test.empty:
        raise ValueError("Training and test target sets must not be empty.")
    if set(X_train.columns) != set(X_test.columns):
        raise ValueError("Train and test feature sets must have the same columns.")

    model_name = "RandomForest"

    # Define parameter grid manually
    param_grid_rf = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "max_features": ["sqrt", "log2"]
    }

    # Generate all combinations
    param_combinations = list(itertools.product(
        param_grid_rf["n_estimators"],
        param_grid_rf["max_depth"],
        param_grid_rf["max_features"]
    ))

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    logging.info(f"{model_name} hyper-parameter finetuning begins...")
    for n_estimators, max_depth, max_features in param_combinations:
        logging.info("*")
        with mlflow.start_run(run_name=f"RF_n{n_estimators}_d{max_depth}_f{max_features}"):
            # Define model with current params
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                max_features=max_features,
                random_state=42
            )

            # Cross-validation ROC-AUC
            cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring="roc_auc")
            mean_cv_score = cv_scores.mean()

            # Log parameters and CV metric
            mlflow.log_params({
                "model_name": model_name,
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "max_features": max_features
            })
            mlflow.log_metric("cv_roc_auc", float(mean_cv_score))

            # Fit on full training set
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            y_proba = rf.predict_proba(X_test)[:, 1]

            # Log test metrics
            mlflow.log_metric("test_accuracy", float(accuracy_score(y_test, y_pred)))
            mlflow.log_metric("test_precision", float(precision_score(y_test, y_pred)))
            mlflow.log_metric("test_recall", float(recall_score(y_test, y_pred)))
            mlflow.log_metric("test_f1", float(f1_score(y_test, y_pred)))
            mlflow.log_metric("test_roc_auc", float(roc_auc_score(y_test, y_proba)))

            # Log model artifact
            mlflow.sklearn.log_model(
                                        rf,
                                        name="rf_model",
                                        skops_trusted_types=["sklearn.tree._tree.Tree"]
            )

    logging.info(f"{model_name} hyper-parameter finetuning is complete")
    logging.info(
    f"Results are logged in MLflow experiment '{experiment_name}' "
    f"under model name '{model_name}'"
    )

    return model_name


if __name__ == "__main__":
    experiment_name = "heart-disease-model-selection"
    setup_mlflow(experiment_name)
    logging.info("Starting data pipeline...")
    df = build_dataset()

    y = df["target_binary"]
    X = df.drop(["target", "target_binary"], axis=1)

    logging.info("Dataset is loaded")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y   # <-- ensures balanced split
    )
    logging.info("Dataset is split is complete")
    logging.info("Logistic Regression Finetuning begin...")
    lr_model_name = logistic_regression_tuning(
        X_train, X_test, y_train, y_test,
        experiment_name)

    logging.info("Random Forest Finetuning begins...")
    rf_model_name = random_forest_tuning(
        X_train, X_test, y_train, y_test,
        experiment_name)

