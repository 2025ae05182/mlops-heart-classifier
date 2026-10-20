import pandas as pd
import pytest
from sklearn.datasets import make_classification
from src.tune import logistic_regression_tuning, random_forest_tuning
from src.tune import setup_mlflow

@pytest.fixture
def sample_data():
    # Small synthetic dataset for testing
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["target_binary"] = y
    return df

def test_logistic_regression_tuning_runs(sample_data):
    X = sample_data.drop("target_binary", axis=1)
    y = sample_data["target_binary"]

    # Split small dataset
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # setup dummy mlflow experiment
    setup_mlflow("test-experiment")
    model_name = logistic_regression_tuning(X_train, X_test, y_train, y_test, "test-experiment")
    assert model_name == "LogisticRegression"

def test_random_forest_tuning_runs(sample_data):
    X = sample_data.drop("target_binary", axis=1)
    y = sample_data["target_binary"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # setup dummy mlflow experiment
    setup_mlflow("test-experiment")
    model_name = random_forest_tuning(X_train, X_test, y_train, y_test, "test-experiment")
    assert model_name == "RandomForest"

def test_empty_dataset_raises_error():
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.Series(dtype=int)
    y_test = pd.Series(dtype=int)

    with pytest.raises(ValueError):
        logistic_regression_tuning(X_train, X_test, y_train, y_test, "test-experiment")
