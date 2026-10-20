import pytest
import pandas as pd
import numpy as np
from src.train import train_lr, train_rf

def make_data():
    X = pd.DataFrame(np.random.rand(10, 3), columns=["a", "b", "c"])
    y = pd.Series(np.random.randint(0, 2, size=10))
    return X, y

def test_train_lr_empty_X_train():
    X, y = make_data()
    with pytest.raises(ValueError, match="must not be empty"):
        train_lr(pd.DataFrame(), X, y, y, pd.Series({"params.C":1,"params.solver":"lbfgs","params.l1_ratio":0}))

def test_train_lr_column_mismatch():
    X, y = make_data()
    X_test = X.rename(columns={"a":"x"})  # mismatch
    with pytest.raises(ValueError, match="same columns"):
        train_lr(X, X_test, y, y, pd.Series({"params.C":1,"params.solver":"lbfgs","params.l1_ratio":0}))

def test_train_lr_empty_params():
    X, y = make_data()
    with pytest.raises(ValueError, match="Parameters are empty"):
        train_lr(X, X, y, y, pd.Series(dtype="object"))
