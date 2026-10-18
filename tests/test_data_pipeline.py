# tests/test_data_pipeline.py
import pandas as pd
import pytest
from src.data_pipeline import build_dataset

def test_build_dataset_success(monkeypatch):
    # Arrange: patch load_data to return a small valid DataFrame
    sample_df = pd.DataFrame({
        "age": [25, 35, 45, 65, 75],
        "chol": [200, 220, 180, 240, 210],
        "thalach": [150, 160, 170, 180, 175],
        "trestbps": [120, 130, 140, 150, 135],
        "target": [0, 1, 2, 1, 0]
    })

    monkeypatch.setattr("src.data_pipeline.load_data", lambda: sample_df)

    # Act
    result = build_dataset()

    # Assert: pipeline adds scaled + engineered features
    assert "chol_scaled" in result.columns
    assert "thalach_scaled" in result.columns
    assert "trestbps_scaled" in result.columns
    assert "age_bin" in result.columns
    assert "target_binary" in result.columns

    # Assert: dummy columns are numeric 0/1 (exclude age_bin itself)
    dummy_cols = [c for c in result.columns if c.startswith("age_") and c != "age_bin"]
    for col in dummy_cols:
        assert set(result[col].unique()) <= {0, 1}


def test_build_dataset_invalid(monkeypatch):
    bad_df = pd.DataFrame({
        "age": ["twenty", "thirty"],
        "chol": ["high", "low"],
        "thalach": ["fast", "slow"],
        "trestbps": ["normal", "high"],
        "target": [0, 1]
    })

    monkeypatch.setattr("src.data_pipeline.load_data", lambda: bad_df)
    
    with pytest.raises(ValueError) as excinfo:
        build_dataset()

    assert "Preprocessing returned None due to invalid data" in str(excinfo.value)
