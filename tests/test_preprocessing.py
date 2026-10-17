# tests/test_data_loader.py

import os
import pandas as pd
import pytest
import logging
from src.data_loader import load_data
from src.preprocessing import data_check_numeric
from src.preprocessing import data_preprocessing

def test_preprocessing_with_object_column(caplog):
    """DataFrame with object column shall not be processed and returned None."""
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male"],
        "age": [45, 55, 65]
    })
    print(df.info())
    with caplog.at_level(logging.INFO):
        df_processed = data_check_numeric(df)

    # Function should return None
    assert df_processed is None

    # Logging should mention categorical columns
    assert "Found non-numeric values" in caplog.text


def test_preprocessing_with_numeric_only(caplog):
    """DataFrame with only numeric columns should be returned unchanged."""
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "cholesterol": [200, 220, 180]
    })

    with caplog.at_level(logging.INFO):
        df_processed = data_check_numeric(df)

    # Should return same DataFrame
    pd.testing.assert_frame_equal(df, df_processed)
    # Logging should mention dataset has only numerical data
    assert "dataset has only numerical data" in caplog.text

def test_data_preprocessing_scaling():
    # Create a small numeric DataFrame
    df = pd.DataFrame({
        "chol": [200, 220, 180, 240],
        "thalach": [150, 160, 170, 180],
        "trestbps": [120, 130, 140, 150]
    })

    # Run preprocessing
    result = data_preprocessing(df)

    # Check that scaled columns exist
    assert "chol_scaled" in result.columns
    assert "thalach_scaled" in result.columns
    assert "trestbps_scaled" in result.columns

    # Check that scaled columns have same length as input
    assert len(result["chol_scaled"]) == len(df)

    # Check that scaled values are numeric
    assert pd.api.types.is_numeric_dtype(result["chol_scaled"])
    assert pd.api.types.is_numeric_dtype(result["thalach_scaled"])
    assert pd.api.types.is_numeric_dtype(result["trestbps_scaled"])
