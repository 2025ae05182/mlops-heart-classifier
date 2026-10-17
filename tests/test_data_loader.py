# tests/test_data_loader.py

import os
import pandas as pd
import pytest
import logging
from src.data_loader import load_data
from src.data_loader import data_preprocessing

def test_load_data_library():
    """Test loading dataset via ucimlrepo library."""
    try:
        df = load_data()
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Check that target column exists after renaming
        assert "target" in df.columns or "num" in df.columns
    except ImportError:
        pytest.skip("ucimlrepo not installed, skipping library test")


def test_preprocessing_with_object_column(caplog):
    """DataFrame with object column shall not be processed and returned None."""
    df = pd.DataFrame({
        "sex": ["Male", "Female", "Male"],
        "age": [45, 55, 65]
    })
    print(df.info())
    with caplog.at_level(logging.INFO):
        df_processed = data_preprocessing(df)

    # Function should return None
    assert df_processed is None

    # Logging should mention categorical columns
    assert "Found non-numeric columns" in caplog.text


def test_preprocessing_with_numeric_only(caplog):
    """DataFrame with only numeric columns should be returned unchanged."""
    df = pd.DataFrame({
        "age": [30, 40, 50],
        "cholesterol": [200, 220, 180]
    })

    with caplog.at_level(logging.INFO):
        df_processed = data_preprocessing(df)

    # Should return same DataFrame
    pd.testing.assert_frame_equal(df, df_processed)
    # Logging should mention dataset has only numerical data
    assert "dataset has only numerical data" in caplog.text
