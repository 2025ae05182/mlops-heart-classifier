# tests/test_data_loader.py

import os
import pandas as pd
import pytest
from src.data_loader import load_data

def test_load_data_library():
    """Test loading dataset via ucimlrepo library."""
    try:
        df = load_data(source="library")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        # Check that target column exists after renaming
        assert "target" in df.columns or "num" in df.columns
    except ImportError:
        pytest.skip("ucimlrepo not installed, skipping library test")

@pytest.mark.xfail(reason="Raw loading not implemented yet")
def test_load_data_raw(tmp_path):
    """Test loading dataset from a raw CSV file."""
    # Create a temporary CSV file
    test_csv = tmp_path / "heart.csv"
    df = pd.DataFrame({"num": [0, 1, 2], "age": [45, 55, 65]})
    df.to_csv(test_csv, index=False)

    # Load using raw mode
    df_loaded = load_data(source="raw", path=str(test_csv))
    assert isinstance(df_loaded, pd.DataFrame)
    assert not df_loaded.empty
    assert "num" in df_loaded.columns
