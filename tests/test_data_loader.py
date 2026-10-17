# tests/test_data_loader.py

import os
import pandas as pd
import pytest
from src.data_loader import load_data

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