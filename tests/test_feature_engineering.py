import pandas as pd
from src.feature_engineering import age_to_bins, simplify_target

def test_age_to_bins_creates_bins_and_dummies():
    # Arrange
    df = pd.DataFrame({"age": [25, 35, 45, 65, 75]})

    # Act
    result = age_to_bins(df)

    # Assert: new column exists
    assert "age_bin" in result.columns

    # Assert: dummy columns exist
    dummy_cols = [col for col in result.columns if col.startswith("age_")]
    assert len(dummy_cols) > 0

    # Assert: bins are assigned correctly
    assert set(result["age_bin"].unique()) <= {1, 2, 3, 4, 5, 6, 7}


def test_simplify_target_creates_binary_column():
    # Arrange
    df = pd.DataFrame({"target": [0, 1, 2, 3, 0, 1]})

    # Act
    result = simplify_target(df)

    # Assert: new column exists
    assert "target_binary" in result.columns

    # Assert: values are only 0 or 1
    assert set(result["target_binary"].unique()) <= {0, 1}

    # Assert: mapping works (0 stays 0, non-zero becomes 1)
    assert result.loc[result["target"] == 0, "target_binary"].eq(0).all()
    assert result.loc[result["target"] != 0, "target_binary"].eq(1).all()
