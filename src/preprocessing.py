import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler

def data_check_numeric(df: pd.DataFrame) -> pd.DataFrame | None:
    # Check if all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        logging.info("Found non-numeric columns. Not expected in ucimlrepo")
        return None
    else:
        logging.info("Loaded dataset has only numerical data. No further processing implemented")
        return df

def data_scaling(feature: pd.Series, transform: str = None) -> pd.Series:
    # Make sure it's a Series
    if not isinstance(feature, pd.Series):
        raise ValueError("Pass a single column (pd.Series) to this function")

    # Optional log transform
    if transform == "log1p":
        values = np.log1p(feature.values.reshape(-1, 1))
    else:
        values = feature.values.reshape(-1, 1)

    # Scale
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values).ravel()

    # Return as DataFrame with a new column name
    # Return as Series with proper name
    return pd.Series(scaled_values, index=feature.index, name=f"{feature.name}_scaled")

def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame | None:
    df_clean = data_check_numeric(df)
    if df_clean is not None:
        for col, transform in [("chol", "log1p"), ("thalach", "log1p"), ("trestbps", None)]:
            df_clean[f"{col}_scaled"] = data_scaling(df_clean[col], transform=transform)
        logging.info("Selected features scaled")
        return df_clean
    else:
        logging.error("Invalid data")
        return None
    


if __name__ == "__main__":

    from data_loader import load_data

    df = load_data()
    df = data_preprocessing(df)
    if df is not None:
        print(df.describe())
    else:
        print("Bad dataset")
