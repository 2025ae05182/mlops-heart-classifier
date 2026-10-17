import pandas as pd
import logging
import numpy as np
from sklearn.preprocessing import StandardScaler

def data_check_numeric(df: pd.DataFrame) -> pd.DataFrame | None:
    # Try converting all columns to numeric
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    # If any NaN introduced, data is invalid
    if df_numeric.isnull().any().any():
        logging.error("Found non-numeric values in DataFrame")
        return None

    logging.info("Loaded dataset has only numerical data")
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
        print("Invalid data .. force error")
        return None
    


if __name__ == "__main__":

    from data_loader import load_data

    df = load_data()
    df = data_preprocessing(df)
    if df is not None:
        print(df.describe())
    else:
        print("Bad dataset")
