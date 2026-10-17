import logging
import pandas as pd
from src.data_loader import load_data
from src.preprocessing import data_preprocessing
from src.feature_engineering import age_to_bins, simplify_target

# Configure logging once at the pipeline level
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def build_dataset() -> pd.DataFrame:

    logging.info("Starting data pipeline...")

    # Step 1: Load
    df = load_data()
    if df is None or df.empty:
        logging.error("Loaded dataset is empty or None")
        raise ValueError("No data loaded")

    logging.info("Data loaded successfully: shape=%s", df.shape)

    # Step 2: Preprocess
    df = data_preprocessing(df)
    if df is None:
        logging.error("Preprocessing failed: non-numeric data detected")
        raise ValueError("Preprocessing returned None due to invalid data")

    logging.info("Preprocessing complete: shape=%s", df.shape)

    # Step 3: Feature engineering
    df = age_to_bins(df)
    df = simplify_target(df)

    logging.info("Feature engineering complete: shape=%s", df.shape)
    logging.info("Data pipeline finished successfully")

    return df

if __name__ == "__main__":
    dataset = build_dataset()
    print(dataset.head())
