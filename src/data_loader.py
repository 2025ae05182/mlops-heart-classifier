
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)


def load_data() -> pd.DataFrame | None:
    """
    Load UCI Heart Disease Cleveland dataset.

    Args:
        None, "Loads from UCIMLREPO, dataset=45 (Heart disease)"

    Returns:
        pd.DataFrame if successful, else None
    """
    try:
        from ucimlrepo import fetch_ucirepo
        # Heart Disease dataset ID in UCI ML repo is 45
        heart_disease = fetch_ucirepo(id=45)

        if heart_disease.data.original.empty:
            logging.info("Dataset is empty. Loading failed")
            return None
        else:
            if "num" in heart_disease.data.original.columns:
                heart_disease.data.original = heart_disease.data.original.rename(columns={'num':'target'})
                df = heart_disease.data.original
                logging.info(f"Heart Disease dataset loaded. Shape: {df.shape}")
                df = df.dropna()
                logging.info(f"Heart Disease dataset cleaned. Shape: {df.shape}")
                return df
    
    except ImportError:
        logging.error("ucimlrepo not installed. Run pip install ucimlrepo ")
        return None

    except Exception as e:
        logging.error(f"Unexpected error while loading dataset: {e}")
        return None


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame | None:
    # Check if all columns are numeric
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in df.dtypes):
        logging.info("Found non-numeric columns. Not expected in ucimlrepo")
        return None
    else:
        logging.info("Loaded dataset has only numerical data. No further processing implemented")
        return df

if __name__ == "__main__":
    # Example usage:
    # call load_data()
    df = load_data()
    if df is not None:
        df = data_preprocessing(df=df)
        if df is not None:
            print(df.info())
        else:
            print("Dataset could not be cleaned. Check logs for errors")
    else:
        print("Dataset could not be loaded. Check logs for errors")
