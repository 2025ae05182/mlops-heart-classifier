
import os
import pandas as pd



def load_data(source: str = "library", path: str = None) -> pd.DataFrame:
    """
    Load UCI Heart Disease Cleveland dataset.

    Args:
        source (str): "library" or "raw"
            - "library": load using ucimlrepo (or similar library)
            - "raw": load from local CSV file (downloaded manually)
            - "path": Path to raw CSV file (required if source="raw")

    Returns:
        pd.DataFrame: Loaded dataset
    """
    if source == "library":
        try:
            from ucimlrepo import fetch_ucirepo
            # Heart Disease dataset ID in UCI ML repo is 45
            heart_disease = fetch_ucirepo(id=45)
            heart_disease.data.original = heart_disease.data.original.rename(columns={'num':'target'})
            df = heart_disease.data.original    

        except ImportError:
            raise ImportError("ucimlrepo not installed. Run pip install ucimlrepo")

    elif source == "raw":
        print("Loading from raw CSV files is not implemented. Set source=library")

    return df


if __name__ == "__main__":
    # Example usage:
    # Option 1: Load via library
    df = load_data(source="library")
    print(df.head())
