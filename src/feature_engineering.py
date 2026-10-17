import pandas as pd
import logging


def age_to_bins(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    # Define bins and labels
    bins = [0, 30, 40, 50, 60, 70, 80, 100]   # bin edges
    labels = [1, 2, 3, 4, 5, 6, 7]            # encoded labels

    df_clean = df.copy()
    # Create new column
    df_clean['age_bin'] = pd.cut(df_clean['age'], bins=bins, labels=labels, right=False)

    # Log distribution
    logging.info("Age bin distribution:\n%s", df_clean['age_bin'].value_counts())

    age_dummies = pd.get_dummies(df_clean['age_bin'], prefix='age')
    df_clean = pd.concat([df_clean, age_dummies], axis=1)
    logging.info("DataFrame after age binning: columns=%s, shape=%s",
                 df_clean.columns.tolist(), df_clean.shape)

    return df_clean

def simplify_target(df_clean: pd.DataFrame) -> pd.DataFrame:
    #Create a new binary target column
    df_clean['target_binary'] = df_clean['target'].apply(lambda x: 0 if x == 0 else 1)
    print(df_clean['target_binary'].value_counts())
    return df_clean