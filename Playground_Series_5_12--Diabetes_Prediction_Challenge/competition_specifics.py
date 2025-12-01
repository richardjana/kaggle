import pandas as pd

TARGET_COL = 'diagnosed_diabetes'
COMPETITION_NAME = 'playground-series-s5e12'
N_FOLDS = 5


def load_and_prepare(file_name: str) -> pd.DataFrame:
    """ Read data from csv file and do some basic preprocessing.
    Args:
        file_name (str): Name of the csv file.
    Returns:
        pd.DataFrame: The created DataFrame.
    """
    df = pd.read_csv(file_name)
    try:  # train and test
        df.drop(columns='id', inplace=True)
    except KeyError:  # original data
        additional_cols = ['diabetes_risk_score', 'diabetes_stage', 'glucose_fasting',
                           'glucose_postprandial', 'hba1c', 'insulin_level']
        df.drop(columns=additional_cols, inplace=True)

    # convert object type columns to category type
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype('category')

    return df
