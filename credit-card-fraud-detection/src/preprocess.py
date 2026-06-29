import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def scale_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = StandardScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def load_and_prepare(path: str):
    df = load_data(path)
    df = scale_features(df)
    return split_data(df)
