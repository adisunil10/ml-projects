import pandas as pd
import numpy as np


def add_amount_log(df: pd.DataFrame) -> pd.DataFrame:
    """Log-transform of Amount reduces right-skew."""
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour-of-day and day-of-week from the raw Time column (seconds)."""
    df = df.copy()
    # Time is seconds elapsed from first transaction in dataset
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Day"] = (df["Time"] // 86400) % 7
    return df


def add_v_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PCA features V1–V28 are orthogonal, but a few pairwise products capture
    non-linear relationships that tree models benefit from explicitly.
    """
    df = df.copy()
    df["V1_V2"] = df["V1"] * df["V2"]
    df["V3_V4"] = df["V3"] * df["V4"]
    df["V1_V3"] = df["V1"] * df["V3"]
    return df


def add_amount_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """Deviation of Amount from the per-hour mean (proxy for behavioural anomaly)."""
    df = df.copy()
    hour_mean = df.groupby("Hour")["Amount"].transform("mean")
    hour_std = df.groupby("Hour")["Amount"].transform("std").replace(0, 1)
    df["Amount_z_hour"] = (df["Amount"] - hour_mean) / hour_std
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_amount_log(df)
    df = add_time_features(df)
    df = add_v_interaction_features(df)
    df = add_amount_deviation(df)
    return df
