import pandas as pd
import numpy as np


def add_amount_log(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Amount_log"] = np.log1p(df["Amount"].clip(lower=0))
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Time is seconds since first transaction in the dataset
    df["Hour"] = (df["Time"] // 3600) % 24
    df["Day"] = (df["Time"] // 86400) % 7
    return df


def add_v_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["V1_V2"] = df["V1"] * df["V2"]
    df["V3_V4"] = df["V3"] * df["V4"]
    df["V1_V3"] = df["V1"] * df["V3"]
    return df


def add_amount_deviation(df: pd.DataFrame) -> pd.DataFrame:
    # Z-score of Amount within each hour — flags unusually large transactions
    df = df.copy()
    hour_mean = df.groupby("Hour")["Amount"].transform("mean")
    hour_std = df.groupby("Hour")["Amount"].transform("std").replace(0, 1)
    df["Amount_z_hour"] = (df["Amount"] - hour_mean) / hour_std
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling transaction count and spend over a 1-hour window, ordered by Time.
    Without card IDs (anonymised dataset) this is population-level, not per-card —
    it still captures fraud clusters that tend to spike in short windows.
    """
    df = df.copy().sort_values("Time").reset_index(drop=True)
    window = 3600  # seconds

    counts, amounts = [], []
    for i, t in enumerate(df["Time"]):
        window_mask = (df["Time"] >= t - window) & (df["Time"] < t)
        counts.append(window_mask.sum())
        amounts.append(df.loc[window_mask, "Amount"].sum())

    df["velocity_count_1h"] = counts
    df["velocity_amount_1h"] = amounts
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_amount_log(df)
    df = add_time_features(df)
    df = add_v_interactions(df)
    df = add_amount_deviation(df)
    df = add_velocity_features(df)
    return df


def compute_hour_stats(df: pd.DataFrame) -> dict:
    """Save per-hour Amount statistics for use at inference time."""
    stats = df.groupby("Hour")["Amount"].agg(["mean", "std"]).rename(
        columns={"mean": "amount_mean", "std": "amount_std"}
    )
    stats["amount_std"] = stats["amount_std"].replace(0, 1)
    return stats.to_dict()


def apply_hour_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Apply precomputed hour stats at inference (avoids leaking test population)."""
    df = df.copy()
    means = pd.Series(stats["amount_mean"])
    stds = pd.Series(stats["amount_std"])
    df["Amount_z_hour"] = (df["Amount"] - df["Hour"].map(means)) / df["Hour"].map(stds)
    return df
