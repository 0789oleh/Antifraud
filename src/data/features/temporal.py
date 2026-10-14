"""Временные фичи из TransactionDT.

TransactionDT — секунды от старта (86400 = 1 день).
Нужно: hour, day_of_week, is_night.
"""
import numpy as np
import pandas as pd


def add_temporal_features(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    start_offset: int = 86400,
) -> pd.DataFrame:
    """
    Добавить hour, day_of_week, is_night.

    Args:
        df: датафрейм с time_col
        time_col: колонка со временем (секунды)
        start_offset: сдвиг (первый день = 86400)
    """
    seconds = df[time_col] - start_offset
    seconds = seconds.clip(lower=0)

    df = df.copy()
    df["hour"] = ((seconds % 86400) // 3600).astype("int8")
    df["day_of_week"] = ((seconds // 86400) % 7).astype("int8")
    df["is_night"] = (df["hour"] < 6).astype("int8")

    return df
