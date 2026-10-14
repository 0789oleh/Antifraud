"""Time-based split для антифрода.

Random split на временных данных → data leakage:
модель видит будущее и «предсказывает» прошлое.
В проде так не работает. Только temporal.
"""
from dataclasses import dataclass

import pandas as pd


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    @property
    def sizes(self) -> dict:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }


def temporal_split(
    df: pd.DataFrame,
    time_col: str = "TransactionDT",
    val_size: float = 0.15,
    test_size: float = 0.15,
    embargo: int = 0,
) -> TemporalSplit:
    """
    Разбить df по времени: train → val → test.

    Args:
        df: датафрейм (отсортируем сами)
        time_col: колонка со временем
        val_size: доля валидации
        test_size: доля теста
        embargo: секунды, которые выкидываем между сплитами

    Returns:
        TemporalSplit с train/val/test
    """
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    df = df.sort_values(time_col).reset_index(drop=True)
    n = len(df)

    test_start = int(n * (1 - test_size))
    val_start = int(n * (1 - test_size - val_size))

    train = df.iloc[:val_start].copy()
    val = df.iloc[val_start:test_start].copy()
    test = df.iloc[test_start:].copy()

    if embargo > 0:
        val_cutoff = train[time_col].max() - embargo
        test_cutoff = val[time_col].max() - embargo
        train = train[train[time_col] <= val_cutoff]
        val = val[(val[time_col] > val_cutoff) & (val[time_col] <= test_cutoff)]
        test = test[test[time_col] > test_cutoff]

    assert train[time_col].max() < val[time_col].min(), \
        "Leakage: train and val overlap in time"
    assert val[time_col].max() < test[time_col].min(), \
        "Leakage: val and test overlap in time"

    return TemporalSplit(train=train, val=val, test=test)