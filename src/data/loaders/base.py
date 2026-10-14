from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class BaseLoader(ABC):
    """Абстрактный загрузчик датасета фрода."""

    name: str
    target_col: str = "is_fraud"
    time_col: str = "timestamp"

    def __init__(self, data_dir: Path, cache: bool = True):
        self.data_dir = Path(data_dir)
        self.cache = cache

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """Загрузить сырые данные (скачать, если нужно)."""

    @abstractmethod
    def get_schema(self) -> dict:
        """Вернуть схему: типы колонок, категориальные, числовые."""

    def load(self) -> pd.DataFrame:
        """Загрузить с кэшированием."""
        cache_path = self.data_dir / f"{self.name}_processed.parquet"
        if self.cache and cache_path.exists():
            return pd.read_parquet(cache_path)
        df = self.load_raw()
        df = self._postprocess(df)
        if self.cache:
            df.to_parquet(cache_path)
        return df

    def _postprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Общая постобработка: сортировка по времени, типы."""
        df = df.sort_values(self.time_col).reset_index(drop=True)
        return df