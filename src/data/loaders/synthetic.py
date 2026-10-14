# src/data/loaders/synthetic.py
import numpy as np
import pandas as pd
from .base import BaseLoader


class SyntheticLoader(BaseLoader):
    """
    Генератор синтетических транзакций с известными правилами фрода.

    Ground truth rules:
        R1: amount > 1000 AND hour < 5
        R2: new_device AND geo_distance > 500
        R3: velocity_1h > 5 AND amount > 500
    """

    name = "synthetic"
    GROUND_TRUTH_RULES = [
        "amount > 1000 and hour < 5",
        "new_device and geo_distance > 500",
        "velocity_1h > 5 and amount > 500",
    ]

    def __init__(self, n_samples: int = 100_000, fraud_rate: float = 0.02,
                 seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.fraud_rate = fraud_rate
        self.seed = seed

    def load_raw(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = self.n_samples

        df = pd.DataFrame({
            "amount": rng.lognormal(mean=4, sigma=1.5, size=n),
            "hour": rng.integers(0, 24, size=n),
            "new_device": rng.random(n) < 0.1,
            "geo_distance": rng.exponential(scale=100, size=n),
            "velocity_1h": rng.poisson(lam=1.5, size=n),
            "timestamp": np.sort(rng.uniform(0, 1e9, size=n)),
        })

        # Ground truth логика
        r1 = (df["amount"] > 1000) & (df["hour"] < 5)
        r2 = df["new_device"] & (df["geo_distance"] > 500)
        r3 = (df["velocity_1h"] > 5) & (df["amount"] > 500)
        df["is_fraud"] = (r1 | r2 | r3).astype(int)

        # Шум: 10% меток переворачиваем, чтобы не было идеального сигнала
        noise_mask = rng.random(n) < 0.1
        df.loc[noise_mask, "is_fraud"] = 1 - df.loc[noise_mask, "is_fraud"]

        return df

    def get_schema(self) -> dict:
        return {
            "numeric": ["amount", "hour", "geo_distance", "velocity_1h"],
            "categorical": ["new_device"],
            "target": "is_fraud",
            "time": "timestamp",
        }
