"""Фабрика загрузчиков датасетов."""
from .base import BaseLoader
from .ieee_cis import IEEECISLoader
# from .synthetic import SyntheticLoader

LOADERS: dict[str, type[BaseLoader]] = {
    "ieee_cis": IEEECISLoader,
    # "synthetic": SyntheticLoader,
}


def get_loader(name: str, **kwargs) -> BaseLoader:
    """
    Вернуть загрузчик по имени.

    Пример:
        loader = get_loader("ieee_cis", data_dir="data/raw")
        df = loader.load()
    """
    if name not in LOADERS:
        available = ", ".join(LOADERS.keys())
        raise ValueError(f"Unknown loader: '{name}'. Available: {available}")
    return LOADERS[name](**kwargs)
