"""Дискретизация числовых фич для rule induction.

LearnableDiscretizer — слой с обучаемыми границами бинов.
Начальные границы задаются квантилями (data-driven),
затем двигаются градиентным спуском.
"""
import numpy as np
import torch
import torch.nn as nn


class LearnableDiscretizer(nn.Module):
    """
    Разбивает числовую фичу на n_bins интервалов.
    Границы — обучаемые параметры.

    Forward:
        x: (batch,) сырые значения
        return: (batch, n_bins) one-hot-ish soft assignment
    """

    def __init__(
        self,
        n_bins: int = 16,
        init_range: tuple[float, float] = (-5.0, 5.0),
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_bins = n_bins
        self.temperature = temperature

        # n_bins - 1 внутренних границ, инициализируются равномерно
        init_boundaries = torch.linspace(init_range[0], init_range[1], n_bins - 1)
        self.boundaries = nn.Parameter(init_boundaries)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch,)
        return: (batch, n_bins) — soft assignment (differentiable)
        """
        # Расстояние от x до каждой границы
        # shape: (batch, n_bins - 1)
        diff = x.unsqueeze(-1) - self.boundaries.unsqueeze(0)

        # Sigmoid с temperature: приближение к step function
        # cumsum даёт "сколько границ пройдено" → номер бина
        soft_bins = torch.sigmoid(diff / self.temperature).sum(dim=-1)

        # Округляем до one-hot через softmax по бинам
        # shape: (batch, n_bins)
        bin_centers = torch.arange(self.n_bins, device=x.device).float()
        logits = -((soft_bins.unsqueeze(-1) - bin_centers) ** 2) / self.temperature
        return torch.softmax(logits, dim=-1)

    def init_from_data(self, x: np.ndarray):
        """Инициализировать границы квантилями данных."""
        quantiles = np.linspace(0, 100, self.n_bins + 1)[1:-1]
        boundaries = np.percentile(x, quantiles)
        with torch.no_grad():
            self.boundaries.copy_(torch.tensor(boundaries, dtype=torch.float32))