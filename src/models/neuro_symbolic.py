# src/models/neuro_symbolic.py
import torch
import torch.nn as nn
from src.data.preprocessing.multi_discretizer import MultiFeatureDiscretizer
from src.rules.engine import RuleEngine
from src.rules.predicates import Predicate
from src.rules.rule_layer import FuzzyRule


class NeuroSymbolicFraudModel(nn.Module):
    """
    Гибридная модель: Learnable Discretizer -> Rule Engine.
    Пока без MLP-бэкбона (добавим позже), чтобы проверить, 
    как чистые правила справляются с задачей.
    """
    def __init__(
        self, 
        n_features: int, 
        n_bins: int = 16, 
        n_rules: int = 10
    ):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins
        
        # 1. Дискретизатор (превращает числа в soft-бины)
        self.discretizer = MultiFeatureDiscretizer(
            n_features=n_features, 
            n_bins=n_bins
        )
        
        # 2. Инициализируем N правил случайным образом.
        # (Позже мы сделаем так, чтобы правила обучались/индуцировались сами)
        rules = []
        for _ in range(n_rules):
            # Каждое правило состоит из 2 предикатов (фича + бин)
            p1 = Predicate(
                feature_idx=torch.randint(0, n_features, (1,)).item(),
                bin_idx=torch.randint(0, n_bins, (1,)).item()
            )
            p2 = Predicate(
                feature_idx=torch.randint(0, n_features, (1,)).item(),
                bin_idx=torch.randint(0, n_bins, (1,)).item()
            )
            rules.append(FuzzyRule([p1, p2]))
            
        self.rule_engine = RuleEngine(rules)
        
        # 3. Финальный слой (сигмоида) для получения вероятности [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (batch, n_features) — сырые числовые фичи
        x_discrete = self.discretizer(x)     # (batch, n_features, n_bins)
        score, rule_outputs = self.rule_engine(x_discrete) # (batch, 1), (batch, n_rules)
        prob = self.sigmoid(score)
        return prob, rule_outputs
