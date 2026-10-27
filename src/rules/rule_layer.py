import torch
import torch.nn as nn
from .predicates import Predicate


class FuzzyRule(nn.Module):
    """
    Fuzzy rule of the form: IF P1 AND P2 AND ... THEN weight
    """
    def __init__(self, predicates: list[Predicate], rule_weight: float = 1.0):
        super().__init__()
        self.predicates = nn.ModuleList(predicates)
        # Trainable weight for the rule (how important it is for the final score)
        self.weight = nn.Parameter(torch.tensor(rule_weight))

    def forward(self, x_discrete: torch.Tensor) -> torch.Tensor:
        # x_discrete: (batch, n_features, n_bins)
        # Computing the truth degree of each predicate: (batch, 1)
        pred_truths = [p(x_discrete) for p in self.predicates]
        
        # Fuzzy AND (product of probabilities)
        rule_truth = torch.cat(pred_truths, dim=-1).prod(dim=-1, keepdim=True)
        
        # Applying the rule weight (softplus ensures it's positive)
        return rule_truth * torch.nn.functional.softplus(self.weight)