import torch
import torch.nn as nn
from .rule_layer import FuzzyRule

class RuleEngine(nn.Module):
    """
    Rule set, applied to transactions.
    """
    def __init__(self, rules: list[FuzzyRule]):
        super().__init__()
        self.rules = nn.ModuleList(rules)

    def forward(self, x_discrete: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_discrete: (batch, n_features, n_bins)
        
        # Collect outputs of all rules: (batch, n_rules)
        rule_outputs = torch.cat([r(x_discrete) for r in self.rules], dim=-1)
        
        # Final score: sum of activated rules
        # (in the real model, this could be an MLP or logistic regression)
        score = rule_outputs.sum(dim=-1, keepdim=True)
        
        return score, rule_outputs