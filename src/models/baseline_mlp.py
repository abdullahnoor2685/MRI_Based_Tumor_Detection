import torch
import torch.nn as nn
from typing import List


class BaselineMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 3 * 224 * 224,
        hidden_dims: List[int] = None,
        num_classes: int = 4,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        layers: List[nn.Module] = [nn.Flatten()]
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(in_features, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ]
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, num_classes))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)