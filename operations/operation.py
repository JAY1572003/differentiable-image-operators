import torch
import torch.nn as nn


class Operation(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.params: nn.ParameterDict = nn.ParameterDict()
        self.param_ranges: dict = {}

    def clamp_params(self):
        with torch.no_grad():
            for k, p in self.params.items():
                lo, hi = self.param_ranges[k]
                p.clamp_(lo, hi)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.params.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
