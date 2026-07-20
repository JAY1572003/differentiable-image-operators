import torch
import torch.nn as nn


class Operation(nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.params: nn.ParameterDict = nn.ParameterDict()
        self.param_ranges: dict = {}

    def clamp_params(self):
        # NOTE: we clamp with a small margin instead of exactly at the
        # boundary. If a param is clamped to EXACTLY lo/hi, torch.clamp()
        # used later in forward() gives it a gradient of 0 forever, so the
        # optimizer can never move it again ("frozen parameter" bug).
        # Leaving a tiny buffer keeps it just inside the valid range so
        # gradients keep flowing.
        with torch.no_grad():
            for k, p in self.params.items():
                lo, hi = self.param_ranges[k]
                margin = (hi - lo) * 0.01
                p.clamp_(lo + margin, hi - margin)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.params.values())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
