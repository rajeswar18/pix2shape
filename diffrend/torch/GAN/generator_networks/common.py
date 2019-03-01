import torch
from torch import nn

class ReshapeSplats(nn.Module):
    """Reshape the splats from a 2D to a 1D shape."""

    def forward(self, x):
        """Forward method."""
        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)
        return x