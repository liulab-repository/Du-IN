#!/usr/bin/env python3
"""
Created on 16:37, Jan. 20th, 2024

@author: Norbert Zheng
"""
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "ResidualConnection",
]

# def ResidualConnection class
class ResidualConnection(nn.Module):
    """
    Residual Connection Module.
    """

    def __init__(self, module, residual_scales=[1., 1.], **kwargs):
        """
        Initialize `ResidualConnection` object.

        Args:
            module: `nn.Module` - The module used to do residual connection.
            residual_scales: list - The list of scale factors, including [input_scale,module_scale].
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(ResidualConnection, self).__init__(**kwargs)

        # Initialize parameters.
        self.module = module; self.residual_scales = residual_scales

        # Initialize variables.
        self._init_model(); self._init_weight()

    """
    init funcs
    """
    # def _init_model func
    def _init_model(self):
        """
        Initialize model architecture.

        Args:
            None

        Returns:
            None
        """
        pass

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    network funcs
    """
    # def forward func
    def forward(self, emb, *args, **kwargs):
        """
        Forward layers in `LambdaLayer` to get the lambda-transformed results.

        Args:
            emb: (batch_size, *, d_model) - The input embedding.
            args: list - The list arguments of module forward process.
            kwargs: dict - The dict arguments of module forward process.

        Returns:
            emb: (batch_size, *, d_model) - The residual-transformed embedding.
        """
        return (self.residual_scales[0] * emb) + (self.residual_scales[1] * self.module(emb, *args, **kwargs))

if __name__ == "__main__":
    import torch

    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128; residual_scales = [1., 1.]
    # Instantiate residual module.
    module = nn.Linear(
        # Modified `Linear` layer parameters.
        in_features=d_model, out_features=d_model,
        # Default `Linear` layer parameters.
        bias=True, device=None, dtype=None
    )

    # Initialize input `emb`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate ResidualConnection.
    rc_inst = ResidualConnection(module=module, residual_scales=residual_scales)
    # Forward layers in `rc_inst`.
    # emb - (batch_size, emb_len, d_model)
    emb = rc_inst(emb)

