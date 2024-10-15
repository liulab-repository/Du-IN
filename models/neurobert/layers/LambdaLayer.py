#!/usr/bin/env python3
"""
Created on 22:17, Jan. 19th, 2024

@author: Norbert Zheng
"""
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "LambdaLayer",
]

# def LambdaLayer class
class LambdaLayer(nn.Module):
    """
    Lambda Layer used to wrap arbitrary expressions as a `nn.Module` object.
    """

    def __init__(self, func, **kwargs):
        """
        Initialize `LambdaLayer` object.

        Args:
            func: callable - The function to be evaluated.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(LambdaLayer, self).__init__(**kwargs)

        # Initialize parameters.
        self.func = func

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
    def forward(self, *args, **kwargs):
        """
        Forward layers in `LambdaLayer` to get the lambda-transformed results.

        Args:
            args: list - The list arguments of lambda function.
            kwargs: dict - The dict arguments of lambda function.

        Returns:
            results: tuple - The results of lambda function.
        """
        return self.func(*args, **kwargs)

if __name__ == "__main__":
    import torch
    # Initialize macros.
    batch_size = 32; seq_len = 240; n_channels = 128
    func = (lambda x: torch.permute(x, dims=[0,2,1]))

    # Initialize input `x`.
    # x - (batch_size, seq_len, n_channels)
    x = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    # Instantiate LambdaLayer.
    ll_inst = LambdaLayer(func=func)
    # Forward layers in `ll_inst`.
    # y - (batch_size, n_channels, seq_len)
    y = ll_inst(x)

