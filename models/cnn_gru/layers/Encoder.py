#!/usr/bin/env python3
"""
Created on 00:33, Mar. 30th, 2024

@author: Norbert Zheng
"""
import copy as cp
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from LambdaLayer import LambdaLayer
else:
    from models.cnn_gru.layers.LambdaLayer import LambdaLayer

__all__ = [
    "GRUEncoder",
]

# def GRUEncoder class
class GRUEncoder(nn.Module):
    """
    GRU encoder used to transform the token sequence.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `GRUEncoder` object.

        Args:
            params: DotDict - The parameters of `GRUEncoder`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(GRUEncoder, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

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
        # Initialize gru layers.
        self.gru_layers = nn.Sequential(
            nn.GRU(
                # Modified `GRU` layer parameters.
                input_size=self.params.d_model, hidden_size=self.params.d_model, num_layers=self.params.n_layers,
                batch_first=True, dropout=self.params.dropout, bidirectional=self.params.use_bigru,
                # Default `GRU` layer parameters.
                bias=True
            ),
            LambdaLayer(func=(lambda x: x[0])),
        )

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
    def forward(self, T):
        """
        Forward layers in `GRUEncoder` to get the transformed tokens.

        Args:
            T: (batch_size, token_len, d_model) - The input tokens.

        Returns:
            T: (batch_size, token_len, (1 + use_bigru) * d_model) - The transformed tokens.
        """
        return self.gru_layers(T)

if __name__ == "__main__":
    import torch
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; token_len = 30; d_model = 128

    ## Forward GRUEncoder.
    # Initialize params.
    params_inst = DotDict({
        # The dimensions of model embedding.
        "d_model": d_model,
        # The number of gru layers.
        "n_layers": 2,
        # The dropout ratio of gru layers.
        "dropout": 0.2,
        # The flag that indicates whether use BiGRU.
        "use_bigru": True,
    })
    # Initialize input `T`.
    # T - (batch_size, token_len, d_model)
    T = torch.rand((batch_size, token_len, d_model), dtype=torch.float32)
    # Instantiate GRUEncoder.
    encoder_gru_inst = GRUEncoder(params=params_inst)
    # Forward layers in `encoder_gru_inst`.
    # T - (batch_size, token_len, (1 + use_bigru) * d_model)
    T = encoder_gru_inst(T)

