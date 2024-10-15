#!/usr/bin/env python3
"""
Created on 00:06, Dec. 3rd, 2023

@author: Norbert Zheng
"""
import torch
import copy as cp
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "SpatialEncoder",
]

class SpatialEncoder(nn.Module):
    """
    Spatial encoder to capture relationship along spatial axis.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `SpatialEncoder` object.

        Args:
            params: DotDict - The parameters of `SpatialEncoder`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(SpatialEncoder, self).__init__(**kwargs)

        # Initilize parameters.
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
        ## Construct encoder block.
        # Initialize the transformer-encoder layer.
        encoder_layer = nn.TransformerEncoderLayer(
            # Modified `TransformerEncoderLayer` layer parameters.
            d_model=self.params.d_model, nhead=self.params.n_heads,
            dim_feedforward=self.params.d_ff, batch_first=True,
            # Default `TransformerEncoderLayer` layer parameters.
            dropout=0.1, activation="relu", layer_norm_eps=1e-5, norm_first=False, device=None, dtype=None
        )
        # Initialize the transformer-encoder block.
        self.trans_enc = nn.TransformerEncoder(
            # Modified `TransformerEncoder` block parameters.
            encoder_layer=encoder_layer, num_layers=self.params.n_blocks,
            # Default `TransformerEncoder` block parameters.
            norm=None, enable_nested_tensor=False
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
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `SpatialEncoder`.

        Args:
            emb: (batch_size, n_channels, n_segs, d_model) - The input embedding.

        Returns:
            emb: (batch_size, n_channels, n_segs, d_model) - The spatial-encoded embedding.
        """
        # Initialize `batch_size` & `n_channels` & `n_segs` & `d_model` from `emb`.
        batch_size, n_channels, n_segs, d_model = emb.shape
        # Transform the input embedding before attention along spatial axis.
        # emb - (batch_size * n_segs, n_channels, d_model)
        emb = torch.reshape(torch.permute(emb, dims=[0,2,1,3]), shape=(batch_size * n_segs, n_channels, d_model))
        # Forward encoder to get the spatial-encoded embedding.
        emb = self.trans_enc(emb)
        # Transform the spatial-encoded embedding after attention along spatial axis.
        # emb - (batch_size, n_channels, n_segs, d_model)
        emb = torch.permute(torch.reshape(emb, shape=(batch_size, n_segs, n_channels, d_model)), dims=[0,2,1,3])
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    import torch
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; seq_len = 800; n_channels = 16; d_model = 768
    seg_len = 200; n_segs = (seq_len // seg_len)

    # Initialize params.
    params_inst = DotDict({
        # The number of attention blocks.
        "n_blocks": 5,
        # The dimensions of model embedding.
        "d_model": d_model,
        # The number of attention heads.
        "n_heads": 16,
        # The dimensions of the hidden layer in ffn.
        "d_ff": d_model * 4,
    })
    # Initialize input `emb`.
    # emb - (batch_size, n_channels, n_segs, d_model)
    emb = torch.randn((batch_size, n_channels, n_segs, d_model), dtype=torch.float32)
    # Instantiate SpatialEncoder.
    se_inst = SpatialEncoder(params=params_inst)
    # Forward layers in `se_inst`.
    # emb - (batch_size, n_channels, n_segs, d_model)
    emb = se_inst(emb)

