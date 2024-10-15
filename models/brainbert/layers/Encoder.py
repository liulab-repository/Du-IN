#!/usr/bin/env python3
"""
Created on 17:45, Mar. 25th, 2024

@author: Norbert Zheng
"""
import copy as cp
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "Encoder",
]

# def Encoder class
class Encoder(nn.Module):
    """
    The encoder used to transform the spectrum tokens.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `Encoder` object.

        Args:
            params: DotDict - The parameters of `Encoder`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(Encoder, self).__init__(**kwargs)

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
        # Initialize transformer block.
        xfmr_block = nn.TransformerEncoderLayer(
            # Modified `TransformerEncoderLayer` layer parameters.
            d_model=self.params.d_model, nhead=self.params.n_heads,
            dim_feedforward=self.params.d_ff, activation=self.params.activation, batch_first=True,
            # Default `TransformerEncoderLayer` layer parameters.
            dropout=0.1, layer_norm_eps=1e-5, norm_first=False, device=None, dtype=None
        )
        # Initialize transformer stack.
        self.transformer = nn.TransformerEncoder(
            # Modified `TransformerEncoder` layer parameters.
            encoder_layer=xfmr_block, num_layers=self.params.n_blocks,
            # Default `TransformerEncoder` layer parameters.
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
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)
        # Fix weights for `transformer`.
        for block_idx, xfmr_block_i in enumerate(self.transformer.layers):
            # Fix weights for `Linear` layers.
            for module_i in xfmr_block_i.modules():
                if isinstance(module_i, nn.Linear):
                    module_i.weight.data.div_(np.sqrt(2. * (block_idx + 1)))

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `Encoder` to get the mha-ffn transformed embeddings.

        Args:
            emb: (batch_size, emb_len, d_model) - The input embeddings.

        Returns:
            emb: (batch_size, emb_len, d_model) - The mha-ffn transformed embeddings.
        """
        # Forward `transformer` to get the mha-ffn transformed embeddings.
        # emb - (batch_size, emb_len, d_model)
        emb = self.transformer(
            # Modified `forward` function arguments.
            src=emb,
            # Default `forward` function arguments.
            mask=None, src_key_padding_mask=None, is_causal=None
        )
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    import torch
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 768

    # Initialize params.
    params_inst = DotDict({
        # The number of attention blocks.
        "n_blocks": 6,
        # The dimensions of model embedding.
        "d_model": d_model,
        # The number of attention heads.
        "n_heads": 12,
        # The dimensions of the hidden layer in ffn.
        "d_ff": d_model * 4,
        # The type of activation.
        "activation": ["relu", "gelu"][-1],
    })
    # Initialize input `emb`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate Encoder.
    encoder_inst = Encoder(params=params_inst)
    # Forward layers in `encoder_inst`.
    # emb - (batch_size, emb_len, d_model)
    emb = encoder_inst(emb)

