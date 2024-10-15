#!/usr/bin/env python3
"""
Created on 21:10, Jan. 20th, 2024

@author: Norbert Zheng
"""
import copy as cp
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from TransformerBlock import TransformerBlock
else:
    from models.neurobert.layers.TransformerBlock import TransformerBlock
import utils.model.torch

__all__ = [
    "TransformerStack",
]

# def TransformerStack class
class TransformerStack(nn.Module):
    """
    Transformer Stack acts as an encoder or a decoder.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `TransformerStack` object.

        Args:
            params: DotDict - The parameters of `TransformerStack`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(TransformerStack, self).__init__(**kwargs)

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
        # Initialize transformer blocks.
        # xfmr_blocks - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.xfmr_blocks = nn.ModuleList(modules=[TransformerBlock(d_model=self.params.d_model,
            n_heads=self.params.n_heads, d_head=self.params.d_head, attn_dropout=self.params.attn_dropout,
            proj_dropout=self.params.proj_dropout, d_ff=self.params.d_ff, ff_dropout=self.params.ff_dropout,
            rot_theta=self.params.rot_theta, norm_first=self.params.norm_first
        ) for block_idx in range(self.params.n_blocks)])

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Fix weights for `xfmr_blocks`.
        for block_idx, xfmr_block_i in enumerate(self.xfmr_blocks):
            # Fix weights for `Linear` layers.
            for module_i in xfmr_block_i.modules():
                if isinstance(module_i, nn.Linear):
                    module_i.weight.data.div_(np.sqrt(2. * (block_idx + 1)))

    """
    network funcs
    """
    # def forward func
    def forward(self, emb, attn_score=None, attn_mask=None, key_padding_mask=None):
        """
        Forward layers in `TransformerStack` to get the mha-ffn transformed embeddings.

        Args:
            emb: (batch_size, emb_len, d_model) - The input embeddings.
            attn_score: (batch_size, n_heads, emb_len, emb_len) - The attention score from the previous layer.
            attn_mask: (emb_len, emb_len) - The pre-defined attention mask within sequence.
            key_padding_mask: (batch_size, emb_len) - The pre-defined key mask within sequence.

        Returns:
            emb: (batch_size, emb_len, d_model) - The mha-ffn transformed embeddings.
            attn_weight: (batch_size, n_heads, emb_len, emb_len) - The attention weight.
            attn_score: (batch_size, n_heads, emb_len, emb_len) - The attention score.
        """
        # Initialize `attn_weight` from `attn_score`.
        attn_weight = utils.model.torch.softmax(attn_score, dim=-1) if attn_score is not None else None
        # Forward `xfmr_blocks` to get the transformed embedding.
        # emb - (batch_size, emb_len, d_model)
        for block_idx in range(len(self.xfmr_blocks)):
            emb, attn_weight, attn_score = self.xfmr_blocks[block_idx](
                emb, attn_score=attn_score if self.params.res_attn else None,
                attn_mask=attn_mask, key_padding_mask=key_padding_mask
            )
        # Return the final `emb` & `attn_weight` & `attn_score`.
        return emb, attn_weight, attn_score

if __name__ == "__main__":
    import torch
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128

    # Initialize params.
    params_inst = DotDict({
        # The number of attention blocks.
        "n_blocks": 4,
        # The flag that indicates whether enable residual attention.
        "res_attn": False,
        # The dimensions of model embedding.
        "d_model": d_model,
        # The number of attention heads.
        "n_heads": 8,
        # The dimensions of attention head.
        "d_head": 64,
        # The power base of rotation angle.
        "rot_theta": 2e1,
        # The dropout probability of attention score.
        "attn_dropout": 0.2,
        # The dropout probability of attention projection.
        "proj_dropout": 0.,
        # The dimensions of the hidden layer in ffn.
        "d_ff": d_model * 4,
        # The dropout probability of the hidden layer in ffn.
        "ff_dropout": [0.2, 0.],
        # The flag that indicates whether execute normalization first.
        "norm_first": False,
    })
    # Initialize input `emb`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate TransformerStack.
    ts_inst = TransformerStack(params=params_inst)
    # Forward layers in `ts_inst`.
    # emb - (batch_size, emb_len, d_model)
    emb, attn_weight, attn_score = ts_inst(emb)

