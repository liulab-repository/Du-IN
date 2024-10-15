#!/usr/bin/env python3
"""
Created on 21:25, Mar. 22nd, 2024

@author: Norbert Zheng
"""
import copy as cp
import torch.nn as nn
from collections import OrderedDict
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from Embedding import PositionEmbedding
else:
    from models.brainbert.layers.Embedding import PositionEmbedding

__all__ = [
    # Spectrum Tokenizers.
    "SpecTokenizer",
]

"""
spectrum tokenizers
"""
# def SpecTokenizer class
class SpecTokenizer(nn.Module):
    """
    Spectrum tokenizer to transform the raw spectrum series.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `SpecTokenizer` object.

        Args:
            params: DotDict - The parameters of `SpecTokenizer`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(SpecTokenizer, self).__init__(**kwargs)

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
        # Initialize input-projection layer.
        # TODO: The original order in paper is [in_proj,positional_encoding,layer_norm,dropout],
        # we change it to [in_proj,layer_norm,positional_encoding,dropout] to avoid training collapse.
        # input_encoding - (batch_size, freq_len, n_freqs) -> (batch_size, token_len, d_model)
        self.input_encoding = nn.Sequential(OrderedDict([
            ("in_proj", nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.params.n_freqs, out_features=self.params.d_model,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            )),
            ("layer_norm", nn.LayerNorm(
                # Modified `LayerNorm` layer parameters.
                normalized_shape=(self.params.d_model,),
                # Default `LayerNorm` layer parameters.
                eps=1e-5, elementwise_affine=True, device=None, dtype=None
            )),
            ("positional_encoding", PositionEmbedding(d_model=self.params.d_model, max_len=self.params.max_len)),
            ("dropout", nn.Dropout(p=self.params.dropout, inplace=False)),
        ]))

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

    """
    network funcs
    """
    # def forward func
    def forward(self, S):
        """
        Forward layers in `SpecTokenizer` to get the transformed tokens.

        Args:
            S: (batch_size, freq_len, n_freqs) - The raw spectrum series.

        Returns:
            T: (batch_size, token_len, d_model) - The sequence of transformed tokens.
        """
        return self.input_encoding(S)

if __name__ == "__main__":
    import torch
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; freq_len = 20; n_freqs = 40; d_model = 384

    ## Forward SpecTokenizer.
    # Instantiate params.
    params_inst = DotDict({
        # The number of frequencies.
        "n_freqs": n_freqs,
        # The dimensions of model embedding.
        "d_model": d_model,
        # The maximum length of embedding sequence.
        "max_len": 5000,
        # The ratio of dropout.
        "dropout": 0.1,
        # The length of token sequence.
        "token_len": freq_len,
    })
    # Initialize input `S`.
    # S - (batch_size, freq_len, n_freqs)
    S = torch.rand((batch_size, freq_len, n_freqs), dtype=torch.float32)
    # Instantiate SpecTokenizer.
    tokenizer_spec_inst = SpecTokenizer(params=params_inst)
    # Forward layers in `tokenizer_spec_inst`.
    # T - (batch_size, token_len, d_model)
    T = tokenizer_spec_inst(S)

