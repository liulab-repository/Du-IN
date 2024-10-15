#!/usr/bin/env python3
"""
Created on 21:08, Nov. 3rd, 2023

@author: Norbert Zheng
"""
import torch
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    # Position Embeddings.
    "PositionEmbedding",
]

"""
position embeddings
"""
# def PositionEmbedding class
class PositionEmbedding(nn.Module):
    """
    Position embedding used to inject position information.
    """

    def __init__(self, d_model, max_len, **kwargs):
        """
        Initialize `PositionEmbedding` object.

        Args:
            d_model: int - The dimensions of model embedding.
            max_len: int - The maximum length of embedding sequence.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(PositionEmbedding, self).__init__(**kwargs)

        # Initialize parameters.
        assert d_model % 2 == 0, (
            "ERROR: The dimensions of model embedding ({:d}) must be a multiples of 2 in PositionEmbedding."
        ).format(d_model)
        self.d_model = d_model; self.max_len = max_len

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
        # Empty `position_encodings` matrix.
        # position_encodings - (1, max_len, d_model)
        position_encodings = np.zeros((1, self.max_len, self.d_model), dtype=np.float32)
        # Get the indexes of available positions (i.e. within `max_len`).
        # pos_idxs - (1, max_len, 1)
        pos_idxs = np.expand_dims(np.arange(0, self.max_len, dtype=np.float32), axis=-1)
        # Get the divide term, i.e. $(1e4)*exp(\frac{-2i}{d_model})$.
        # div_term - (d_model//2,)
        div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32) * -(np.log(1e4) / self.d_model))
        # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$.
        position_encodings[...,0::2] = np.sin(pos_idxs * div_term)
        # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
        position_encodings[...,1::2] = np.cos(pos_idxs * div_term)
        # Use `register_buffer` to ensure that `position_encodings` will be saved to `model.state_dict()`.
        # The only difference between `buffer` and `nn.Parameter` is that `buffer` can calculate its gradient,
        # but `buffer` cannot be updated according to `optimizer`.
        position_encodings = torch.tensor(position_encodings, dtype=torch.float32)
        self.pe = nn.Parameter(position_encodings, requires_grad=False)

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
    def forward(self, emb):
        """
        Forward layers in `PositionEmbedding` to get the position-embedded embeddings.

        Args:
            emb: (batch_size, emb_len, d_model) - The sequence of embeddings.

        Returns:
            emb: (batch_size, emb_len, d_model) - The sequence of position-embedded embeddings.
        """
        # Get the position embeddings `pos_emb`.
        # pos_emb - (1, n_segs, d_model)
        pos_emb = self.pe[:,:emb.shape[-2],:]
        # Add `pos_emb` to `emb` to get the position-embedded embeddings.
        emb = emb + pos_emb
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; emb_len = 5000; d_model = 768

    ## Forward temporal embeddings.
    # Initialize embedded input `emb`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate PositionEmbedding.
    emb_pos_inst = PositionEmbedding(d_model=d_model, max_len=emb_len)
    # Forward layers in `emb_pos_inst`.
    # emb - (batch_size, emb_len, d_model)
    emb = emb_pos_inst(emb)

