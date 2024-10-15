#!/usr/bin/env python3
"""
Created on 21:15, Dec. 3rd, 2023

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
    # Temporal Embeddings.
    "TimeEmbedding",
]

"""
temporal embeddings
"""
# def TimeEmbedding class
class TimeEmbedding(nn.Module):
    """
    Time encoding for non-recurrent neural networks.
    """

    def __init__(self, d_model, max_len, mode=None, **kwargs):
        """
        Initialize `TimeEmbedding` object.

        Args:
            d_model: int - The dimensions of model embedding.
            max_len: int - The maximum length of the element sequence.
            mode: str - The mode of time embedding, `None` by default to measure the impact of time embedding.
            kwargs: The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(TimeEmbedding, self).__init__(**kwargs)

        # Initialize parameters.
        assert d_model % 2 == 0
        assert mode in [None, "zero", "zeros", "normal", "uniform", "sincos"], (
            "ERROR: Get unknown time embedding mode {} in layers.TimeEmbedding."
        ).format(mode)
        self.d_model = d_model; self.max_len = max_len; self.mode = mode

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
        # Initialize model architecture according to `mode`.
        getattr(self, "_init_model_{}".format(str(self.mode).lower()))()

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

    # def _init_model_none func
    def _init_model_none(self):
        """
        Initialize the `None` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Set non-trainable empty `time_encodings` matrix.
        # time_encodings - (max_len, d_model)
        time_encodings = np.random.uniform(low=-2e-2, high=2e-2, size=(self.max_len, self.d_model)).astype(np.float32)
        # Convert `time_encodings` to torch.Tensor.
        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)
        # Use `register_buffer` to ensure that `time_encodings` will be saved to `model.state_dict()`.
        # The only difference between `buffer` and `nn.Parameter` is that `buffer` can calculate its gradient,
        # but `buffer` cannot be updated according to `optimizer`.
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=False)

    # def _init_model_zero func
    def _init_model_zero(self):
        """
        Initialize the `zero` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Set empty `time_encodings` matrix.
        # time_encodings - (max_len, d_model)
        time_encodings = np.random.uniform(low=-2e-2, high=2e-2, size=(self.max_len, 1)).astype(np.float32)
        # Convert `time_encodings` to torch.Tensor.
        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)
        # Set `time_encodings` as `nn.Parameter`, i.e. trainable.
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)

    # def _init_model_zeros func
    def _init_model_zeros(self):
        """
        Initialize the `zeros` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Set empty `time_encodings` matrix.
        # time_encodings - (max_len, d_model)
        time_encodings = np.random.uniform(low=-2e-2, high=2e-2, size=(self.max_len, self.d_model)).astype(np.float32)
        # Convert `time_encodings` to torch.Tensor.
        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)
        # Set `time_encodings` as `nn.Parameter`, i.e. trainable.
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)

    # def _init_model_normal func
    def _init_model_normal(self):
        """
        Initialize the `normal` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Set normal-distributed `time_encodings` matrix.
        # time_encodings - (max_len, d_model)
        time_encodings = np.random.normal(loc=0., scale=1., size=(self.max_len, self.d_model)).astype(np.float32)
        # Convert `time_encodings` to torch.Tensor.
        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)
        # Set `time_encodings` as `nn.Parameter`, i.e. trainable.
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)

    # def _init_model_uniform func
    def _init_model_uniform(self):
        """
        Initialize the `uniform` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Set uniform-distributed `time_encodings` matrix.
        # time_encodings - (max_len, d_model)
        time_encodings = np.random.uniform(low=0., high=1., size=(self.max_len, self.d_model)).astype(np.float32)
        # Convert `time_encodings` to torch.Tensor.
        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)
        # Set `time_encodings` as `nn.Parameter`, i.e. trainable.
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)

    # def _init_model_sincos func
    def _init_model_sincos(self):
        """
        Initialize the `sincos` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Empty `time_encodings` matrix.
        # time_encodings - (max_len, d_model)
        time_encodings = np.zeros((self.max_len, self.d_model), dtype=np.float32)
        # Get the indexes of available times (i.e. within `max_len`).
        # time_idxs - (max_len, 1)
        time_idxs = np.expand_dims(np.arange(0, self.max_len, dtype=np.float32), axis=-1)
        # Get the divide term, i.e. $(1e4)*exp(\frac{-2i}{d_model})$.
        # div_term - (d_model//2,)
        div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32) * -(np.log(1e4) / self.d_model))
        # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$.
        time_encodings[:,0::2] = np.sin(time_idxs * div_term)
        # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
        time_encodings[:,1::2] = np.cos(time_idxs * div_term)
        # Convert `time_encodings` to torch.Tensor.
        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)
        # Use `register_buffer` to ensure that `time_encodings` will be saved to `model.state_dict()`.
        # The only difference between `buffer` and `nn.Parameter` is that `buffer` can calculate its gradient,
        # but `buffer` cannot be updated according to `optimizer`.
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=False)

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `TimeEmbedding` to get the time-embedded result.

        Args:
            emb: (batch_size, n_channels, n_segs, d_model) - The sequence of elements.

        Returns:
            emb: (batch_size, n_channels, n_segs, d_model) - The sequence of time-embedded elements.
        """
        # Get the time embeddings `pe` according to the `n_segs`.
        # time_emb - (n_segs, d_model)
        time_emb = self.time_encodings[:emb.shape[-2],:]
        # Add `time_emb` to `emb` to get the time-embedded embedding.
        # Note: We have to make sure that `emb` is 0-mean 1-var distribution.
        # If we apply layer normalization over `emb`, `emb` is 0-mean 1/sqrt(d_model)-var
        # distribution, i.e. we have to multiply `emb` with `sqrt(d_model)`.
        emb = emb + time_emb[None,None,...]
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; seq_len = 4000; n_channels = 16; d_model = 768
    seg_len = 1000; n_segs = (seq_len // seg_len); mode = "normal"

    ## Forward temporal embeddings.
    # Initialize embedded input `emb`.
    # emb - (batch_size, n_channels, n_segs, d_model)
    emb = torch.randn((batch_size, n_channels, n_segs, d_model), dtype=torch.float32)
    # Instantiate TimeEmbedding.
    emb_time_inst = TimeEmbedding(d_model=d_model, max_len=n_segs, mode=mode)
    # Forward layers in `emb_time_inst`.
    # emb - (batch_size, n_channels, n_segs, d_model)
    emb = emb_time_inst(emb)

