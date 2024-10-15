#!/usr/bin/env python3
"""
Created on 19:08, Jan. 4th, 2024

@author: Norbert Zheng
"""
import torch
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "SubjectLayer",
]

# def SubjectLayer class
class SubjectLayer(nn.Module):
    """
    Subject Layer used to transform embeddings with specified subj id.
    """

    def __init__(self, d_input, n_subjects, d_output, use_bias=False, **kwargs):
        """
        Initialize `SubjectLayer` object.

        Args:
            d_input: int - The number of input embedding.
            n_subjects: int - The number of available subjects.
            d_output: int - The dimensions of output embedding.
            use_bias: bool - The flag that indicates whether use learnable bias to further shift projection.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(SubjectLayer, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_input = d_input; self.n_subjects = n_subjects
        self.d_output = d_output; self.use_bias = use_bias

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
        # Initialize weight variables.
        # Note: The output of `W` only acts as the projection matrix, no need to enable bias.
        # TODO: As the output of `W` acts like the weight part of `Linear` layer, change
        # the [kernel,bias] initializer to `kernel_initializer` used in `Linear` layer.
        # W - ((n_subjects,) -> (d_input*d_output,))
        self.W = nn.Linear(
            # Modified `Linear` layer parameters.
            in_features=self.n_subjects, out_features=(self.d_input * self.d_output), bias=False,
            # Default `Linear` layer parameters.
            device=None, dtype=None
        )
        # Initialize bias variables.
        # TODO: As the output of `B` acts like the bias part of `Linear` layer, change
        # the [kernel,bias] initializer to `bias_initializer` used in `Linear` layer.
        # B - (n_subjects,) -> (d_output,)
        self.B = nn.Linear(
            # Modified `Linear` layer parameters.
            in_features=self.n_subjects, out_features=self.d_output, bias=False,
            # Default `Linear` layer parameters.
            device=None, dtype=None
        ) if self.use_bias else None

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for `W`.
        nn.init.trunc_normal_(self.W.weight, mean=0., std=0.02)
        if self.W.bias is not None: nn.init.zeros_(self.W.bias)
        # Initialize weights for `B`.
        if self.B is not None:
            nn.init.constant_(self.B.weight, val=0.)
            if self.B.bias is not None: nn.init.zeros_(self.B.bias)

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `SubjectLayer` to get the subject-transformed embeddings.

        Args:
            inputs: tuple - The input data, containing [X,subj_id]. `X` is the input embeddings of shape
                (batch_size, *, d_input), `subj_id` is the subject indices of shape (batch_size, n_subjects).

        Returns:
            Z: (batch_size, *, d_output) - The subject-transformed embeddings.
        """
        # Initialize `X` & `subj_id` from `inputs`.
        X = inputs[0]; subj_id = inputs[1]
        # Get subject-specified transformation matrix.
        # W_s - (batch_size, d_input, d_output)
        W_s = torch.reshape(self.W(subj_id), shape=(-1, self.d_input, self.d_output))
        # Use subject-specified transformation matrix to get the subject-transformed embeddings.
        # Z - (batch_size, *, d_output)
        Z = torch.reshape(torch.matmul(
            torch.reshape(X, shape=(X.shape[0], -1, X.shape[-1])), W_s
        ), shape=(*X.shape[:-1], W_s.shape[-1]))
        # Use subject-specified shift vector to get the subject-transformed embeddings.
        # TODO: Support `Z = W * (X + B)`-style linear projection, enhance the relationship between `W` and `B`.
        Z = torch.reshape((
            torch.reshape(Z, shape=(Z.shape[0], -1, Z.shape[-1])) + torch.unsqueeze(self.B(subj_id), dim=-2)
        ), shape=Z.shape) if self.use_bias else Z
        # Return the final `Z`.
        return Z

    """
    tool funcs
    """
    # def get_weight_i func
    def get_weight_i(self):
        """
        Get the contribution weights of each input channel.

        Args:
            None

        Returns:
            weight: (n_subjects, d_input) - The contribution weights corresponding to each input channel.
        """
        # Get weight from `W`.
        # weight - (n_subjects, d_input, d_output)
        weight = torch.reshape(self.W.weight, shape=(self.n_subjects, self.d_input, self.d_output)); weight = weight.detach().cpu()
        # Average across multiple spatial channels to get the weights of each channel.
        # weight - (n_subjects, d_input)
        weight = torch.mean(torch.abs(weight), dim=-1)
        # Return the final `weight`.
        return weight

if __name__ == "__main__":
    import numpy as np

    # Initialize macros.
    batch_size = 32; seq_len = 3000; d_input = 10; n_subjects = 12; d_output = 16; use_bias = True

    # Initialize input `X` & `subj_id`.
    # X - (batch_size, seq_len, d_input), subj_id - (batch_size, n_subjects)
    X = torch.rand((batch_size, seq_len, d_input), dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate SubjectLayer.
    sl_inst = SubjectLayer(d_input=d_input, n_subjects=n_subjects, d_output=d_output, use_bias=use_bias)
    # Forward layers in `sl_inst`.
    # Z - (batch_size, seq_len, d_output)
    Z = sl_inst((X, subj_id))

