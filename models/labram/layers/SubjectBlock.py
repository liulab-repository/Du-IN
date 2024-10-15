#!/usr/bin/env python3
"""
Created on 19:31, Jan. 4th, 2024

@author: Norbert Zheng
"""
import copy as cp
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from SubjectLayer import *
else:
    from models.labram.layers.SubjectLayer import *

__all__ = [
    "SubjectBlock",
]

# def SubjectBlock class
class SubjectBlock(nn.Module):
    """
    Subject Block used to transform embeddings with specified subj id.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `SubjectBlock` object.

        Args:
            params: DotDict - The parameters of `SubjectBlock`.
            kwargs: dict - The arguments related to initialize `tf.keras.layers.Layer`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(SubjectBlock, self).__init__(**kwargs)

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
        # Initialize the subject layer.
        self.subj_layer = SubjectLayer(
            # Modified `SubjectLayer` layer parameters.
            d_input=self.params.d_input, n_subjects=self.params.n_subjects,
            d_output=self.params.d_output, use_bias=self.params.use_bias
        )
        # Initialize the projection layer.
        # TODO: As `Conv1D` layer (w/ kernel size `1`) acts like `Dense` layer, change the [kernel,bias] initializer.
        self.proj_layer = nn.Linear(
            # Modified `Linear` layer parameters.
            in_features=self.params.d_output, out_features=self.params.d_output, bias=True,
            # Default `Linear` layer parameters.
            device=None, dtype=None
        ) if self.params.use_proj else None

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for `proj_layer`.
        if self.proj_layer is not None:
            nn.init.trunc_normal_(self.proj_layer.weight, mean=0., std=0.02)
            if module_i.bias is not None: nn.init.constant_(self.proj_layer.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `SubjectBlock` to get the subject-transformed embeddings.

        Args:
            inputs: tuple - The input data.

        Returns:
            emb: (batch_size, *, d_output) - The subject-transformed embeddings.
        """
        # Initialize `X` & `subj_id` from `inputs`.
        # X - (batch_size, *, d_input); subj_id - (batch_size, n_subjects)
        X = inputs[0]; subj_id = inputs[1]
        # Forward the subject layer to get the subject-transformed embeddings.
        # emb - (batch_size, *, d_output)
        emb = self.subj_layer((X, subj_id))
        # Forward the projection layer to further integrate spatial information.
        emb = self.proj_layer(emb) if self.params.use_proj else emb
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    import torch
    import numpy as np
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; seq_len = 240; d_input = 128; d_output = 256; n_subjects = 10; use_proj = True; use_bias = True
    # Instantiate params.
    sb_params_inst = DotDict({
        # The number of subjects.
        "n_subjects": n_subjects,
        # The dimensions of input embedding.
        "d_input": d_input,
        # The dimensions of output embedding.
        "d_output": d_output,
        # The flag that indicates whether enable embedding shift.
        "use_bias": use_bias,
        # The flag that indicates whether enable projection layer.
        "use_proj": use_proj,
    })

    # Instantiate SubjectBlock.
    sb_inst = SubjectBlock(sb_params_inst)
    # Initialize input `X` & `subj_id`.
    # X - (batch_size, seq_len, d_input); subj_id - (batch_size, n_subjects)
    X = torch.rand((batch_size, seq_len, d_input), dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Forward layers in `sb_inst`.
    # emb - (batch_size, seq_len, d_output)
    emb = sb_inst((X, subj_id))

