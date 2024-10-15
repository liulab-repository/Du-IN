#!/usr/bin/env python3
"""
Created on 17:14, Nov. 30th, 2023

@author: Norbert Zheng
"""
import copy as cp
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    # Regression Heads.
    "SpecRGSHead",
    # Classification Heads.
    "LabelCLSHead",
]

"""
regression heads
"""
# def SpecRGSHead class
class SpecRGSHead(nn.Module):
    """
    Spectrum regression head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `SpecRGSHead` object.

        Args:
            params: DotDict - The parameters of `SpecRGSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(SpecRGSHead, self).__init__(**kwargs)

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
        # Initialize the regression head.
        self.rgs_head = nn.Sequential()
        # Add hidden block to further transform embedding.
        self.rgs_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.params.d_model, out_features=self.params.d_model,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
            nn.GELU(approximate="none"),
            nn.LayerNorm(
                # Modified `LayerNorm` layer parameters.
                normalized_shape=(self.params.d_model,),
                # Default `LayerNorm` layer parameters.
                eps=1e-5, elementwise_affine=True, device=None, dtype=None
            ),
        ))
        # Add the final regression layer.
        self.rgs_head.append(nn.Linear(
            # Modified `Linear` layer parameters.
            in_features=self.params.d_model, out_features=self.params.n_freqs,
            # Default `Linear` layer parameters.
            bias=True, device=None, dtype=None
        ))

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

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `SpecRGSHead`.

        Args:
            emb: (batch_size, seq_len, d_model) - The encoder-transformed embedding.

        Returns:
            S: (batch_size, seq_len, n_freqs) - The regression spectrum series.
        """
        return self.rgs_head(emb)

"""
classification heads
"""
# def LabelCLSHead class
class LabelCLSHead(nn.Module):
    """
    Label classification head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `LabelCLSHead` object.

        Args:
            params: DotDict - The parameters of `LabelCLSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(LabelCLSHead, self).__init__(**kwargs)

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
        # Initialize classification head.
        self.cls_head = nn.Sequential()
        # Add `Flatten` layer to flatten embedding sequence.
        # TODO: Support multiple time-integration methods, e.g., average.
        self.cls_head.append(nn.Flatten(start_dim=1, end_dim=-1))
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add `Linear` layer.
            self.cls_head.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.d_feature),
                    out_features=self.params.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                nn.ReLU(inplace=False),
            ))
        # Add `Dropout` after hidden `Linear` layers.
        if self.params.dropout > 0.:
            self.cls_head.append(nn.Dropout(p=self.params.dropout, inplace=False))
        # Add the final classification `Linear` layer.
        # Note: Add `nn.Sigmoid()` to avoid gradient explotion.
        self.cls_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=(self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.d_feature),
                out_features=self.params.n_labels,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
            nn.Sigmoid(),
        ))

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

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `LabelCLSHead` to get the predicted labels.

        Args:
            emb: (batch_size, *, d_model) - The encoder-transformed embedding sequence.

        Returns:
            y_pred: (batch_size, n_labels) - The predicted labels.
        """
        return self.cls_head(emb)

if __name__ == "__main__":
    import torch
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; emb_len = 31; d_model = 128; n_freqs = 40; n_channels = 10; n_labels = 10

    ## Forward regression heads.
    # Instantiate params for SpecRGSHead.
    rgsh_spec_params_inst = DotDict({
        # The dimensions of model embedding.
        "d_model": d_model,
        # The number of frequencies.
        "n_freqs": n_freqs,
    })
    # Initialize input `emb`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.randn((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate SpecRGSHead.
    rgsh_spec_inst = SpecRGSHead(params=rgsh_spec_params_inst)
    # Forward layers in `rgsh_spec_inst`.
    # S - (batch_size, emb_len, n_freqs)
    S = rgsh_spec_inst(emb)
    ## Forward classification heads.
    # Instantiate params for LabelCLSHead.
    clsh_label_params_inst = DotDict({
        # The dimensions of feature embedding.
        "d_feature": (n_channels * d_model),
        # The dimensions of the hidden layer in classification block.
        "d_hidden": [128,],
        # The dropout probability of the hidden layer in classification block.
        "dropout": 0.,
        # The number of output classes.
        "n_labels": n_labels,
    })
    # Initialize input `emb`.
    # emb - (batch_size, *, d_model)
    emb = torch.randn((batch_size, n_channels, d_model), dtype=torch.float32)
    # Instantiate LabelCLSHead.
    clsh_label_inst = LabelCLSHead(params=clsh_label_params_inst)
    # Forward layers in `clsh_label_inst`.
    # y_pred - (batch_size, n_labels)
    y_pred = clsh_label_inst(emb)

