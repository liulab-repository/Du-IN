#!/usr/bin/env python3
"""
Created on 18:58, Mar. 28th, 2024

@author: Norbert Zheng
"""
import copy as cp
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    # Classification Heads.
    "LabelCLSHead",
]

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

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize weights.
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
        # Add `Flatten` layer.
        self.cls_head.append(nn.Flatten(start_dim=1, end_dim=-1))
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add current hidden layer.
            self.cls_head.append(nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.d_feature),
                out_features=self.params.d_hidden[hidden_idx],
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ))
        # Add `Dropout` layer.
        self.cls_head.append(nn.Dropout(p=self.params.dropout, inplace=False))
        # Add the final classification `Linear` layer.
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
            emb: (batch_size, emb_len, d_model) - The input embedding.

        Returns:
            y_pred: (batch_size, n_labels) - The predicted labels.
        """
        return self.cls_head(emb)

if __name__ == "__main__":
    import torch
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; emb_len = 30; d_model = 128; n_labels = 61

    ## Forward classification heads.
    # Initialize params for LabelCLSHead.
    clsh_label_params_inst = DotDict({
        # The dimensions of feature embedding.
        "d_feature": (emb_len * d_model),
        # The dimensions of hidden layers.
        "d_hidden": [128,],
        # The dropout ratio after hidden layers.
        "dropout": 0.2,
        # The number of labels.
        "n_labels": n_labels,
    })
    # Initialize input `emb`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate LabelCLSHead.
    clsh_label_inst = LabelCLSHead(params=clsh_label_params_inst)
    # Forward layers in `clsh_label_inst`.
    # y_pred - (batch_size, n_labels)
    y_pred = clsh_label_inst(emb)

