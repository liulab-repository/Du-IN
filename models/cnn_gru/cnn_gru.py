#!/usr/bin/env python3
"""
Created on 01:03, Mar. 30th, 2024

@author: Norbert Zheng
"""
import copy as cp
import torch.nn as nn
import torch.nn.functional as F
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from layers import *
else:
    from .layers import *
from utils import DotDict

__all__ = [
    "cnn_gru",
]

# def cnn_gru class
class cnn_gru(nn.Module):
    """
    Convolution BiGRU model for classification task introduced by Moses et al. 2021.

    [1] Moses D A, Metzger S L, Liu J R, et al. Neuroprosthesis for decoding speech in a paralyzed person with anarthria[J].
        New England Journal of Medicine, 2021, 385(3): 217-227.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `cnn_gru` object.

        Args:
            params: DotDict - Model parameters initialized by cnn_gru_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(cnn_gru, self).__init__(**kwargs)

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
        # Initialize tokenizer block.
        # tokenizer - (batch_size, seq_len, n_channels) -> (batch_size, token_len, d_model)
        self.tokenizer = ConvTokenizer(params=self.params.cnn)
        # Initialize encoder block.
        # encoder - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        self.encoder = GRUEncoder(params=self.params.gru)
        # Initialize classification block.
        # cls_block - (batch_size, token_len, d_model) -> (batch_size, n_labels)
        self.cls_block = LabelCLSHead(params=self.params.cls)

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
    def forward(self, inputs):
        """
        Forward `conv_xfmr` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,y_true].

        Returns:
            y_pred: (batch_size, n_labels) - The output labels.
            loss: DotDict - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels)
        X = inputs[0]; y_true = inputs[1]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X)
        # Forward encoder block to get the transformed tokens.
        # Only use the last token to further prediction.
        # T - (batch_size, (1 + use_bigru) * d_model)
        T = self.encoder(T); T = T[:,-1,:]
        # Forward classification block to get the predictions.
        # y_pred - (batch_size, n_labels)
        y_pred = self.cls_block(T)
        # Calculate the binary cross entropy loss.
        # loss_cls - torch.float32
        loss_cls = self._loss_cls(y_pred, y_true)
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
        })
        # Return the final `y_pred` & `loss`.
        return y_pred, loss

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target):
        """
        Calculate classification loss between value and target.

        Args:
            value: (batch_size, n_labels) - Value of the object.
            target: (batch_size, n_labels) - Target of the object.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the cross-entropy loss.
        # loss - torch.float32
        loss = F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=value, target=target,
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        # Return the final `loss`.
        return loss

if __name__ == "__main__":
    import torch
    import numpy as np
    # local dep
    import utils.model.torch
    from params.cnn_gru_params import cnn_gru_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32; seq_len = 3000; n_channels = 10; n_labels = 61

    # Initialize random seeds.
    utils.model.torch.set_seeds(42)

    ## Forward cnn_gru.
    # Instantiate params.
    cnn_gru_params_inst = cnn_gru_params(dataset=dataset)
    cnn_gru_params_inst.model.n_channels = n_channels
    cnn_gru_params_inst.model.cnn.n_channels = n_channels
    cnn_gru_params_inst.model.seq_len = seq_len
    token_len = (
        seq_len // np.prod(cnn_gru_params_inst.model.cnn.n_strides)
    ); cnn_gru_params_inst.model.gru.token_len = cnn_gru_params_inst.model.cnn.token_len = token_len
    cnn_gru_params_inst.model.n_labels = n_labels
    cnn_gru_params_inst.model.cls.n_labels = n_labels
    # Initialize input `X` & `y_true`.
    # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    # Instantiate cnn_gru.
    cnn_gru_inst = cnn_gru(params=cnn_gru_params_inst.model); print(cnn_gru_inst)
    # Forward layers in `cnn_gru_inst`.
    # y_pred - (batch_size, n_labels); loss - DotDict
    y_pred, loss = cnn_gru_inst((X, y_true))

