#!/usr/bin/env python3
"""
Created on 18:58, Mar. 28th, 2024

@author: Norbert Zheng
"""
import torch
import copy as cp
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from LambdaLayer import LambdaLayer
else:
    from models.neurobert.layers.LambdaLayer import LambdaLayer

__all__ = [
    # Regression Heads.
    "TimeRGSHead",
    # Classification Heads.
    "LabelCLSHead",
]

"""
regression heads
"""
# def TimeRGSHead class
class TimeRGSHead(nn.Module):
    """
    Time series regression head.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `TimeRGSHead` object.

        Args:
            params: DotDict - The parameters of `TimeRGSHead`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(TimeRGSHead, self).__init__(**kwargs)

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
        # Initialize regression head.
        self.rgs_head = nn.Sequential()
        # Add `ConvTNDBlock` layers.
        # TODO: Add `ConvNDBlock` layers to improve model ability.
        seq_len = self.params.emb_len
        for deconv_idx in range(len(self.params.n_filters)):
            # Initialize arguments for deconvolution block.
            n_channels = self.params.n_filters[deconv_idx-1] if deconv_idx > 0 else self.params.d_model
            seq_len = seq_len * self.params.n_strides[deconv_idx]; n_filters = self.params.n_filters[deconv_idx]
            kernel_size = self.params.kernel_sizes[deconv_idx]; n_strides = self.params.n_strides[deconv_idx]
            # Add the deconvolution layer.
            self.rgs_head.append(TimeRGSHead._make_deconv_block(
                # Modified `_make_deconv_block` parameters.
                n_channels=n_channels, seq_len=seq_len, n_filters=n_filters,
                kernel_size=kernel_size, n_strides=n_strides
            ))
        # Add hidden `Linear` layers.
        for hidden_idx in range(len(self.params.d_hidden)):
            # Add `Linear` layer.
            self.rgs_head.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.params.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.params.n_filters[-1]),
                    out_features=self.params.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                nn.GELU(approximate="none"),
                nn.LayerNorm(
                    # Modified `LayerNorm` layer parameters.
                    normalized_shape=(self.params.d_hidden[hidden_idx],),
                    # Default `LayerNorm` layer parameters.
                    eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None
                ),
            ))
        # Add the final regression `Linear` layer.
        self.rgs_head.append(nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.params.d_hidden[-1] if len(self.params.d_hidden) > 0 else self.params.n_filters[-1],
                out_features=self.params.n_channels,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
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
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)

    # def _make_deconv_block func
    @staticmethod
    def _make_deconv_block(n_channels, seq_len, n_filters, kernel_size, n_strides, **kwargs):
        """
        Make one deconvolution block, which contains [ConvTranspose1d,Conv1d,Conv1d].

        Args:
            n_channels: int - The number of input channels.
            seq_len: int - The length of embedding sequence.
            n_filters: int - The number of deconvolution filters.
            kernel_size: int - The dimensions of deconvolution kernel.
            n_strides: int - The number of deconvolution strides.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            deconv_block: nn.Module - The deconvolution block, which contains [ConvTranspose1d,Conv1d,Conv1d].
        """
        # Initialize the deconvolution block.
        deconv_block = nn.Sequential(**kwargs)
        # Add `ConvTranspose1d` layer.
        padding, output_padding = _cal_conv_padding(input_len=(seq_len * n_strides),
            output_len=seq_len, kernel_size=kernel_size, n_strides=n_strides, dilation_rate=1)
        deconv_block.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            nn.ConvTranspose1d(
                # Modified `ConvTranspose1d` layer parameters.
                in_channels=n_channels, out_channels=n_filters, kernel_size=kernel_size,
                stride=n_strides, padding=padding, output_padding=output_padding,
                # Default `ConvTranspose1d` layer parameters.
                dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
        ))
        # Return the final `deconv_block`.
        return deconv_block

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `TimeRGSHead` to get the regression time series.

        Args:
            emb: (batch_size, emb_len, d_model) - The encoder-transformed embedding sequence.

        Returns:
            X: (batch_size, seq_len, n_channels) - The regression time series.
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

"""
tool funcs
"""
# def _cal_conv_padding func
def _cal_conv_padding(input_len, output_len, kernel_size, n_strides, dilation_rate):
    """
    Calculate the padding of convolution.

    Args:
        input_len: int - The length of input embedding sequence.
        output_len: int - The length of output embedding sequence.
        kernel_size: int - The size of convolution kernel.
        n_strides: int - The number of convolution strides.
        dilation_rate: int - The dilation rate of convolution.

    Returns:
        padding: int - The padding will be added to both sides of the input.
        output_padding: int - The additional size added to one side of the output shape.
    """
    # Calculate the padding of convolution.
    padding_diff = (output_len - 1) * n_strides + (dilation_rate * (kernel_size - 1) + 1) - input_len
    padding = int(np.ceil(padding_diff / 2))
    output_padding = 2 * padding - padding_diff
    # Return the final `padding`.
    return padding, output_padding

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

