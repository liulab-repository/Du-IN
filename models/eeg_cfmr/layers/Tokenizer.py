#!/usr/bin/env python3
"""
Created on 15:27, Mar. 28th, 2024

@author: Norbert Zheng
"""
import torch
import copy as cp
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from LambdaLayer import LambdaLayer
else:
    from models.eeg_cfmr.layers.LambdaLayer import LambdaLayer

__all__ = [
    "ConvTokenizer",
]

# def ConvTokenizer class
class ConvTokenizer(nn.Module):
    """
    Convolution tokenizer used to transform the raw time series.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `ConvTokenizer` object.

        Args:
            params: DotDict - The parameters of `ConvTokenizer`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(ConvTokenizer, self).__init__(**kwargs)

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
        # Initialize convolution blocks.
        self.conv_blocks = nn.Sequential()
        # Add pre-transform layer.
        # pre_trans - (batch_size, seq_len, n_channels) -> (batch_size, 1, n_channels, seq_len)
        self.conv_blocks.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], 1, *x.shape[1:])))),
        ))
        # Add temporal convolution block.
        # conv_t - (batch_size, 1, n_channels, seq_len) -> (batch_size, n_filters, n_channels, token_len)
        for conv_idx in range(len(self.params.n_filters)):
            # Initialize parameters for `Conv2d`.
            n_channels = self.params.n_filters[conv_idx-1] if conv_idx > 0 else 1; n_filters = self.params.n_filters[conv_idx]
            kernel_size = self.params.kernel_sizes[conv_idx]; n_strides = self.params.n_strides[conv_idx]
            padding = _cal_conv_padding(kernel_size=kernel_size, dilation_rate=1)
            # Add `Conv2d` layer.
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(
                    # Modified `Conv2d` layer parameters.
                    in_channels=n_channels, out_channels=n_filters,
                    kernel_size=(1, kernel_size), stride=(1, n_strides), padding=(0, padding),
                    # Default `Conv2d` layer parameters.
                    dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
                ),
            ))
        # Add spatial convlution layer.
        # conv_s - (batch_size, n_filters, n_channels, token_len) -> (batch_size, n_filters, 1, token_len)
        self.conv_blocks.append(nn.Sequential(
            nn.Conv2d(
                # Modified `Conv2d` layer parameters.
                in_channels=self.params.n_filters[-1], out_channels=self.params.n_filters[-1],
                kernel_size=(self.params.n_channels, 1), stride=(1, 1),
                # Default `Conv2d` layer parameters.
                padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
        ))
        # Add batch-norm & ELU layer.
        self.conv_blocks.append(nn.Sequential(
            nn.BatchNorm2d(
                # Modified `BatchNorm2d` layer parameters.
                num_features=self.params.n_filters[-1],
                # Default `BatchNorm2d` layer parameters.
                eps=1e-5, momentum=0.1, affine=True,
                track_running_stats=True, device=None, dtype=None
            ),
            nn.ELU(alpha=1., inplace=False),
        ))
        # Add temporal pooling layer.
        # pool_t - (batch_size, n_filters, 1, token_len) -> (batch_size, n_filters, 1, token_len)
        padding = _cal_conv_padding(kernel_size=self.params.pool_size, dilation_rate=1)
        self.conv_blocks.append(nn.AvgPool2d(
            # Modified `AvgPool2d` layer parameters.
            kernel_size=(1, self.params.pool_size), stride=(1, self.params.pool_stride), padding=(0, padding),
            # Default `AvgPool2d` layer parameters.
            ceil_mode=False, count_include_pad=True, divisor_override=None
        ))
        # Add dropout layer.
        self.conv_blocks.append(nn.Dropout(p=self.params.dropout, inplace=False))
        # Add the final projection layer.
        # proj - (batch_size, n_filters, 1, token_len) -> (batch_size, d_model, 1, token_len)
        self.conv_blocks.append(nn.Sequential(
            nn.Conv2d(
                # Modified `Conv2d` layer parameters.
                in_channels=self.params.n_filters[-1],
                out_channels=self.params.d_model, kernel_size=(1, 1), stride=(1, 1),
                # Default `Conv2d` layer parameters.
                padding=0, dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
        ))
        # Add post-transform layer.
        # post_trans - (batch_size, d_model, 1, token_len) -> (batch_size, token_len, d_model)
        self.conv_blocks.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,3,1]))),
            LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], -1, x.shape[-1])))),
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
        pass

    """
    network funcs
    """
    # def forward func
    def forward(self, X):
        """
        Forward layers in `ConvTokenizer` to get the transformed tokens.

        Args:
            X: (batch_size, seq_len, n_channels) - The raw time series.

        Returns:
            T: (batch_size, token_len, d_model) - The transformed tokens.
        """
        return self.conv_blocks(X)

"""
tool funcs
"""
# def _cal_conv_padding func
def _cal_conv_padding(kernel_size, dilation_rate):
    """
    Calculate the padding of convolution.

    Args:
        kernel_size: int - The size of convolution kernel.
        dilation_rate: int - The dilation rate of convolution.

    Returns:
        padding: int - The padding will be added to both sides of the input.
    """
    # Calculate the padding of convolution.
    padding = int((dilation_rate * (kernel_size - 1)) / 2)
    # Return the final `padding`.
    return padding

if __name__ == "__main__":
    # local dep
    from utils import DotDict

    # Initialize macros.
    batch_size = 32; seq_len = 3000; n_channels = 10; d_model = 128

    ## Forward ConvTokenizer.
    # Initialize params.
    params_inst = DotDict({
        # The number of channels.
        "n_channels": n_channels,
        # The number of convolution filters.
        "n_filters": [40,],
        # The size of convolution kernel.
        "kernel_sizes": [25,],
        # The number of convolution strides.
        "n_strides": [1,],
        # The size of pooling kernel.
        "pool_size": 75,
        # The number of pooling strides.
        "pool_stride": 15,
        # The ratio of dropout.
        "dropout": 0.5,
        # The dimensions of model embedding.
        "d_model": d_model,
    })
    # Initialize input `X`.
    # X - (batch_size, seq_len, n_channels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    # Instantiate ConvTokenizer.
    tokenizer_conv_inst = ConvTokenizer(params=params_inst)
    # Forward layers in `tokenizer_conv_inst`.
    # T - (batch_size, token_len, d_model)
    T = tokenizer_conv_inst(X)

