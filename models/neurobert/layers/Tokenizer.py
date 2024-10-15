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
    from models.neurobert.layers.LambdaLayer import LambdaLayer

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
        # Add the convolution blocks.
        for conv_idx in range(len(self.params.n_filters)):
            # Initialize arguments for convolution block.
            n_channels = (
                self.params.n_filters[conv_idx-1] if conv_idx > 0 else self.params.n_channels
            ); n_filters = self.params.n_filters[conv_idx]
            kernel_size = self.params.kernel_sizes[conv_idx]; n_strides = self.params.n_strides[conv_idx]
            # Add the convolution block.
            self.conv_blocks.append(ConvTokenizer._make_conv_block(
                # Modified `_make_conv_block` parameters.
                n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size, n_strides=n_strides
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
        # Initialize weights for `conv_blocks`.
        for module_i in self.conv_blocks.modules():
            # Note: We do not re-initialize the weights of `nn.Conv1d`, we
            # use the default initialization implemented by pytorch.
            if isinstance(module_i, nn.GroupNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)

    # def _make_conv_block func
    @staticmethod
    def _make_conv_block(n_channels, n_filters, kernel_size, n_strides, **kwargs):
        """
        Make one convolution block, which contains [Conv1d,GroupNorm,GELU].

        Args:
            n_channels: int - The number of input channels.
            n_filters: int - The number of convolution filters.
            kernel_size: int - The dimensions of convolution kernel.
            n_strides: int - The number of convolution strides.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            conv_block: nn.Module - The convolution block, which contains [Conv1d,GroupNorm,GELU].
        """
        # Initialize the convolution block.
        conv_block = nn.Sequential(**kwargs)
        # Add `Conv1d` layer.
        padding = _cal_conv_padding(kernel_size=kernel_size, dilation_rate=1)
        conv_block.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            nn.Conv1d(
                # Modified `Conv1d` layer parameters.
                in_channels=n_channels, out_channels=n_filters,
                kernel_size=kernel_size, stride=n_strides, padding=padding,
                # Default `Conv1d` layer parameters.
                dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
        ))
        # Add `GroupNorm` layer.
        assert n_filters % 2 == 0; n_groups = n_filters // 2
        conv_block.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            nn.GroupNorm(
                # Modified `GroupNorm` layer parameters.
                num_groups=n_groups, num_channels=n_filters,
                # Default `GroupNorm` layer parameters.
                eps=1e-5, affine=True, device=None, dtype=None
            ),
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
        ))
        # Add `GELU` layer.
        conv_block.append(nn.GELU(approximate="none"))
        # Return the final `conv_block`.
        return conv_block

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
    batch_size = 32; seq_len = 1200; n_channels = 10; d_model = 128

    ## Forward ConvTokenizer.
    # Initialize params.
    params_inst = DotDict({
        # The number of channels.
        "n_channels": n_channels,
        # The number of filters of each convolution block.
        "n_filters": [128,],
        # The size of kernel of each convolution block.
        "kernel_sizes": [40,],
        # The number of strides of each convolution block.
        "n_strides": [40,],
    })
    # Initialize input `X`.
    # X - (batch_size, seq_len, n_channels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    # Instantiate ConvTokenizer.
    tokenizer_conv_inst = ConvTokenizer(params=params_inst)
    # Forward layers in `tokenizer_conv_inst`.
    # T - (batch_size, token_len, d_model)
    T = tokenizer_conv_inst(X)

