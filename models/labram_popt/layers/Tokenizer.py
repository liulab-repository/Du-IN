#!/usr/bin/env python3
"""
Created on 15:43, Jan. 16th, 2024

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
    from GradScaler import GradScaler
else:
    from models.labram_popt.layers.LambdaLayer import LambdaLayer
    from models.labram_popt.layers.GradScaler import GradScaler

__all__ = [
    # 1D-signal Tokenizers.
    "PatchTokenizer",
]

"""
1D-signal tokenizers
"""
# def PatchTokenizer class
class PatchTokenizer(nn.Module):
    """
    Patch tokenizer to transform the raw time series.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `PatchTokenizer` object.

        Args:
            params: DotDict - The parameters of `PatchTokenizer`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(PatchTokenizer, self).__init__(**kwargs)

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
        ## Construct convolution blocks.
        # Initialize convolution blocks.
        self.conv_blocks = nn.Sequential()
        # Add the convolution blocks.
        for conv_idx in range(len(self.params.kernel_sizes)):
            # Initialize arguments for convolution block.
            n_channels = self.params.n_filters[conv_idx-1] if conv_idx > 0 else 1; n_filters = self.params.n_filters[conv_idx]
            kernel_size = (1, self.params.kernel_sizes[conv_idx]); n_strides = (1, self.params.n_strides[conv_idx])
            # Add the convolution block.
            self.conv_blocks.append(PatchTokenizer._make_conv_block(
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
            # Note: We do not re-initialize the weights of `nn.Conv2d`, we
            # use the default initialization implemented by pytorch.
            if isinstance(module_i, nn.GroupNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)

    # def _make_conv_block func
    @staticmethod
    def _make_conv_block(n_channels, n_filters, kernel_size, n_strides, **kwargs):
        """
        Make one convolution block, which contains [Conv2d,GroupNorm,GELU].

        Args:
            n_channels: int - The number of input channels.
            n_filters: int - The number of convolution filters.
            kernel_size: int - The dimensions of convolution kernel.
            n_strides: int - The number of convolution strides.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            conv_block: nn.Module - The convolution block, which contains [Conv2d,GroupNorm,GELU].
        """
        # Initialize the convolution block.
        conv_block = nn.Sequential(**kwargs)
        # Add `Conv2d` layer.
        padding = (_cal_conv_padding(kernel_size=kernel_size_i, dilation_rate=1) for kernel_size_i in kernel_size)
        conv_block.append(nn.Sequential(
            nn.Conv2d(
                # Modified `Conv2d` layer parameters.
                in_channels=n_channels, out_channels=n_filters,
                kernel_size=kernel_size, stride=n_strides, padding=padding,
                # Default `Conv2d` layer parameters.
                dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
        ))
        # Add `GroupNorm` layer.
        assert n_filters % 2 == 0; n_groups = n_filters // 2
        conv_block.append(nn.Sequential(
            nn.GroupNorm(
                # Modified `GroupNorm` layer parameters.
                num_groups=n_groups, num_channels=n_filters,
                # Default `GroupNorm` layer parameters.
                eps=1e-5, affine=True, device=None, dtype=None
            ),
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
        Forward layers in `PatchTokenizer` to get the convolved tokens.

        Args:
            X: (batch_size, n_channels, n_segs, seg_len) - The raw time series.

        Returns:
            T: (batch_size, n_channels, n_segs, d_model) - The sequence of convolved tokens.
        """
        # Initialize `batch_size` & `n_channels` & `n_segs` & `seg_len` from `X`.
        batch_size, n_channels, n_segs, seg_len = X.shape
        # Prepare `X` for convolution over each channel separately.
        # X - (batch_size, 1, n_channels * n_segs, seg_len)
        X = torch.unsqueeze(torch.reshape(X, shape=(batch_size, -1, seg_len)), dim=1)
        # Get the convolved tokens for each channel separately.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = torch.reshape(torch.permute(self.conv_blocks(X), dims=[0,2,3,1]), shape=(batch_size, n_channels, n_segs, -1))
        # Return the final `T`.
        return T

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
    batch_size = 32; seq_len = 800; n_channels = 128; seg_len = 200; n_segs = seq_len // seg_len

    ## Forward PatchTokenizer.
    # Initialize params.
    params_inst = DotDict({
        # The length of patch segment.
        "seg_len": seg_len,
        # The number of filters of each convolution block.
        "n_filters": [8, 8, 8],
        # The size of kernel of each deconvolution block.
        "kernel_sizes": [15, 3, 3],
        # The number of strides of each deconvolution block.
        "n_strides": [8, 1, 1],
    })
    # Initialize input `X`.
    # X - (batch_size, n_channels, n_segs, seg_len)
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    # Instantiate PatchTokenizer.
    tokenizer_patch_inst = PatchTokenizer(params=params_inst)
    # Forward layers in `tokenizer_patch_inst`.
    # T - (batch_size, n_channels, n_segs, d_model)
    T = tokenizer_patch_inst(X)

