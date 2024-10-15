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
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
    from LambdaLayer import LambdaLayer
    from GradScaler import GradScaler
else:
    from models.poms.duin.layers.LambdaLayer import LambdaLayer
    from models.poms.duin.layers.GradScaler import GradScaler

__all__ = [
    "PatchTokenizer",
]

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
        for conv_idx in range(len(self.params.n_filters)):
            # Initialize arguments for convolution block.
            n_channels = self.params.n_filters[conv_idx-1] if conv_idx > 0 else self.params.n_channels
            n_filters = self.params.n_filters[conv_idx]; kernel_size = self.params.kernel_sizes[conv_idx]
            n_strides = self.params.n_strides[conv_idx]; dilation_rate = self.params.dilation_rates[conv_idx]
            pool_size = self.params.pool_sizes[conv_idx]; use_bn = self.params.use_bn[conv_idx]; use_res = self.params.use_res[conv_idx]
            # Add the convolution block.
            self.conv_blocks.append(PatchTokenizer._make_conv_block(
                # Modified `_make_conv_block` parameters.
                n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size, n_strides=n_strides,
                dilation_rate=dilation_rate, pool_size=pool_size, use_bn=use_bn, use_res=use_res
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
            if isinstance(module_i, nn.BatchNorm1d):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)

    # def _make_conv_block func
    @staticmethod
    def _make_conv_block(n_channels, n_filters, kernel_size, n_strides,
        dilation_rate, pool_size=1, use_bn=False, use_res=False, **kwargs):
        """
        Make one convolution block, which contains [Conv1d,BatchNorm1d,AvgPool1d].

        Args:
            n_channels: int - The number of input channels.
            n_filters: int - The number of convolution filters.
            kernel_size: int - The dimensions of convolution kernel.
            n_strides: int or (n_dims[list],) - The number of convolution strides.
            dilation_rate: int or (n_dims[list],) - The rate of dilation convolution.
            pool_size: int or (n_dims[list],) - The size of pooling kernel, default as `1`.
            use_bn: bool - The flag that indicates whether use batch-norm, default as `False`.
            use_res: bool - The flag that indicates whether use residual connection, default as `False`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            conv_block: nn.Module - The convolution block, which contains [Conv1d,BatchNorm1d,AvgPool1d].
        """
        # Initialize the convolution block.
        conv_block = nn.Sequential(**kwargs)
        # Add `Conv1d` layer.
        # TODO: Add `Activation` layer after `Conv1d` layer.
        padding = "same" if n_strides == 1 else _cal_conv_padding(kernel_size, dilation_rate)
        conv_block.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            nn.Conv1d(
                # Modified `Conv1d` layer parameters.
                in_channels=n_channels, out_channels=n_filters, kernel_size=kernel_size,
                stride=n_strides, padding=padding, dilation=dilation_rate,
                # Default `Conv1d` layer parameters.
                groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
        ))
        # Add `BatchNorm1d` layer.
        if use_bn:
            conv_block.append(nn.Sequential(
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
                nn.BatchNorm1d(
                    # Modified `BatchNorm1d` layer parameters.
                    num_features=n_filters,
                    # Default `BatchNorm1d` layer parameters.
                    eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None
                ),
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            ))
        # Add `ResidualConnection` layer.
        if use_res:
            # Initialize the residual-convolution block.
            resconv_block = ResidualConnection(module=conv_block, residual_scales=[1., 1.])
            # Update the convolution block.
            conv_block = nn.Sequential(**kwargs); conv_block.append(resconv_block)
        # Add `AvgPool1d` layer.
        if pool_size > 1:
            conv_block.append(nn.Sequential(
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
                nn.AvgPool1d(
                    # Modified `AvgPool1d` layer parameters.
                    kernel_size=pool_size,
                    # Default `AvgPool1d` layer parameters.
                    stride=None, padding=0, ceil_mode=False, count_include_pad=True
                ),
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            ))
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
            X: (batch_size, seq_len, n_channels) - The raw time series.

        Returns:
            T: (batch_size, emb_len, d_model) - The sequence of convolved tokens.
        """
        # Initialize `batch_size` & `seq_len` & `n_channels` from `X`.
        batch_size, seq_len, n_channels = X.shape; n_segs = seq_len // self.params.seg_len
        # Split `X` to get segment-style data.
        # X - (batch_size * n_segs, seg_len, n_channels)
        X = torch.reshape(X, shape=(-1, self.params.seg_len, n_channels))
        # Forward the convolution block to get the convolved tokens.
        # T - (batch_size, n_segs, d_model)
        T = torch.reshape(self.conv_blocks(X), shape=(batch_size, n_segs, -1))
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
    batch_size = 32; seq_len = 3000; n_channels = 10

    ## Forward PatchTokenizer.
    # Initialize params.
    params_inst = DotDict({
        # The number of channels.
        "n_channels": n_channels,
        # The length of element segment.
        "seg_len": 100,
        # The number of filters of each convolution block.
        "n_filters": [128, 128, 16],
        # The size of kernel of each convolution block.
        "kernel_sizes": [19, 3, 3],
        # The number of strides of each convolution block.
        "n_strides": [10, 1, 1],
        # The dilation rate of each convolution block.
        "dilation_rates": [1, 1, 1],
        # The flag that indicates whether use batch-norm.
        "use_bn": [True, True, True],
        # The flag that indicates whether use residual connection.
        "use_res": [False, False, False],
        # The size of pooling of each convolution block.
        "pool_sizes": [1, 1, 1],
    })
    # Initialize input `X`.
    # X - (batch_size, seq_len, n_channels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    # Instantiate PatchTokenizer.
    tokenizer_patch_inst = PatchTokenizer(params=params_inst)
    # Forward layers in `tokenizer_patch_inst`.
    # T - (batch_size, token_len, d_model)
    T = tokenizer_patch_inst(X)

