#!/usr/bin/env python3
"""
Created on 21:16, Mar. 16th, 2024

@author: Norbert Zheng
"""
import torch
import copy as cp
import torch.nn as nn
import torch.nn.functional as F
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from LambdaLayer import LambdaLayer
else:
    from models.brant.layers.LambdaLayer import LambdaLayer

__all__ = [
    "LinearTokenizer",
    "ConvTokenizer",
    "PatchTokenizer",
]

# def LinearTokenizer class
class LinearTokenizer(nn.Module):
    """
    Linear tokenizer to transform the raw time series.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `LinearTokenizer` object.

        Args:
            params: DotDict - The parameters of `LinearTokenizer`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(LinearTokenizer, self).__init__(**kwargs)

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
        # Initialize signal mask embedding.
        # mask_encoding - (seg_len,)
        mask_encoding = torch.zeros((self.params.seg_len,), dtype=torch.float32)
        self.mask_encoding = nn.Parameter(mask_encoding, requires_grad=False)
        # Initialize power mask embedding.
        # power_mask_encoding - (d_model,)
        power_mask_encoding = torch.zeros((self.params.d_model,), dtype=torch.float32)
        self.power_mask_encoding = nn.Parameter(power_mask_encoding, requires_grad=False)
        # Initialize band embedding.
        # band_encoding - (n_bands, d_model)
        band_encoding = torch.ones((self.params.n_bands, self.params.d_model), dtype=torch.float32)
        self.band_encoding = nn.Parameter(band_encoding, requires_grad=True)
        # Initialize position embedding.
        # positional_encoding - (seq_len, d_model)
        positional_encoding = torch.ones((self.params.max_segs, self.params.d_model), dtype=torch.float32)
        self.positional_encoding = nn.Parameter(positional_encoding, requires_grad=True)
        # Initialize projection layer.
        # proj - (batch_size, n_channels, n_segs, seg_len) -> (batch_size, n_channels, n_segs, d_model)
        self.proj = nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.params.seg_len, out_features=self.params.d_model,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
        )

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for `band_encoding`.
        nn.init.trunc_normal_(self.band_encoding, mean=0., std=1.)
        # Initialize weights for `positional_encoding`.
        nn.init.trunc_normal_(self.positional_encoding, mean=0., std=1.)
        # Initialize weights for `proj`.
        for module_i in self.proj.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `LinearTokenizer` to get the transformed tokens.

        Args:
            inputs: tuple - The input data, including [X,S,mask].

        Returns:
            T: (batch_size, n_channels, n_segs, d_model) - The transformed tokens.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, n_segs, seg_len); S - (batch_size, n_channels, n_segs, n_bands)
        # mask - (batch_size, n_channels, n_segs)
        X = inputs[0]; S = inputs[1]; mask = inputs[2]
        # Mask raw signal data `X` to get masked `X`.
        if mask is not None:
            # mask_emb - (batch_size, n_channels, n_segs, seg_len)
            mask_emb = self.mask_encoding[None,None,None,...].expand(*mask.shape, -1)
            # X - (batch_size, n_channels, n_segs, seg_len)
            X = X * (1. - mask[...,None].to(dtype=X.dtype)) + mask_emb * mask[...,None].to(dtype=X.dtype)
        # Project `X` to get `T`.
        # T_x - (batch_size, n_channels, n_segs, d_model)
        T_x = self.proj(X)
        # Use `F.softmax` to regularize `S`, then weight band embedding.
        # T_s - (batch_size, n_channels, n_segs, d_model)
        T_s = torch.einsum("hijk, kl->hijl", F.softmax(S, dim=-1), self.band_encoding)
        # Mask embeded spectrum data `T_s` to get masked `T_s`.
        if mask is not None:
            # power_mask_emb - (batch_size, n_channels, n_segs, d_model)
            power_mask_emb = self.power_mask_encoding[None,None,None,...].expand(*mask.shape, -1)
            # T_s - (batch_size, n_channels, n_segs, d_model)
            T_s = T_s * (1. - mask[...,None].to(dtype=T_s.dtype)) + power_mask_emb * mask[...,None].to(dtype=T_s.dtype)
        # Add `T_x` & `T_s` & `T_p` to get the final `T`.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = T_x + T_s + self.positional_encoding[:T_x.shape[-2],:]
        # Return the final `T`.
        return T

# def ConvTokenizer class
class ConvTokenizer(nn.Module):
    """
    Convolution tokenizer to transform the raw time series.
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
        # Initialize signal mask embedding.
        # mask_encoding - (seg_len,)
        mask_encoding = torch.zeros((self.params.seg_len,), dtype=torch.float32)
        self.mask_encoding = nn.Parameter(mask_encoding, requires_grad=False)
        # Initialize power mask embedding.
        # power_mask_encoding - (d_model,)
        power_mask_encoding = torch.zeros((self.params.d_model,), dtype=torch.float32)
        self.power_mask_encoding = nn.Parameter(power_mask_encoding, requires_grad=False)
        # Initialize band embedding.
        # band_encoding - (n_bands, d_model)
        band_encoding = torch.ones((self.params.n_bands, self.params.d_model), dtype=torch.float32)
        self.band_encoding = nn.Parameter(band_encoding, requires_grad=True)
        # Initialize position embedding.
        # positional_encoding - (seq_len, d_model)
        positional_encoding = torch.ones((self.params.max_segs, self.params.d_model), dtype=torch.float32)
        self.positional_encoding = nn.Parameter(positional_encoding, requires_grad=True)
        # Initialize convolution blocks.
        self.conv_blocks = nn.Sequential()
        # Add the convolution blocks.
        assert len(self.params.n_filters) > 0
        for conv_idx in range(len(self.params.n_filters)):
            # Initialize arguments for convolution block.
            n_channels = self.params.n_filters[conv_idx-1] if conv_idx > 0 else 1; n_filters = self.params.n_filters[conv_idx]
            kernel_size = self.params.kernel_sizes[conv_idx]; n_strides = self.params.n_strides[conv_idx]
            dropout = self.params.dropout_rates[conv_idx]; pool_size = self.params.pool_sizes[conv_idx]
            # Add the convolution block.
            self.conv_blocks.append(ConvTokenizer._make_conv_block(
                # Modified `_make_conv_block` function arguments.
                n_channels=n_channels, n_filters=n_filters, kernel_size=kernel_size,
                n_strides=n_strides, dropout=dropout, pool_size=pool_size
            ))
        # Add the average layer.
        self.conv_blocks.append(LambdaLayer(func=(lambda x: torch.mean(x, dim=-1))))
        # Add the projection layer.
        self.conv_blocks.append(nn.Linear(
            # Modified `Linear` layer parameters.
            in_features=self.params.n_filters[-1], out_features=self.params.d_model,
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
        # Initialize weights for `band_encoding`.
        nn.init.trunc_normal_(self.band_encoding, mean=0., std=1.)
        # Initialize weights for `positional_encoding`.
        nn.init.trunc_normal_(self.positional_encoding, mean=0., std=1.)
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    # def _make_conv_block func
    @staticmethod
    def _make_conv_block(n_channels, n_filters, kernel_size, n_strides, dropout=0.2, pool_size=2, **kwargs):
        """
        Make one convolution block, which contains [Conv1d,ReLU,Dropout,MaxPool1d].

        Args:
            n_channels: int - The number of input channels.
            n_filters: int - The number of convolution filters.
            kernel_size: int - The dimensions of convolution kernel.
            n_strides: int - The number of convolution strides.
            dropout: float - The dropout ratio after convolution.
            pool_size: int - The size of pooling kernel, `pool_stride = pool_size // 2`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            conv_block: nn.Module - The convolution block, which contains [Conv1d,ReLU,Dropout,MaxPool1d].
        """
        # Initialize the convolution block.
        conv_block = nn.Sequential(**kwargs)
        # Add `Conv1d` layer.
        padding = "same" if n_strides == 1 else _cal_conv_padding(kernel_size, dilation_rate=1)
        conv_block.append(nn.Sequential(
            nn.Conv1d(
                # Modified `Conv1d` layer parameters.
                in_channels=n_channels, out_channels=n_filters,
                kernel_size=kernel_size, stride=n_strides, padding=padding,
                # Default `Conv1d` layer parameters.
                dilation=1, groups=1, bias=True, padding_mode="zeros", device=None, dtype=None
            ),
        ))
        # Add `ReLU` layer.
        conv_block.append(nn.ReLU(inplace=False))
        # Add `Dropout` layer.
        if dropout > 0.: conv_block.append(nn.Dropout(p=dropout, inplace=False))
        # Add `MaxPool1d` layer.
        if pool_size > 1:
            assert pool_size % 2 == 0
            conv_block.append(nn.MaxPool1d(
                # Modified `MaxPool1d` layer parameters.
                kernel_size=pool_size, stride=(pool_size // 2),
                # Default `MaxPool1d` layer parameters.
                padding=0, dilation=1, return_indices=False, ceil_mode=False
            ))
        # Return the final `conv_block`.
        return conv_block

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `ConvTokenizer` to get the transformed tokens.

        Args:
            inputs: tuple - The input data, including [X,S,mask].

        Returns:
            T: (batch_size, n_channels, n_segs, d_model) - The transformed tokens.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, n_segs, seg_len); S - (batch_size, n_channels, n_segs, n_bands)
        # mask - (batch_size, n_channels, n_segs)
        X = inputs[0]; S = inputs[1]; mask = inputs[2]
        # Mask raw signal data `X` to get masked `X`.
        if mask is not None:
            # mask_emb - (batch_size, n_channels, n_segs, seg_len)
            mask_emb = self.mask_encoding[None,None,None,...].expand(*mask.shape, -1)
            # X - (batch_size, n_channels, n_segs, seg_len)
            X = X * (1. - mask[...,None].to(dtype=X.dtype)) + mask_emb * mask[...,None].to(dtype=X.dtype)
        # Convolve `X` to get `T`.
        # T_x - (batch_size, n_channels, n_segs, d_model)
        T_x = torch.reshape(self.conv_blocks(torch.reshape(X, shape=(-1, 1, X.shape[-1]))), shape=(*X.shape[:-1], -1))
        # Use `F.softmax` to regularize `S`, then weight band embedding.
        # T_s - (batch_size, n_channels, n_segs, d_model)
        T_s = torch.einsum("hijk, kl->hijl", F.softmax(S, dim=-1), self.band_encoding)
        # Mask embeded spectrum data `T_s` to get masked `T_s`.
        if mask is not None:
            # power_mask_emb - (batch_size, n_channels, n_segs, d_model)
            power_mask_emb = self.power_mask_encoding[None,None,None,...].expand(*mask.shape, -1)
            # T_s - (batch_size, n_channels, n_segs, d_model)
            T_s = T_s * (1. - mask[...,None].to(dtype=T_s.dtype)) + power_mask_emb * mask[...,None].to(dtype=T_s.dtype)
        # Add `T_x` & `T_s` & `T_p` to get the final `T`.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = T_x + T_s + self.positional_encoding
        # Return the final `T`.
        return T

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
        # Initialize convolution blocks.
        self.conv_blocks = nn.Sequential()
        # Add the convolution blocks.
        for conv_idx in range(len(self.params.n_filters)):
            # Initialize arguments for convolution block.
            n_channels = self.params.n_filters[conv_idx-1] if conv_idx > 0 else 1; n_filters = self.params.n_filters[conv_idx]
            kernel_size = self.params.kernel_sizes[conv_idx]; n_strides = self.params.n_strides[conv_idx]
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
            nn.Conv1d(
                # Modified `Conv1d` layer parameters.
                in_channels=n_channels, out_channels=n_filters,
                kernel_size=kernel_size, stride=n_strides, padding=padding,
                # Default `Conv1d` layer parameters.
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
        # X - (batch_size * n_channels * n_segs, 1, seg_len)
        X = torch.unsqueeze(torch.reshape(X, shape=(-1, X.shape[-1])), dim=-2)
        # Get the convolved tokens for each channel separately.
        # T - (batch_size * n_channels * n_segs, d_model)
        T = torch.flatten(self.conv_blocks(X), start_dim=1, end_dim=-1)
        # Reshape back the convolved tokens.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = torch.reshape(T, shape=(batch_size, n_channels, -1, T.shape[-1]))
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
    batch_size = 32; seq_len = 800; n_channels = 10; d_model = 768
    seg_len = 200; n_bands = 8; n_segs = (seq_len // seg_len)

    ## Forward LinearTokenizer.
    # Initialize params.
    params_inst = DotDict({
        # The maximum number of time segments.
        "max_segs": max_segs,
        # The number of frequency bands.
        "n_bands": n_bands,
        # The length of element segment.
        "seg_len": seg_len,
        # The dimensions of model embedding.
        "d_model": d_model,
    })
    # Initialize input `X` & `S` & `mask`.
    # X - (batch_size, n_channels, n_segs, seg_len); S - (batch_size, n_channels, n_segs, n_bands)
    # mask - (batch_size, n_channels, n_segs)
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    S = torch.rand((batch_size, n_channels, n_segs, n_bands), dtype=torch.float32)
    mask = (torch.rand((batch_size, n_channels, n_segs), dtype=torch.float32) < 0.5)
    # Instantiate LinearTokenizer.
    tokenizer_linear_inst = LinearTokenizer(params=params_inst)
    # Forward layers in `tokenizer_linear_inst`.
    # T - (batch_size, n_channels, n_segs, d_model)
    T = tokenizer_linear_inst((X, S, mask))
    ## Forward ConvTokenizer.
    # Initialize params.
    params_inst = DotDict({
        # The maximum number of time segments.
        "max_segs": max_segs,
        # The number of frequency bands.
        "n_bands": n_bands,
        # The length of element segment.
        "seg_len": seg_len,
        # The number of convolution filters.
        "n_filters": [128, 128, 128],
        # The dimensions of convolution kernel.
        "kernel_sizes": [19, 3, 3],
        # The number of convolution strides.
        "n_strides": [10, 1, 1],
        # The dropout ratio after convolution.
        "dropout_rates": [0.2, 0.2, 0.2],
        # The size of pooling kernel.
        "pool_sizes": [1, 1, 1],
        # The dimensions of model embedding.
        "d_model": d_model,
    })
    # Initialize input `X` & `S` & `mask`.
    # X - (batch_size, n_channels, n_segs, seg_len); S - (batch_size, n_channels, n_segs, n_bands)
    # mask - (batch_size, n_channels, n_segs)
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    S = torch.rand((batch_size, n_channels, n_segs, n_bands), dtype=torch.float32)
    mask = (torch.rand((batch_size, n_channels, n_segs), dtype=torch.float32) < 0.5)
    # Instantiate ConvTokenizer.
    tokenizer_conv_inst = ConvTokenizer(params=params_inst)
    # Forward layers in `tokenizer_conv_inst`.
    # T - (batch_size, n_channels, n_segs, d_model)
    T = tokenizer_conv_inst((X, S, mask))
    ## Forward PatchTokenizer.
    # Initialize params.
    params_inst = DotDict({
        # The length of patch segment.
        "seg_len": seg_len,
        # The number of filters of each convolution block.
        "n_filters": [16, 16, 16],
        # The size of kernel of each deconvolution block.
        "kernel_sizes": [19, 19, 3],
        # The number of strides of each deconvolution block.
        "n_strides": [10, 10, 1],
    })
    # Initialize input `X`.
    # X - (batch_size, n_channels, n_segs, seg_len)
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    # Instantiate PatchTokenizer.
    tokenizer_patch_inst = PatchTokenizer(params=params_inst)
    # Forward layers in `tokenizer_patch_inst`.
    # T - (batch_size, n_channels, n_segs, d_model)
    T = tokenizer_patch_inst(X)

