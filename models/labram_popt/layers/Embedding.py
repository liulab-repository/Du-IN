#!/usr/bin/env python3
"""
Created on 19:38, Jan. 20th, 2024

@author: Norbert Zheng
"""
import torch
import numpy as np
import torch.nn as nn
from einops import repeat
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from LambdaLayer import LambdaLayer
else:
    from models.labram_popt.layers.LambdaLayer import LambdaLayer

__all__ = [
    # Temporal Embeddings.
    "TimeEmbedding",
    # Spatial Embeddings.
    "ChannelEmbedding",
    # Position Embeddings.
    "RotaryEmbedding",
]

"""
temporal embeddings
"""
# def TimeEmbedding class
class TimeEmbedding(nn.Module):
    """
    Time embedding used to inject time information.
    """

    def __init__(self, d_model, max_segs, mode="normal", **kwargs):
        """
        Initialize `TimeEmbedding` object.

        Args:
            d_model: int - The dimensions of model embedding.
            max_segs: int - The maximum number of segments.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(TimeEmbedding, self).__init__(**kwargs)

        # Initialize parameters.
        assert d_model % 2 == 0, (
            "ERROR: The dimensions of model embedding ({:d}) must be a multiples of 2 in layers.TimeEmbedding."
        ).format(d_model)
        assert mode in [None, "zero", "zeros", "normal", "uniform", "sincos"], (
            "ERROR: Get unknown time embedding mode {} in layers.TimeEmbedding."
        ).format(mode)
        self.d_model = d_model; self.max_segs = max_segs; self.mode = mode

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
        # Initialize model architecture according to `mode`.
        getattr(self, "_init_model_{}".format(str(self.mode).lower()))()

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

    # def _init_model_none func
    def _init_model_none(self):
        """
        Initialize the `None` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize the time embeddings.
        # time_encodings - (max_segs, d_model)
        time_encodings = torch.ones((self.max_segs, self.d_model), dtype=torch.float32)
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=False)
        # Initialize weights for `time_encodings`.
        nn.init.uniform_(self.time_encodings, a=-0.02, b=0.02)

    # def _init_model_zero func
    def _init_model_zero(self):
        """
        Initialize the `zero` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize the time embeddings.
        # time_encodings - (max_segs, 1)
        time_encodings = torch.ones((self.max_segs, 1), dtype=torch.float32)
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)
        # Initialize weights for `time_encodings`.
        nn.init.uniform_(self.time_encodings, a=-0.02, b=0.02)

    # def _init_model_zeros func
    def _init_model_zeros(self):
        """
        Initialize the `zeros` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize the time embeddings.
        # time_encodings - (max_segs, d_model)
        time_encodings = torch.ones((self.max_segs, self.d_model), dtype=torch.float32)
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)
        # Initialize weights for `time_encodings`.
        nn.init.uniform_(self.time_encodings, a=-0.02, b=0.02)

    # def _init_model_normal func
    def _init_model_normal(self):
        """
        Initialize the `normal` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize the time embeddings.
        # time_encodings - (max_segs, d_model)
        time_encodings = torch.ones((self.max_segs, self.d_model), dtype=torch.float32)
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)
        # Initialize weights for `time_encodings`.
        nn.init.trunc_normal_(self.time_encodings, mean=0., std=1.)

    # def _init_model_uniform func
    def _init_model_uniform(self):
        """
        Initialize the `uniform` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Initialize the time embeddings.
        # time_encodings - (max_segs, d_model)
        time_encodings = torch.ones((self.max_segs, self.d_model), dtype=torch.float32)
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=True)
        # Initialize weights for `time_encodings`.
        nn.init.uniform_(self.time_encodings, a=0., b=1.)

    # def _init_model_sincos func
    def _init_model_sincos(self):
        """
        Initialize the `sincos` model architecture.

        Args:
            None

        Returns:
            None
        """
        # Empty `time_encodings` matrix.
        # time_encodings - (max_segs, d_model)
        time_encodings = np.zeros((self.max_segs, self.d_model), dtype=np.float32)
        # Get the indexes of available time steps (i.e. within `max_segs`).
        # time_idxs - (max_segs, 1)
        time_idxs = np.expand_dims(np.arange(0, self.max_segs, dtype=np.float32), axis=-1)
        # Get the divide term, i.e. $(1e4)*exp(\frac{-2i}{d_model})$.
        # div_term - (d_model//2,)
        div_term = np.exp(np.arange(0, self.d_model, 2, dtype=np.float32) * -(np.log(1e4) / self.d_model))
        # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$.
        time_encodings[:,0::2] = np.sin(time_idxs * div_term)
        # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
        time_encodings[:,1::2] = np.cos(time_idxs * div_term)
        # Use `register_buffer` to ensure that `time_encodings` will be saved to `model.state_dict()`.
        # The only difference between `buffer` and `nn.Parameter` is that `buffer` can calculate its gradient,
        # but `buffer` cannot be updated according to `optimizer`.
        time_encodings = torch.tensor(time_encodings, dtype=torch.float32)
        self.time_encodings = nn.Parameter(time_encodings, requires_grad=False)

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `TimeEmbedding` to get the time-embedded embeddings.

        Args:
            emb: (batch_size, n_channels, n_segs, d_model) - The sequence of embeddings.

        Returns:
            emb: (batch_size, n_channels, n_segs, d_model) - The sequence of time-embedded embeddings.
        """
        # Get the time embeddings `time_emb`.
        # time_emb - (n_segs, d_model)
        time_emb = self.time_encodings[:emb.shape[-2],:]
        # Add `time_emb` to `emb` to get the time-embedded embeddings.
        emb = emb + time_emb
        # Return the final `emb`.
        return emb

"""
spatial embeddings
"""
# def ChannelEmbedding class
class ChannelEmbedding(nn.Module):
    """
    Channel embedding used to inject spatial information.
    """

    def __init__(self, d_model, n_axes, **kwargs):
        """
        Initialize `ChannelEmbedding` object.

        Args:
            d_model: int - The dimensions of model embedding.
            n_axes: int - The number of coordinate axes.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(ChannelEmbedding, self).__init__(**kwargs)

        # Initialize parameters.
        assert (d_model % 2 == 0), (
            "ERROR: The dimensions of model embedding ({:d}) must be a multiples of 2 in layers.ChannelEmbedding."
        ).format(d_model)
        self.d_model = d_model; self.n_axes = n_axes

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
        ## Construct channel embedding layer.
        # Define axis-wise channel embedding function.
        def _emb_axis(d_axis, axis_pos):
            """
            Get the axis-wise channel embedding according to each axis.

            Args:
                d_axis: int - The dimensions of axis-wise model embedding.
                axis_pos: (*, n_channels) - The position along that axis.

            Returns:
                axis_emb: (*, n_channels, d_axis) - The axis-wise channel embedding.
            """
            # Empty `time_encodings` matrix.
            # axis_emb - (*, n_channels, d_model)
            axis_emb = torch.zeros((*axis_pos.shape, d_axis), dtype=torch.float32)
            # Get the divide term, i.e. $(1e4)*exp(\frac{-2i}{d_model})$.
            # div_term - (d_axis//2,)
            div_term = torch.exp(torch.arange(0, d_axis, 2, dtype=torch.float32) * -(np.log(1e4) / d_axis))
            # Transfer the generated tensors to the specified device.
            axis_emb = axis_emb.to(device=axis_pos.device); div_term = div_term.to(device=axis_pos.device)
            # $PE_{p,2i} = sin\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$.
            axis_emb[...,0::2] = torch.sin(torch.unsqueeze(axis_pos, dim=-1) * div_term)
            # $PE_{p,2i + 1} = cos\Bigg(\frac{p}{10000^{\frac{2i}{d_{model}}}}\Bigg)$
            axis_emb[...,1::2] = torch.cos(torch.unsqueeze(axis_pos, dim=-1) * div_term)
            # Return the final `axis_emb`.
            return axis_emb
        # Initialize channel embedding layer.
        # emb_channel - (batch_size, *, n_channels, n_axes) -> (batch_size, *, n_channels, d_model)
        d_axis = int(np.ceil(self.d_model / self.n_axes)); d_axis = d_axis if (d_axis % 2 == 0) else (d_axis + 1)
        self.emb_channel = nn.Sequential(
            LambdaLayer(func=(lambda x: torch.concat([_emb_axis(
                d_axis=d_axis, axis_pos=torch.squeeze(x_i, dim=-1)
            ) for x_i in torch.split(x, split_size_or_sections=1, dim=-1)], dim=-1))),
            LambdaLayer(func=(lambda x: x[...,:self.d_model])),
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
        pass

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `ChannelEmbedding` to get the channel-embedded embeddings.

        Args:
            inputs: tuple - The input data, including [emb,channel_pos].

        Returns:
            emb: (batch_size, n_segs, n_channels, d_model) - The sequence of channel-embedded embeddings.
        """
        # Initialize `emb` & `channel_pos` from `inputs`.
        # emb - (batch_size, *, n_channels, d_model); channel_pos - (batch_size, *, n_channels, n_axes)
        emb = inputs[0]; channel_pos = inputs[1]
        # Get the channel embeddings `channel_emb`.
        # channel_emb - (batch_size, 1, n_channels, d_model)
        channel_emb = self.emb_channel(channel_pos)[:,None,...]
        # Add `channel_emb` to `emb` to get the channel-embedded embeddings.
        emb = emb + channel_emb
        # Return the final `emb`.
        return emb

"""
position embeddings
"""
# def RotaryEmbedding class
class RotaryEmbedding(nn.Module):
    """
    Rotary embedding used to inject relative position information.
    """

    def __init__(self, d_model, theta=1e4, **kwargs):
        """
        Initialize `RotaryEmbedding` object.

        Args:
            d_model: int - The dimensions of model embedding.
            theta: float - The power base of rotation angle.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(RotaryEmbedding, self).__init__(**kwargs)

        # Initialize parameters.
        assert d_model % 2 == 0, (
            "ERROR: The dimensions of model embedding ({:d}) must be a multiples of 2 in layers.RotaryEmbedding."
        ).format(d_model)
        self.d_model = d_model; self.theta = theta

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
        # TODO: Support theta rescale according to `theta_rescale_factor`, which is proposed by reddit user `bloc97`.
        # To rescale rotary embeddings to longer sequence length without fine-tuning has some connection to NTK literature
        # (https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/).
        # >>> self.theta *= theta_rescale_factor ** (self.d_model / (self.d_model - 2))
        # Initialize `freqs`, i.e., the rotation angles, only support language `freqs` currently.
        # TODO: Support `freqs` for different modalities (e.g., language, pixel, constant) as in `lucidrains`'s implementation
        # (https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py).
        # >>> if self.freqs is not None:
        # >>>     freqs = self.freqs
        # >>> elif self.freq_type == "language":
        # >>>     freqs = 1. / (self.theta ** (tf.range(0, self.d_model, 2)[:(self.d_model // 2)] / self.d_model))
        # >>> elif self.freq_type == "pixel":
        # >>>     freqs = tf.linspace(1., self.max_freq / 2, self.d_model // 2) * np.pi
        # >>> elif self.freq_type == "constant":
        # >>>     freqs = tf.ones((self.n_freqs,))
        # >>> else:
        # >>>     raise ValueError("ERROR: Unknown frequency type {}.".format(self.freq_type))
        # freqs - (d_model // 2,)
        freqs = 1. / (self.theta ** (np.arange(0, self.d_model, 2)[:(self.d_model // 2)] / self.d_model))
        # TODO: Support trainable `freqs` according to trainable flag.
        self.freqs = nn.Parameter(torch.tensor(freqs, dtype=torch.float32), requires_grad=False)

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
    def forward(self, emb):
        """
        Forward layers in `RotaryEmbedding` to get the rotary-embedded embeddings.

        Args:
            emb: (batch_size, emb_len, d_model) - The sequence of elements.

        Returns:
            emb: (batch_size, emb_len, d_model) - The sequence of rotary-embedded embeddings.
        """
        # Initialize the indices of available positions.
        # position_idxs - (emb_len,)
        position_idxs = torch.arange(emb.shape[-2], dtype=emb.dtype).to(device=emb.device)
        # Construct `freqs` according to `freqs` & `position_idxs`.
        # freqs - (emb_len, d_model // 2)
        freqs = torch.einsum("..., f -> ... f", position_idxs, self.freqs)
        # freqs - (emb_len, d_model)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        # Calculate the rotary-embedded embeddings.
        # emb - (batch_size, *, emb_len, d_model)
        emb = torch.cos(freqs) * emb + torch.sin(freqs) * torch.reshape((
            torch.flip(torch.reshape(emb, shape=(*emb.shape[:-1], emb.shape[-1] // 2, 2)), dims=[-1]) *\
            torch.tensor([-1., 1.], dtype=emb.dtype).to(device=emb.device)
        ), shape=emb.shape)
        # Return the final `emb`.
        return emb

"""
plot funcs
"""
# def plot_rope_decay func
def plot_rope_decay(freqs, max_len=80, path_img=None):
    """
    Plot the long-term decay of RoPE.

    Args:
        freqs: (d_model // 2,) - The rotation angles.
        max_len: int - The maximum length of sequence.
        path_img: str - The path of images to save.

    Returns:
        None
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    # Construct the decay function according to `freqs`.
    func = lambda x: np.sum([np.sum(np.cos(x * freqs[:freq_idx+1])) for freq_idx in range(len(freqs))]) / len(freqs)
    # Generate data for the corresponding function.
    # x - (2 * max_len - 1,); y - (2 * max_len - 1,)
    x = np.linspace(start=-(max_len - 1), stop=(max_len - 1), num=(2 * max_len - 1))
    y = np.array([func(x_i) for x_i in x], dtype=x.dtype)
    # Create a line plot.
    sns.set(style="whitegrid"); sns.lineplot(x=x, y=y)
    # Add labels and a title.
    plt.xlabel("Relative Distance"); plt.ylabel("Relative Upper Bound"); plt.title("Long-term Decay of RoPE")
    # Show or save the plot.
    if path_img is None:
        plt.show()
    else:
        plt.savefig(os.path.join(path_img, "rope_decay.png"))
    plt.close("all")

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; n_channels = 64; n_segs = 8; d_model = 128; emb_len = n_channels * n_segs; rot_theta = 1e2
    path_img = os.path.join(os.getcwd(), "__image__")
    if not os.path.exists(path_img): os.makedirs(path_img)

    ## Forward temporal embeddings.
    # Initialize embedded input `emb`.
    # emb - (batch_size, n_channels, n_segs, d_model)
    emb = torch.rand((batch_size, n_channels, n_segs, d_model), dtype=torch.float32)
    # Instantiate `TimeEmbedding`.
    emb_time_inst = TimeEmbedding(d_model=d_model, max_segs=n_segs)
    # Forward layers in `TimeEmbedding`.
    # emb - (batch_size, n_channels, n_segs, d_model)
    emb = emb_time_inst(emb)
    ## Forward spatial embeddings.
    # Initialize embedded input `emb`.
    # emb - (batch_size, n_segs, n_channels, d_model)
    emb = torch.permute(torch.rand((batch_size, n_channels, n_segs, d_model), dtype=torch.float32), dims=[0,2,1,3])
    # Instantiate `ChannelEmbedding`.
    emb_channel_inst = ChannelEmbedding(d_model=d_model, n_channels=n_channels)
    # Forward layers in `ChannelEmbedding`.
    # emb - (batch_size, n_segs, n_channels, d_model)
    emb = emb_channel_inst(emb)
    ## Forward position embeddings.
    # Initialize embedded input `emb`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate `RotaryEmbedding`.
    emb_rot_inst = RotaryEmbedding(d_model=d_model, theta=rot_theta)
    # Forward layers in `RotaryEmbedding`.
    # emb - (batch_size, emb_len, d_model)
    emb = emb_rot_inst(emb)
    # Plot RoPE decay of `RotaryEmbedding`.
    plot_rope_decay(emb_rot_inst.freqs.numpy(), max_len=emb_len, path_img=path_img)

