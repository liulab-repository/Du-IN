#!/usr/bin/env python3
"""
Created on 20:38, Jan. 15th, 2024

@author: Norbert Zheng
"""
import torch
import copy as cp
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
from models.duin.layers.LambdaLayer import LambdaLayer
import utils.model.torch

__all__ = [
    "KmeansVectorQuantizer",
]

# def KmeansVectorQuantizer class
class KmeansVectorQuantizer(nn.Module):
    """
    Vector Quantizer using straight pass-through estimator (i.e., K-means), which is proposed by Van et al. 2017.

    [1] Van Den Oord A, Vinyals O. Neural discrete representation learning[J].
        Advances in neural information processing systems, 2017, 30.
    """

    def __init__(self, d_model, codex_size, d_codex, n_groups, share_group=False, beta=0.25, use_norm=False, **kwargs):
        """
        Initialize `KmeansVectorQuantizer` object.

        Args:
            d_model: int - The dimensions of model embedding.
            codex_size: int - The number of discrete embeddings per group.
            d_codex: int - The dimensions of codex embedding.
            n_groups: int - The number of groups for vector quantization.
            share_group: bool - The flag that indicates whether all groups share one codex.
            beta: float - The scale factor of commitment loss (which is a part of vq loss).
            use_norm: bool - The flag that indicates whether use l2-normalization before distance calculation.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(KmeansVectorQuantizer, self).__init__(**kwargs)

        # Initialize parameters.
        assert d_codex % n_groups == 0, (
            "ERROR: The dimensions of codex ({:d}) should be divisible by the number of"+\
            " groups ({:d}) for concatenation in layers.KmeansVectorQuantizer."
        ).format(d_codex, n_groups)
        self.d_model = d_model; self.codex_size = codex_size; self.d_codex = d_codex
        self.n_groups = n_groups; self.share_group = share_group; self.beta = beta; self.use_norm = use_norm

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
        ## Construct the pre-projection layer.
        # Initialize the pre-projection layer.
        # pre_proj - (batch_size, *, d_model) -> (batch_size, *, d_codex)
        self.pre_proj = nn.Sequential()
        # Add group-wise `Conv1d` layer.
        self.pre_proj.append(nn.Sequential(
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
            nn.Conv1d(
                # Modified `Conv1d` layer parameters.
                in_channels=self.d_model, out_channels=self.d_codex,
                kernel_size=1, padding="same", groups=self.n_groups, bias=False,
                # Default `Conv1d` layer parameters.
                stride=1, dilation=1, padding_mode="zeros", device=None, dtype=None
            ),
            nn.GroupNorm(
                # Modified `GroupNorm` layer parameters.
                num_groups=self.n_groups, num_channels=self.d_codex,
                # Default `GroupNorm` layer parameters.
                eps=1e-5, affine=True, device=None, dtype=None
            ),
            LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1]))),
        ))
        ## Construct the discrete codex.
        # Initialize discrete codex according to `codex_size` & `d_codex`.
        # Note: As we use Gumbel Softmax to get the corresponding codex indices, the norm of init value is not limited.
        # codex - (n_groups, codex_size, d_codex // n_groups)
        codex_shape = (self.n_groups, self.codex_size, self.d_codex // self.n_groups)\
            if not self.share_group else (1, self.codex_size, self.d_codex // self.n_groups)
        codex = torch.ones(codex_shape, dtype=torch.float32); self.codex = nn.Parameter(codex, requires_grad=True)
        # Initialize the codex statistics.
        # codex_counts - (codex_size,)
        codex_counts = torch.ones(*codex_shape[:-1], dtype=torch.float32)
        self.codex_counts = nn.Parameter(codex_counts, requires_grad=False)
        ## Construct the layer-norm layer.
        # Initialize the layer-norm layer.
        self.layernorm = nn.LayerNorm(
            # Modified `LayerNorm` layer parameters.
            normalized_shape=(self.d_codex,),
            # Default `LayerNorm` layer parameters.
            eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None
        )
        ## Construct the post-projection layer.
        # Initialize the post-projection layer.
        # post_proj - (batch_size, *, d_codex) -> (batch_size, *, d_model)
        self.post_proj = nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.d_codex, out_features=self.d_model,
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
        # Initialize weights for `codex`.
        nn.init.normal_(self.codex, mean=0., std=1.)
        # Initialize weights for `pre_proj`.
        # TODO: As `Conv1d` acts like a group-wise linear projection, enable initialization trick.
        # Initialize weights for `layernorm`.
        for module_i in self.layernorm.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)
        # Initialize weights for `post_proj`.
        for module_i in self.post_proj.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    state funcs
    """
    # def init_counts func
    def init_counts(self):
        """
        Initialize codex counts.

        Args:
            None

        Returns:
            None
        """
        self.codex_counts.data.copy_(torch.zeros(self.codex_counts.shape))

    # def update_counts func
    def update_counts(self, codex_probs, decay=0.99):
        """
        Update codex counts.

        Args:
            codex_probs: (n_samples, n_groups, codex_size) - The one-hot probability assigned to each codex.

        Returns:
            None
        """
        # Get the usage count of each codex.
        # codex_counts - (n_groups, codex_size) or (1, codex_size)
        codex_counts = torch.sum(codex_probs, dim=0)
        if self.share_group: codex_counts = torch.sum(codex_counts, dim=0, keepdim=True)
        utils.model.torch.all_reduce(codex_counts)
        # Update the codex counts.
        ema_inplace(self.codex_counts, codex_counts, decay=decay, use_norm=False)

    # def get_counts func
    def get_counts(self):
        """
        Get codex counts.

        Args:
            None

        Returns:
            counts: (codex_size,) - The pseudo-counts of each codex.
        """
        return self.codex_counts.detach().cpu().numpy()

    """
    network funcs
    """
    # def forward func
    def forward(self, Z):
        """
        Forward layers in `KmeansVectorQuantizer` to get vector-quantized embeddings.

        Args:
            Z: (batch_size, *, d_model) - The input embeddings.

        Returns:
            Z_q: (batch_size, *, d_model) - The vector-quantized embeddings.
            loss: torch.float32 - The vector-quantizer loss.
            codex_probs: (batch_size, *, n_groups, codex_size) - The one-hot probabilities of input embeddings.
        """
        # Forward the pre-projection layer to get the embeddings in codex space.
        # Z_e - (batch_size, *, n_groups, d_codex // n_groups)
        Z_e = torch.reshape(self.pre_proj(Z), shape=(*Z.shape[:-1], self.n_groups, self.d_codex // self.n_groups))
        # Get l2-normalized `Z_e` & `codex`.
        # Z_e - (batch_size, *, n_groups, d_codex // n_groups); codex - (n_groups, codex_size, d_codex // n_groups)
        codex_norm = np.sqrt(self.d_codex // self.n_groups)
        Z_e = utils.model.torch.normalize(Z_e, p=2., dim=-1, eps=1e-12) if self.use_norm else Z_e
        codex = utils.model.torch.normalize(self.codex, p=2., dim=-1, eps=1e-12) if self.use_norm else self.codex
        # Calculate the distances between `Z_e` and `codex`.
        # I.e., (z - e) ^ 2 = (z ^ 2) + (e ^ 2) - (2 * e * z)
        # codex_dists - (batch_size, *, n_groups, codex_size)
        codex_dists = (
            torch.sum(torch.reshape(Z_e, shape=(-1, *Z_e.shape[-2:])) ** 2, dim=-1, keepdim=True)
            + torch.unsqueeze(torch.sum(codex ** 2, dim=-1), dim=0)
            - 2. * torch.permute(torch.matmul(
                torch.permute(torch.reshape(Z_e, shape=(-1, *Z_e.shape[-2:])), dims=[1,0,2]),
                torch.permute(codex, dims=[0,2,1])
            ), dims=[1,0,2])
        )
        codex_dists = torch.reshape(codex_dists, shape=(*Z_e.shape[:-1], -1))
        # Get the corresponding indices & ont-hot probs of input embeddings.
        # codex_idxs - (batch_size, *, n_groups, 1)
        codex_idxs = torch.argmin(codex_dists, dim=-1, keepdim=True)
        # codex_probs - (batch_size, *, n_groups, codex_size)
        codex_probs = torch.zeros_like(codex_dists, dtype=codex_dists.dtype).scatter_(dim=-1, index=codex_idxs, value=1.)
        # Get the vector-quantized embeddings.
        # Note: We use `codex_probs`, instead of `codex_idxs`, to get `Z_q`!
        # This operation will make a gradient flow between `Z_q` and `codex`.
        # Z_q - (batch_size, *, n_groups, d_codex // n_groups)
        Z_q = torch.reshape(torch.permute(torch.matmul(
            torch.permute(torch.reshape(codex_probs, shape=(-1, *codex_probs.shape[-2:])), dims=[1,0,2]), codex
        ), dims=[1,0,2]), shape=Z_e.shape)
        # Calculate the vector-quantizer loss.
        # loss - torch.float32
        loss = self.loss(Z_e, Z_q)
        # Re-parameterize `Z_q` to allow gradients to flow back to other layers.
        Z_q = Z_q.detach() + (Z_e - Z_e.detach())
        # Concatenate codex embeddings from different groups.
        # Z_q - (batch_size, *, d_codex)
        Z_q = torch.reshape(Z_q, shape=(*Z_q.shape[:-2], -1))
        # Calculate perplexity to measure the codex usage.
        # codex_usage - (n_groups, codex_size) or (1, codex_size)
        codex_usage = torch.mean(torch.reshape(codex_probs, shape=(-1, *codex_probs.shape[-2:])), dim=0)
        codex_usage = torch.mean(codex_usage, dim=0, keepdim=True) if self.share_group else codex_usage
        # perplexity - torch.float32
        perplexity = torch.mean(torch.exp(-torch.sum(codex_usage * torch.log(codex_usage + 1e-12), dim=-1)))
        # Update codex statistics.
        self.update_counts(torch.reshape(codex_probs, shape=(-1, *codex_probs.shape[-2:])), decay=0.99)
        # Normalize the vector-quantized embeddings to align the scale.
        Z_q = self.layernorm(Z_q)
        # Forward the post-projection layer to get the embeddings in embedding space.
        # Z_q - (batch_size, *, d_model)
        Z_q = self.post_proj(Z_q)
        # Return the final `Z_q` & `loss` & `codex_probs`.
        return Z_q, loss, codex_probs

    """
    loss funcs
    """
    # def loss func
    def loss(self, Z_e, Z_q):
        """
        Calculate the loss in `KmeansVectorQuantizer`.

        Args:
            Z_e: (batch_size, *, n_groups, d_codex // n_groups) - The input embeddings.
            Z_q: (batch_size, *, n_groups, d_codex // n_groups) - The vector-quantized embeddings.

        Returns:
            loss: torch.float - The vector-quantizer loss.
        """
        # Calculate commitment loss and codebook loss.
        # loss_* - torch.float32
        loss_commitment = torch.mean(torch.mean((Z_e - Z_q.detach()) ** 2, dim=-1))
        loss_codebook = torch.mean(torch.mean((Z_q - Z_e.detach()) ** 2, dim=-1))
        # Calculate teh final vector-quantizer loss.
        # loss - torch.float32
        loss = self.beta * loss_commitment + loss_codebook
        # Return the final `loss`.
        return loss

"""
tool funcs
"""
# def ema_inplace func
def ema_inplace(weight, value, decay=0.99, use_norm=False):
    """
    Update weight with EMA (i.e., Exponential Moving Average).

    Args:
        weight: torch.Parameter - The variable to be updated.
        value: torch.Tensor - The update value.
        decay: float - The exponential update factor.
        use_norm: bool - The flag that indicates whether use l2-normalization.

    Returns:
        None
    """
    # Execute EMA (i.e., Exponential Moving Average).
    weight.data.mul_(decay).add_(value, alpha=(1. - decay))
    # L2-normalize the updated value.
    if use_norm: weight.data.copy_(l2norm(weight.data))

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128
    codex_size = 8192; d_codex = 256; n_groups = 4; share_group = False; beta = 0.25; use_norm = True

    ## Forward KmeansVectorQuantizer.
    # Initialize input `Z`.
    # Z - (batch_size, emb_len, d_model)
    Z = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate KmeansVectorQuantizer.
    kvq_inst = KmeansVectorQuantizer(d_model=d_model, codex_size=codex_size,
        d_codex=d_codex, n_groups=n_groups, share_group=share_group, beta=beta, use_norm=use_norm
    )
    # Forward layers in `kvq_inst`.
    # Z_q - (batch_size, emb_len, d_model); loss - torch.float32; codex_probs - (batch_size, emb_len, n_groups, codex_size)
    Z_q, loss, codex_probs = kvq_inst(Z)

