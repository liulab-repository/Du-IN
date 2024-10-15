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
from models.labram.layers.LambdaLayer import LambdaLayer
import utils.model.torch

__all__ = [
    "GumbelVectorQuantizer",
]

# def GumbelVectorQuantizer class
class GumbelVectorQuantizer(nn.Module):
    """
    Vector Quantizer using Gumbel Softmax, which is proposed by Baevski et al. 2019.

    [1] Baevski A, Schneider S, Auli M. vq-wav2vec: Self-supervised learning of discrete speech representations[J].
        arXiv preprint arXiv:1910.05453, 2019.
    """

    def __init__(self, d_model, codex_size, d_codex, n_groups, share_group=False, use_hard=True,
        d_hidden=[], activation=nn.GELU(approximate="none"), tau_factors=(1., 1., 1.), **kwargs):
        """
        Initialize `GumbelVectorQuantizer` object.

        Args:
            d_model: int - The dimensions of model embedding.
            codex_size: int - The number of discrete embeddings per group.
            d_codex: int - The dimensions of codex embedding.
            n_groups: int - The number of groups for vector quantization.
            share_group: bool - The flag that indicates whether all groups share one codex.
            use_hard: bool - The flag that indicates whether use hard Gumbel Softmax.
            d_hidden: list - The dimensions of hidden layers between input and codex.
            activation: nn.Module - The activation module used in hidden layers.
            tau_factors: (3[tuple],) - The temperature parameters, including (tau_max, tau_min, tau_decay).
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(GumbelVectorQuantizer, self).__init__(**kwargs)

        # Initialize parameters.
        assert d_codex % n_groups == 0, (
            "ERROR: The dimensions of codex ({:d}) should be divisible by the number of"+\
            " groups ({:d}) for concatenation in layers.GumbelVectorQuantizer."
        ).format(d_codex, n_groups); assert len(tau_factors) == 3
        self.d_model = d_model; self.codex_size = codex_size; self.d_codex = d_codex; self.n_groups = n_groups
        self.share_group = share_group; self.use_hard = use_hard; self.d_hidden = d_hidden; self.activation = activation
        self.tau_max = tau_factors[0]; self.tau_min = tau_factors[1]; self.tau_decay = tau_factors[2]
        # Update parameters.
        self.update_params(iteration=0)

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
        self.pre_proj = nn.Sequential()
        # Add the hidden layers.
        for hidden_idx in range(len(self.d_hidden)):
            self.pre_proj.append(nn.Sequential(
                nn.Linear(
                    # Modified `Linear` layer parameters.
                    in_features=(self.d_hidden[hidden_idx-1] if hidden_idx > 0 else self.d_model),
                    out_features=self.d_hidden[hidden_idx],
                    # Default `Linear` layer parameters.
                    bias=True, device=None, dtype=None
                ),
                self.activation,
            ))
        # Add the final logit-projection layer.
        self.pre_proj.append(nn.Linear(
            # Modified `Linear` layer parameters.
            in_features=(self.d_hidden[-1] if len(self.d_hidden) > 0 else self.d_model),
            out_features=(self.n_groups * self.codex_size),
            # Default `Linear` layer parameters.
            bias=True, device=None, dtype=None
        ))
        ## Construct the discrete codex.
        # Initialize discrete codex according to `codex_size` & `d_codex`.
        # Note: As we use Gumbel Softmax to get the corresponding codex indices, the norm of init value is not limited.
        # codex - (n_groups, codex_size, d_codex // n_groups)
        codex_shape = (self.n_groups, self.codex_size, self.d_codex // self.n_groups)\
            if not self.share_group else (1, self.codex_size, self.d_codex // self.n_groups)
        codex = torch.rand(codex_shape, dtype=torch.float32); self.codex = nn.Parameter(codex, requires_grad=True)
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
        nn.init.uniform_(self.codex, a=0., b=1.)
        # Initialize weights for `pre_proj`.
        for module_i in self.pre_proj.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)
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
    update funcs
    """
    # def update_params func
    def update_params(self, iteration):
        """
        Update parameters according to the specified iteration.

        Args:
            iteration: int - The index of current iteration (i.e., the number of updates).

        Returns:
            None
        """
        # Update the temperature parameter.
        self.tau_i = max(self.tau_max * (self.tau_decay ** iteration), self.tau_min)

    """
    network funcs
    """
    # def forward func
    def forward(self, Z):
        """
        Forward layers in `GumbelVectorQuantizer` to get vector-quantized embeddings.

        Args:
            Z: (batch_size, *, d_model) - The input embeddings.

        Returns:
            Z_q: (batch_size, *, d_model) - The vector-quantized embeddings.
            loss: torch.float32 - The vector-quantizer loss.
            codex_probs: (batch_size, *, n_groups, codex_size) - The one-hot probabilities of input embeddings.
        """
        # Forward the pre-projection layer to get the codex logits.
        # codex_logits - (batch_size, *, n_groups, codex_size)
        codex_logits = torch.reshape(self.pre_proj(Z), shape=(*Z.shape[:-1], self.n_groups, self.codex_size))
        # Forward Gumbel Softmax to get the codex probabilities.
        # codex_probs - (batch_size, *, n_groups, codex_size)
        codex_probs = self._gumbel_softmax(codex_logits, tau=self.tau_i, use_hard=self.use_hard, dim=-1)
        # Get the codex of vector-quantizer.
        # codex - (n_groups, codex_size, d_codex // n_groups)
        codex = torch.tile(self.codex, dims=[self.n_groups, 1, 1]) if self.share_group else self.codex
        # Get the vector-quantized embeddings.
        # Z_q - (batch_size, *, n_groups, codex_size, d_codex // n_groups)
        Z_q = torch.unsqueeze(codex_probs, dim=-1) * codex
        # Z_q - (batch_size, *, d_codex)
        Z_q = torch.reshape(torch.sum(Z_q, dim=-2), shape=(*Z.shape[:-1], self.d_codex))
        # Calculate the vector-quantizer loss.
        # loss - torch.float32
        loss = torch.tensor(0., requires_grad=True)
        # Calculate perplexity to measure the codex usage.
        # codex_usage - (n_groups, codex_size) or (1, codex_size)
        codex_usage = torch.mean(torch.reshape(codex_probs, shape=(-1, *codex_probs.shape[-2:])), dim=0)
        codex_usage = torch.mean(codex_usage, dim=0, keepdim=True) if self.share_group else codex_usage
        # perplexity - torch.float32
        perplexity = torch.mean(torch.exp(-torch.sum(codex_usage * torch.log(codex_usage + 1e-12), dim=-1)))
        # Normalize the vector-quantized embeddings to align the scale.
        Z_q = self.layernorm(Z_q)
        # Forward the post-projection layer to get the embeddings in embedding space.
        # Z_q - (batch_size, *, d_model)
        Z_q = self.post_proj(Z_q)
        # Return the final `Z_q` & `loss` & `codex_probs`.
        return Z_q, loss, codex_probs

    """
    tool funcs
    """
    # def _gumbel_softmax func
    def _gumbel_softmax(self, logits, tau=1., use_hard=False, dim=-1):
        """
        Samples from Gumbel-Softmax distribution and optionally discretizes.
        The main trick for `soft` (i.e., `use_hard=False`) is reparameterization.
        The main trick for `hard` (i.e., `use_hard=True`) is straight through
        (i.e., `probs_hard - probs_soft.detach() + probs_soft`). It achieves two things:
         - makes the output value exactly one-hot (since we add then subtract `probs_soft` value),
         - makes the gradient equal to `probs_soft` gradient (since we strip all other gradients).

        [1] Maddison C J, Mnih A, Teh Y W. The concrete distribution: A continuous relaxation of
            discrete random variables[J]. arXiv preprint arXiv:1611.00712, 2016.
        [2] Jang E, Gu S, Poole B. Categorical reparameterization with gumbel-softmax[J].
            arXiv preprint arXiv:1611.01144, 2016.

        Args:
            logits: (*, n_features) - The un-normalized log probabilities.
            tau: float - The non-negative scalar temperature.
            use_hard: bool - The flag that indicates whether return discretized one-hot vectors.
            dim: int - The dimension along which softmax will be computed.

        Returns:
            probs: (*, n_features) - The Gumbel-Softmax probabilities.
        """
        # Initialize gumbels from Gumbel distribution.
        # gumbels - (*, n_features)
        gumbels = -torch.log(-torch.log(torch.rand(logits.shape, dtype=logits.dtype).to(device=logits.device)))
        # Add `logits` & `gumbels` to get the corresponding `probs`, then softmax it.
        # probs* - (*, n_features)
        probs_soft = utils.model.torch.softmax((logits + gumbels) / tau, dim=dim)
        # Re-parameterize `probs` according to `use_hard`.
        if use_hard:
            # Use straight-through trick (proposed by Jang et al. 2016) to reparameterize `probs`.
            probs_hard = torch.zeros_like(probs_soft, dtype=probs_soft.dtype).scatter_(dim=dim,
                index=torch.argmax(probs_soft, dim=dim, keepdim=True), value=1.)
            probs = probs_hard - probs_soft.detach() + probs_soft
        else:
            # Use reparameterization trick (proposed by Maddison et al. 2016) to reparameterize `probs`.
            probs = probs_soft
        # Return the final `probs`.
        return probs

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128
    codex_size = 8192; d_codex = 256; n_groups = 4; share_group = False
    use_hard = True; d_hidden = []; activation = nn.GELU(approximate="none"); tau_factors = (1., 1., 1.)

    ## Forward GumbelVectorQuantizer.
    # Initialize input `Z`.
    # Z - (batch_size, emb_len, d_model)
    Z = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate GumbelVectorQuantizer.
    gvq_inst = GumbelVectorQuantizer(d_model=d_model, codex_size=codex_size,
        d_codex=d_codex, n_groups=n_groups, share_group=share_group, use_hard=use_hard,
        d_hidden=d_hidden, activation=activation, tau_factors=tau_factors
    )
    # Forward layers in `gvq_inst`.
    # Z_q - (batch_size, emb_len, d_model); loss - torch.float32; codex_probs - (batch_size, emb_len, n_groups, codex_size)
    Z_q, loss, codex_probs = gvq_inst(Z)

