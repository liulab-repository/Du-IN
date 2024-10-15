#!/usr/bin/env python3
"""
Created on 21:59, Feb. 15th, 2024

@author: Norbert Zheng
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
import utils.model.torch

__all__ = [
    "JukeBoxVectorQuantizer",
]

# def JukeBoxVectorQuantizer class
class JukeBoxVectorQuantizer(nn.Module):
    """
    Vector Quantizer used in JukeBox, which is proposed by Dhariwal et al. 2020.

    [1] Dhariwal P, Jun H, Payne C, et al. Jukebox: A generative model for music[J]. arXiv preprint arXiv:2005.00341, 2020.
    """

    def __init__(self, d_model, codex_size, d_codex, beta=1., decay=0.99, thres_usage=1., **kwargs):
        """
        Initialize `JukeBoxVectorQuantizer` object.

        Args:
            d_model: int - The dimensions of model embedding.
            codex_size: int - The number of discrete embeddings.
            d_codex: int - The dimensions of codex embedding.
            beta: float - The scale factor of commitment loss (which is a part of vq loss).
            decay: float - The decay factor used to update embeddings according to `decay * weight_old + (1 - decay) * weight_new`.
            thres_usage: float - The threshold to update codex items with low usage.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(JukeBoxVectorQuantizer, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_model = d_model; self.codex_size = codex_size; self.d_codex = d_codex
        self.beta = beta; self.decay = decay; self.thres_usage = thres_usage

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
        self.pre_proj = nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.d_model, out_features=self.d_model,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
            nn.Tanh(),
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.d_model, out_features=self.d_codex,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
        )
        ## Construct the discrete codex.
        # Initialize distributed functions.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print("INFO: DDP is enabled, use ddp_reduce to sync across multi-GPUs!")
        # Reset the discrete codex.
        self.reset_codex()
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
        # Initialize weights for `proj`.
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
    codex funcs
    """
    # def reset_codex func
    def reset_codex(self):
        """
        Reset the discrete codex.

        Args:
            None

        Returns:
            None
        """
        # The flag that indicates whether `codex` is initialized.
        self.codex_initted = False
        # The pseudo-usage count of each codex.
        # codex_counts - (codex_size,)
        self.codex_counts = None
        # The pseudo-sum of embeddings assigned to each codex.
        # codex_sum - (codex_size, d_codex)
        self.codex_sum = None
        # The discrete codex.
        # codex - (codex_size, d_codex)
        codex = torch.zeros((self.codex_size, self.d_codex), dtype=torch.float32)
        self.codex = nn.Parameter(codex, requires_grad=False)

    # def init_codex func
    def init_codex(self, samples):
        """
        Initialize the discrete codex and start codex statistics.

        Args:
            samples: (n_samples, d_codex) - The input samples, i.e., un-quantized embeddings.

        Returns:
            None
        """
        # Only initialize codex when `codex_initted` is False.
        if not self.codex_initted:
            # Tile samples to make sure `n_samples >= codex_size`.
            n_samples, d_codex = samples.shape; assert d_codex == self.d_codex
            if n_samples < self.codex_size:
                samples = samples.repeat(((self.codex_size + n_samples - 1) // n_samples), 1)
                samples = samples + ((0.01 / np.sqrt(self.d_codex)) * torch.randn(samples.shape, dtype=samples.dtype))
            n_samples = samples.shape[0]; assert n_samples >= self.codex_size
            # Randomly sample K means from samples.
            # means - (codex_size, d_codex)
            means = samples[torch.randperm(n_samples)[:self.codex_size],...]; utils.model.torch.broadcast(means, 0)
            self.codex.data.copy_(l2norm(means))
            # Initialize the codex statistics.
            self.codex_counts = torch.ones((self.codex_size,)).to(device=self.codex.device); self.codex_sum = self.codex.clone()
            # Update the initted flag of codex.
            self.codex_initted = True

    # def restore_codex func
    def restore_codex(self, n_samples=None, thres_usage=1.):
        """
        Restore the discrete codex to re-start codex statistics.

        Args:
            n_samples: int - The number of samples that codex have seen.
            thres_usage: float - The threshold to update codex items with low usage.

        Returns:
            None
        """
        # Update parameters.
        self.thres_usage = thres_usage
        # Initialize the codex statistics.
        self.codex_counts = torch.ones((self.codex_size,)).to(device=self.codex.device); self.codex_sum = self.codex.clone()
        # Update the codex statistics according to `n_samples`.
        if n_samples is not None:
            usage_factor = n_samples / self.codex_size
            self.codex_counts.data.mul_(usage_factor); self.codex_sum.data.mul_(usage_factor)
        # Update the initted flag of codex.
        self.codex_initted = True

    # def update_codex func
    def update_codex(self, samples, codex_probs):
        """
        Update the discrete codex.

        Args:
            samples: (n_samples, d_codex) - The l2-normalized embeddings in codex space.
            codex_probs: (n_samples, codex_size) - The probabilities corresponding to each l2-normalized embeddings.

        Returns:
            None
        """
        with torch.no_grad():
            # Get the sum of embeddings assigned to each codex.
            # codex_sum - (codex_size, d_codex)
            codex_sum = torch.matmul(torch.permute(codex_probs, dims=[1,0]), samples); utils.model.torch.all_reduce(codex_sum)
            # Get the usage count of each codex.
            # codex_counts - (codex_size,)
            codex_counts = torch.sum(codex_probs, dim=0); utils.model.torch.all_reduce(codex_counts)
            # Tile samples to make sure `n_samples >= codex_size`.
            n_samples, d_codex = samples.shape; assert d_codex == self.d_codex
            if n_samples < self.codex_size:
                samples = samples.repeat(((self.codex_size + n_samples - 1) // n_samples), 1)
                samples = samples + ((0.01 / np.sqrt(self.d_codex)) * torch.randn(samples.shape, dtype=samples.dtype))
            n_samples = samples.shape[0]; assert n_samples >= self.codex_size
            # Randomly sample K means from samples.
            # means_r - (codex_size, d_codex)
            means_r = samples[torch.randperm(n_samples)[:self.codex_size],...]; utils.model.torch.broadcast(means_r, 0)
            # Update the codex statistics.
            self.codex_sum = self.decay * self.codex_sum + (1. - self.decay) * codex_sum
            self.codex_counts = self.decay * self.codex_counts + (1. - self.decay) * codex_counts
            # Get the statistical K means from codex_sum.
            # usage_mask - (codex_size,)
            usage_mask = (self.codex_counts >= self.thres_usage)
            # counts_clamped - (codex_size,)
            counts_clamped = self.codex_counts.masked_fill(usage_mask, value=1)
            # means_n - (codex_size, d_codex)
            means_n = self.codex_sum / counts_clamped[...,None]
            # Update codex with `codex_n` & `codex_r`.
            self.codex.data.copy_(l2norm(torch.where(usage_mask[...,None], means_n, means_r)))

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
        if self.codex_counts is not None:
            self.codex_counts.data.copy_(torch.zeros(self.codex_counts.shape))

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
        Forward layers in `LaBraMVectorQuantizer` to get the vector-quantized embeddings.

        Args:
            Z: (batch_size, *, d_model) - The input embeddings.

        Returns:
            Z_q: (batch_size, *, d_model) - The vector-quantized embeddings.
            loss: torch.float32 - The vector-quantizer loss.
            codex_probs: (batch_size, *, codex_size) - The one-hot probabilities of input embeddings.
        """
        # Forward the pre-projection layer to get the embeddings in codex space.
        # Z_e - (batch_size, *, d_codex)
        Z_e = self.pre_proj(Z)
        # Get l2-normalized `Z_e`, initialize `codex` with l2-normalized `Z_e`.
        # TODO: Use `all_reduce_fn` before initialize codex to avoid initialize different codex on different GPUs.
        # Z_e - (batch_size, *, d_codex); codex - (codex_size, d_codex)
        Z_e = l2norm(Z_e); self.init_codex(torch.reshape(Z_e, shape=(-1, Z_e.shape[-1]))); codex = self.codex
        # Calculate the distances between `Z_flattened` and `codex`.
        # I.e., (z - e) ^ 2 = (z ^ 2) + (e ^ 2) - (2 * e * z)
        # codex_dists - (batch_size, *, codex_size)
        codex_dists = (
            torch.sum(torch.reshape(Z_e, shape=(-1, Z_e.shape[-1])).pow(2), dim=-1, keepdim=True)
            + torch.unsqueeze(torch.sum(codex.pow(2), dim=-1), dim=0)
            - 2. * torch.einsum("nd,cd->nc", torch.reshape(Z_e, shape=(-1, Z_e.shape[-1])), codex)
        ); codex_dists = torch.reshape(codex_dists, shape=(*Z_e.shape[:-1], -1))
        # Get the corresponding indices & ont-hot probs of input embeddings.
        # codex_idxs - (batch_size, *)
        codex_idxs = torch.argmin(codex_dists, dim=-1)
        # codex_probs - (batch_size, *, codex_size)
        codex_probs = F.one_hot(codex_idxs, num_classes=self.codex_size).to(dtype=codex_dists.dtype)
        #  Get the vector-quantized embeddings.
        # This operation will make a gradient flow between `Z_q` and `codex`.
        # Z_q - (batch_size, *, d_codex)
        Z_q = torch.reshape(F.embedding(input=codex_idxs, weight=self.codex), shape=Z_e.shape)
        # Calculate the vector-quantizer loss.
        # loss - torch.float32
        loss = self.loss(Z_e, Z_q)
        # Re-parameterize `Z_q` to allow gradients to flow back to other layers.
        Z_q = Z_e + (Z_q - Z_e).detach()
        # Calculate perplexity to measure the codex usage.
        # perplexity - torch.float32
        perplexity = torch.tensor(0., dtype=torch.float32)
        # Update `codex` according to `training`.
        if self.training:
            self.update_codex(
                torch.reshape(Z_e, shape=(-1, Z_e.shape[-1])),
                torch.reshape(codex_probs, shape=(-1, codex_probs.shape[-1]))
            )
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
        Calculate the loss in `LaBraMVectorQuantizer`.

        Args:
            Z_e: (batch_size, *, d_codex) - The input embeddings.
            Z_q: (batch_size, *, d_codex) - The vector-quantized embeddings.

        Returns:
            loss: torch.float - The vector-quantizer loss.
        """
        # Calculate commitment loss and codebook loss.
        # loss_* - torch.float32
        loss_commitment = F.mse_loss(
            # Modified `mse_loss` function arguments.
            input=Z_e, target=Z_q.detach(),
            # Default `mse_loss` function arguments.
            size_average=None, reduce=None, reduction="mean"
        )
        # Calculate teh final vector-quantizer loss.
        # loss - torch.float32
        loss = self.beta * loss_commitment
        # Return the final `loss`.
        return loss

"""
tool funcs
"""
# def l2norm func
def l2norm(emb):
    """
    L2-normalize the embeddings.

    Args:
        emb: (batch_size, *, d_emb) - The input embeddings.

    Returns:
        emb_normed: (batch_size, *, d_emb) - The l2-normalized embeddings.
    """
    # Get the l2-normalized embeddings.
    # emb_normed - (batch_size, *, d_emb)
    emb_normed = F.normalize(
        # Modified `normalize` function arguments.
        input=emb, p=2., dim=-1,
        # Default `normalize` function arguments.
        eps=1e-12, out=None
    )
    # Return the final `emb_normed`.
    return emb_normed

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128
    codex_size = 8192; d_codex = 32; beta = 1.; decay = 0.99; thres_usage = 1.

    ## Forward JukeBoxVectorQuantizer.
    # Initialize input `Z`.
    # Z - (batch_size, emb_len, d_model)
    Z = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate JukeBoxVectorQuantizer.
    jbvq_inst = JukeBoxVectorQuantizer(
        d_model=d_model, codex_size=codex_size, d_codex=d_codex,
        beta=beta, decay=decay, thres_usage=thres_usage
    )
    # Forward layers in `jbvq_inst`.
    # Z_q - (batch_size, emb_len, d_model); loss - torch.float32; codex_probs - (batch_size, emb_len, codex_size)
    Z_q, loss, codex_probs = jbvq_inst(Z)

