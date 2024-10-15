#!/usr/bin/env python3
"""
Created on 14:46, Feb. 15th, 2024

@author: Norbert Zheng
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as distributed
from einops import rearrange, repeat
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
import utils.model.torch

__all__ = [
    "LaBraMVectorQuantizer",
]

"""
core classes
"""
# def LaBraMVectorQuantizer class
class LaBraMVectorQuantizer(nn.Module):
    """
    Normalized EMA Vector Quantizer.
    """

    def __init__(self, d_model, codex_size, d_codex, beta=1., decay=0.99, init_kmeans=True, **kwargs):
        """
        Initialize `LaBraMVectorQuantizer` object.

        Args:
            d_model: int - The dimensions of model embedding.
            codex_size: int - The number of discrete embeddings.
            d_codex: int - The dimensions of codex embedding.
            beta: float - The scale factor of commitment loss (which is a part of vq loss).
            decay: float - The decay factor used to update embeddings according to `decay * weight_old + (1 - decay) * weight_new`.
            init_kmeans: bool - The flag that indicates whether use kmeans to initialize weight.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(LaBraMVectorQuantizer, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_model = d_model; self.codex_size = codex_size; self.d_codex = d_codex
        self.beta = beta; self.decay = decay; self.init_kmeans = init_kmeans

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
        # Initialize the discrete codex.
        # codex - (codex_size, d_codex)
        self.codex = EMAEmbedding(n_embs=self.codex_size, d_emb=self.d_codex, decay=self.decay, init_kmeans=self.init_kmeans)
        # Initialize counts related to each codex.
        # counts - (codex_size,)
        counts = torch.zeros((self.codex_size,), dtype=torch.float32)
        self.counts = nn.Parameter(counts, requires_grad=False)
        # Initialize distributed functions.
        if distributed.is_available() and distributed.is_initialized():
            print("INFO: DDP is enabled, use ddp_reduce to sync across multi-GPUs!")
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
    state funcs
    """
    # def init_counts func
    def init_counts(self):
        """
        Initialize embedding counts.

        Args:
            None

        Returns:
            None
        """
        self.counts.data.copy_(torch.zeros(self.counts.shape))

    # def get_counts func
    def get_counts(self):
        """
        Get codex counts.

        Args:
            None

        Returns:
            counts: (codex_size,) - The pseudo-counts of each codex.
        """
        return self.counts.detach().cpu().numpy()

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
        Z_e = l2norm(Z_e); self.codex.init_emb(torch.reshape(Z_e, shape=(-1, Z_e.shape[-1]))); codex = self.codex.embeddings
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
        Z_q = torch.reshape(self.codex(codex_idxs), shape=Z_e.shape)
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
            # Get counts of current call.
            # counts - (codex_size,)
            counts = torch.sum(torch.reshape(codex_probs, shape=(-1, codex_probs.shape[-1])), dim=0)
            utils.model.torch.all_reduce(counts)
            # Update the whole counts with ema_inplace, this is not the true counts!
            ema_inplace(self.counts, counts, decay=self.decay, use_norm=False)
            # If there is any 0-count items, keep them unchanged!
            # zero_mask - (codex_size,)
            zero_mask = (counts == 0); counts = counts.masked_fill(zero_mask, value=1.)
            # codex_n - (codex_size, d_codex)
            codex_n = torch.matmul(
                torch.permute(torch.reshape(codex_probs, shape=(-1, codex_probs.shape[-1])), dims=[1,0]),
                torch.reshape(Z_e, shape=(-1, Z_e.shape[-1]))
            ); utils.model.torch.all_reduce(codex_n)
            codex_n = l2norm(codex_n / torch.unsqueeze(counts, dim=-1))
            codex_n = torch.where(zero_mask[...,None], codex, codex_n)
            # Update the codex with ema_inplace.
            ema_inplace(self.codex.embeddings, codex_n, decay=self.decay, use_norm=True)
        else:
            with torch.no_grad():
                # Get counts of current call.
                # counts - (codex_size,)
                counts = torch.sum(torch.reshape(codex_probs, shape=(-1, codex_probs.shape[-1])), dim=0)
                utils.model.torch.all_reduce(counts)
                # Update the whole counts with ema_inplace, this is not the true counts!
                ema_inplace(self.counts, counts, decay=self.decay, use_norm=False)
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
module classes
"""
# def EMAEmbedding class
class EMAEmbedding(nn.Module):
    """
    Embedding updated with EMA, instead of gradient backpropagation.
    """

    def __init__(self, n_embs, d_emb, decay=0.99, init_kmeans=True, **kwargs):
        """
        Initialize `EMAEmbedding` object.

        Args:
            n_embs: int - The number of embeddings.
            d_emb: int - The dimensions of embedding.
            decay: float - The decay factor used to update embeddings according to `decay * weight_old + (1 - decay) * weight_new`.
            init_kmeans: bool - The flag that indicates whether use kmeans to initialize weight.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(EMAEmbedding, self).__init__(**kwargs)

        # Initialize parameters.
        self.n_embs = n_embs; self.d_emb = d_emb; self.decay = decay; self.init_kmeans = init_kmeans

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
        # Initialize embeddings.
        # embeddings - (n_embs, d_emb)
        embeddings = l2norm(torch.randn((self.n_embs, self.d_emb), dtype=torch.float32))
        self.embeddings = nn.Parameter(embeddings, requires_grad=False)
        # Initialize initted flag.
        self.initted = nn.Parameter(torch.Tensor([not self.init_kmeans]), requires_grad=False)
        # Initialize distributed functions.
        if distributed.is_available() and distributed.is_initialized():
            print("INFO: DDP is enabled, use ddp_reduce to sync across multi-GPUs!")

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

    # def init_emb func
    @torch.jit.ignore
    def init_emb(self, emb_data):
        """
        Initialize embeddings from input embeddings.

        Args:
            emb_data: (n_samples, d_emb) - The input embeddings.

        Returns:
            None
        """
        # If the embeddings are already initialized, return.
        if self.initted: return
        # Otherwise, initialize `embeddings` according to `emb_data`.
        print("INFO: Perform K-means initialization for embeddings in layers.LaBraMVectorQuantizer.EMAEmbedding.")
        # emb_init - (n_embs, d_emb)
        emb_init, _ = kmeans(
            # Modified `kmeans` function arguments.
            samples=emb_data, n_clusters=self.n_embs,
            # Default `kmeans` function arguments.
            n_iters=10, use_cossim=True
        ); utils.model.torch.broadcast(emb_init, 0); emb_init = l2norm(emb_init)
        self.embeddings.data.copy_(emb_init); self.initted.data.copy_(torch.Tensor([True]))

    """
    network funcs
    """
    # def forward func
    def forward(self, index):
        """
        Forward layers in `EMAEmbedding` to get the indexed embeddings.

        Args:
            index: (batch_size,) - The indices used to index embeddings.

        Returns:
            emb: (batch_size, d_emb) - The indexed embeddings.
        """
        # Get the indexed embeddings.
        # emb - (batch_size, d_emb)
        emb = F.embedding(
            # Modified `embedding` function arguments.
            input=index, weight=self.embeddings,
            # Default `embedding` function arguments.
            padding_idx=None, max_norm=None, norm_type=2.,
            scale_grad_by_freq=False, sparse=False
        )
        # Return the final `emb`.
        return emb

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

# def kmeans func
def kmeans(samples, n_clusters, n_iters=10, use_cossim=False):
    """
    K-means clustering.

    Args:
        samples: (n_samples, d_emb) - The input samples.
        n_clusters: int - The number of clusters, i.e., K.
        n_iters: int - The number of K-means iterations.
        use_cossim: bool - The flag that indicates whether cosine similarity.

    Returns:
        means: (n_clusters, d_emb) - The K-means cluster means.
        bins: (n_clusters,) - The sample counts of each cluster.
    """
    assert n_iters > 0
    # Sample K means from the input samples.
    # mean_idxs - (n_clusters,)
    mean_idxs = torch.randperm(samples.shape[0], device=samples.device)[:n_clusters]\
        if samples.shape[0] >= n_clusters else torch.randint(0, samples.shape[0], size=(n_clusters,), device=samples.device)
    # means - (n_clusters, d_emb)
    means = samples[mean_idxs,...]; bins = None
    # Loop over K-means iterations to update `means` & `bins`.
    for iter_idx in range(n_iters):
        # Calculate `similarity` according to `use_cossim`.
        # similarity - (n_samples, n_clusters)
        similarity = (samples @ torch.permute(means, dims=[1,0])) if use_cossim else\
            -torch.sum((torch.unsqueeze(samples, dim=-2) - torch.unsqueeze(means, dim=0)) ** 2, dim=-1)
        # cluster_idxs - (n_samples,)
        cluster_idxs = torch.argmax(similarity, dim=-1)
        # Count the frequency of each value in an array of non-negative ints.
        # bins - (n_samples,)
        bins = torch.bincount(cluster_idxs, minlength=n_clusters)
        # Cannot have 0s, clamp the minimum counts with 1s.
        # zero_mask - (n_samples,)
        zero_mask = (bins == 0)
        # bins_clamped - (n_samples,)
        bins_clamped = bins.masked_fill(zero_mask, value=1)
        # Calculate the new K means.
        # means_n - (n_clusters,)
        means_n = torch.zeros(means.shape, dtype=means.dtype).to(device=samples.device)
        means_n.scatter_add_(dim=0, index=repeat(cluster_idxs, "n -> n d", d=means.shape[-1]), src=samples)
        means_n = means_n / bins_clamped[...,None]
        # If `use_cossim` is True, l2-normalize `means_n`.
        if use_cossim: means_n = l2norm(means_n)
        # Update `means` according to `zero_mask`, remove frequently used means.
        means = torch.where(zero_mask[...,None], means, means_n)
    # Return the final `means` & `bins`.
    return means, bins

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128
    codex_size = 8192; d_codex = 32; beta = 1.; decay = 0.99; init_kmeans = True

    ## Forward LaBraMVectorQuantizer.
    # Initialize input `Z`.
    # Z - (batch_size, emb_len, d_model)
    Z = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate LaBraMVectorQuantizer.
    lbmvq_inst = LaBraMVectorQuantizer(
        d_model=d_model, codex_size=codex_size, d_codex=d_codex,
        beta=beta, decay=decay, init_kmeans=init_kmeans
    )
    # Forward layers in `lbmvq_inst`.
    # Z_q - (batch_size, emb_len, d_model); loss - torch.float32; codex_probs - (batch_size, emb_len, codex_size)
    Z_q, loss, codex_probs = lbmvq_inst(Z)

