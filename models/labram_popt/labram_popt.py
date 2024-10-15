#!/usr/bin/env python3
"""
Created on 21:51, Jan. 19th, 2024

@author: Norbert Zheng
"""
import re, torch
import copy as cp
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from layers import *
else:
    from .layers import *
from utils import DotDict

__all__ = [
    "labram_popt_vqvae",
    "labram_popt_mae",
    "labram_popt_cls",
]

# def labram_popt_vqvae class
class labram_popt_vqvae(nn.Module):
    """
    LaBraM model for neural signal prediction introduced by Jiang et al. 2024.

    [1] Jiang W B, Zhao L M, Lu B L. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI[C].
        International Conference on Learning Representations (ICLR), 2024.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `labram_popt_vqvae` object.

        Args:
            params: DotDict - Model parameters initialized by labram_popt_vqvae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(labram_popt_vqvae, self).__init__(**kwargs)

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
        # Initialize tokenizer block.
        # tokenizer - (batch_size, n_channels, n_segs, seg_len) -> (batch_size, n_channels, n_segs, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize encoder block.
        # encoder - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_model)
        self.encoder = nn.Sequential(OrderedDict([
            ("emb_chan", nn.Sequential(
                LambdaLayer(func=(lambda x: [torch.permute(x[0], dims=[0,2,1,3]), x[1]])),
                ChannelEmbedding(d_model=self.params.encoder.d_model, n_axes=self.params.encoder.n_axes),
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1,3]))),
            )),
            ("emb_time", TimeEmbedding(d_model=self.params.encoder.d_model, max_segs=self.params.encoder.max_segs)),
            ("xfmr_stack", nn.Sequential(
                LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], -1, x.shape[-1])))),
                TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
                LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], self.params.encoder.n_channels, -1, x.shape[-1])))),
            )),
        ]))
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_codex)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize decoder block.
        # decoder - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_model)
        self.decoder = nn.Sequential(OrderedDict([
            ("emb_chan", nn.Sequential(
                LambdaLayer(func=(lambda x: [torch.permute(x[0], dims=[0,2,1,3]), x[1]])),
                ChannelEmbedding(d_model=self.params.encoder.d_model, n_axes=self.params.encoder.n_axes),
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1,3]))),
            )),
            ("emb_time", TimeEmbedding(d_model=self.params.encoder.d_model, max_segs=self.params.encoder.max_segs)),
            ("xfmr_stack", nn.Sequential(
                LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], -1, x.shape[-1])))),
                TransformerStack(self.params.decoder), LambdaLayer(func=(lambda x: x[0])),
                LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], self.params.encoder.n_channels, -1, x.shape[-1])))),
            )),
        ]))
        # Initialize regression blocks.
        # rgs_blocks - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, seg_len)
        self.rgs_blocks = nn.ModuleList(modules=[SpecRGSHead(params=self.params.rgs) for _ in range(2)])

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
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
            "([^.]*\.)*vq_block": "vq_block",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) in models.labram.labram_popt_vqvae."
        ).format(modules))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `labram_popt_vqvae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,Fs,channel_pos].

        Returns:
            Fs_reconstr: (2[list], batch_size, n_channels, n_segs, seg_len) - The reconstructed fourier spectrum.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, n_segs, seg_len); Fs - (2[list], batch_size, n_channels, n_segs, seg_len)
        # channel_pos - (batch_size, n_channels, n_axes)
        X = inputs[0]; Fs = inputs[1]; channel_pos = inputs[2]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, n_channels), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = self.tokenizer(X)
        # Forward encoder block to get the transformed embedding sequence.
        # E - (batch_size, n_channels, n_segs, d_model)
        E = self.encoder((T, channel_pos))
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, n_channels, n_segs, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Forward decoder block to get the corresponding token reconstructon.
        # T_reconstr - (batch_size, n_channels, n_segs, d_model)
        T_reconstr = self.decoder((E_vq, channel_pos))
        # Forward time regression block to  get the reconstruction fourier spectrum.
        # Fs_reconstr - (2[list], batch_size, n_channels, n_segs, seg_len)
        Fs_reconstr = [rgs_block_i(T_reconstr) for rgs_block_i in self.rgs_blocks]
        # Calculate the regression loss.
        # loss_rgs* - torch.float32
        loss_rgs = [self._loss_rgs(F_reconstr_i, F_i) for (F_i, F_reconstr_i) in zip(Fs, Fs_reconstr)]
        loss_rgs_amp = loss_rgs[0]; loss_rgs_ang = loss_rgs[1]; loss_rgs = (loss_rgs_amp + loss_rgs_ang) / 2.
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.rgs_loss_scale * loss_rgs +\
            self.params.vq_loss_scale * loss_vq
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "vq": loss_vq,
            "rgs_amp": loss_rgs_amp,
            "rgs_ang": loss_rgs_ang,
        })
        # Return the final `Fs_reconstr` & `loss`.
        return Fs_reconstr, loss

    # def quantize func
    def quantize(self, inputs):
        """
        Forward `labram_popt_vqvae` to get the quantized embeddings.

        Args:
            inputs: tuple - The input data, including [X,channel_pos].

        Returns:
            E_vq: (batch_size, n_channels, n_segs, d_model) - The quantized embeddings.
            loss_vq: torch.float32 - The vector-quantizer loss.
            codex_probs: (batch_size, n_channels, n_segs, codex_size) - The one-hot probabilities of the embeddings.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, n_segs, seg_len); channel_pos - (batch_size, n_channels, n_axes)
        X = inputs[0]; channel_pos = inputs[1]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, n_channels), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = self.tokenizer(X)
        # Forward encoder block to get the transformed embedding sequence.
        # E - (batch_size, n_channels, n_segs, d_model)
        E = self.encoder((T, channel_pos))
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, n_channels, n_segs, d_model); loss_vq - torch.float32
        # codex_probs - (batch_size, n_channels, n_segs, codex_size)
        E_vq, loss_vq, codex_probs = self.vq_block(E)
        # Return the final `E_vq` & `loss_vq` & `codex_probs`.
        return E_vq, loss_vq, codex_probs

    """
    loss funcs
    """
    # def _loss_rgs func
    def _loss_rgs(self, value, target):
        """
        Calculate regresion error between (list of) tensors value and target. Include a factor
        0.5 to squared error by convention. Set `keepdims` to false, then get sum over last dimension to keep
        losses of different batches separate.

        Args:
            value: (batch_size, *, d_model) - Value of the object.
            target: (batch_size, *, d_model) - Traget of the object.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the mean-squared-error loss.
        # loss - torch.float32
        loss = F.mse_loss(
            # Modified `mse_loss` function arguments.
            input=value, target=target,
            # Default `mse_loss` function arguments.
            size_average=None, reduce=None, reduction="mean"
        )
        # Return the final `loss`.
        return loss

# def labram_popt_mae class
class labram_popt_mae(nn.Module):
    """
    LaBraM model for neural token prediction introduced by Jiang et al. 2024.

    [1] Jiang W B, Zhao L M, Lu B L. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI[C].
        International Conference on Learning Representations (ICLR), 2024.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `labram_popt_mae` object.

        Args:
            params: DotDict - Model parameters initialized by labram_popt_mae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(labram_popt_mae, self).__init__(**kwargs)

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
        # Initialize mask embedding.
        # mask_emb - (d_model,)
        mask_emb = torch.ones((self.params.encoder.d_model,), dtype=torch.float32)
        self.mask_emb = nn.Parameter(mask_emb, requires_grad=True)
        # Initialize tokenizer block.
        # tokenizer - (batch_size, n_channels, n_segs, seg_len) -> (batch_size, n_channels, n_segs, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize encoder block.
        # encoder - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_model)
        self.encoder = nn.Sequential(OrderedDict([
            ("emb_chan", nn.Sequential(
                LambdaLayer(func=(lambda x: [torch.permute(x[0], dims=[0,2,1,3]), x[1]])),
                ChannelEmbedding(d_model=self.params.encoder.d_model, n_axes=self.params.encoder.n_axes),
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1,3]))),
            )),
            ("emb_time", TimeEmbedding(d_model=self.params.encoder.d_model, max_segs=self.params.encoder.max_segs)),
            ("xfmr_stack", nn.Sequential(
                LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], -1, x.shape[-1])))),
                TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
                LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], self.params.encoder.n_channels, -1, x.shape[-1])))),
            )),
        ]))
        # Initialize classification block.
        # cls_block - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, n_tokens)
        self.cls_block = TokenCLSHead(params=self.params.cls)

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for `mask_emb`.
        nn.init.trunc_normal_(self.mask_emb, mean=0., std=0.02)

    """
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) in models.labram.labram_popt_mae."
        ).format(modules))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `labram_popt_mae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,c_true,channel_pos].

        Returns:
            c_pred: (batch_size, n_channels, n_segs, codex_size) - The predicted codex.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, n_segs, seg_len); c_true - (batch_size, n_channels, n_segs, codex_size)
        # channel_pos - (batch_size, n_channels, n_axes)
        X = inputs[0]; c_true = inputs[1]; channel_pos = inputs[2]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, n_channels), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = self.tokenizer(X)
        # Generate mask according to the init embedding `E`.
        # mask - (batch_size, n_channels, n_segs)
        mask = self.gen_mask(T, mask_ratio=self.params.mask_ratio)
        # Get the masked embedding `T_masked` according to `mask`.
        # mask_emb - (batch_size, n_channels, n_segs, d_model)
        mask_emb = self.mask_emb[None,None,None,...].expand(*mask.shape, -1)
        # T_masked - (2[list], batch_size, n_channels, n_segs, d_model)
        T_masked = [
            (T * (1. - mask[...,None].to(dtype=T.dtype)) + mask_emb * mask[...,None].to(dtype=T.dtype)),
            (T * mask[...,None].to(dtype=T.dtype) + mask_emb * (1. - mask[...,None].to(dtype=T.dtype))),
        ]
        # Forward encoder block to get the transformed embedding sequence.
        # E - (2[list], batch_size, n_channels, n_segs, d_model)
        E = [self.encoder((T_i, channel_pos)) for T_i in T_masked]
        # Forward classification block to get the corresponding prediction.
        # c_pred - (batch_size, n_channels, n_segs, codex_size)
        c_pred = [self.cls_block(E_i) for E_i in E]
        c_pred = (
            (c_pred[0] * mask[...,None].to(dtype=c_pred[0].dtype)) +\
            (c_pred[1] * (1. - mask[...,None].to(dtype=c_pred[1].dtype)))
        )
        # Calculate the binary cross entropy loss.
        # loss_cls - torch.float32
        loss_cls = self._loss_cls(
            torch.reshape(c_pred, shape=(-1, c_pred.shape[-1])),
            torch.reshape(c_true, shape=(-1, c_true.shape[-1])),
        )
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
        })
        # Return the final `c_pred` & `loss`.
        return c_pred, loss

    # def gen_mask func
    def gen_mask(self, E, mask_ratio=0.5):
        """
        Generate mask for embedding sequence.

        Args:
            E: (batch_size, n_channels, n_segs, d_model) - The embedding sequence.
            mask_ratio: float - The mask ratio of each embedding item.

        Returns:
            mask: (batch_size, n_channels, n_segs) - The generated mask.
        """
        # Initialize `batch_size` & `n_channels` & `n_segs` & `d_model` from `E`.
        batch_size, n_channels, n_segs, d_model = E.shape; emb_len = n_channels * n_segs
        # Initialize the length of keep embedding items.
        keep_len = int(emb_len * (1. - mask_ratio))
        # Initialize the noise for further argsort.
        # noise - (batch_size, emb_len)
        noise = torch.rand((batch_size, emb_len), dtype=E.dtype).to(device=E.device)
        # Get the corresponding `shuffle_idxs` & `restore_idxs`.
        # Note: `torch.argsort` is reversible, we have `shuffle_idxs = torch.argsort(restore_idxs)`.
        shuffle_idxs = torch.argsort(noise, dim=-1); restore_idxs = torch.argsort(shuffle_idxs, dim=-1)
        # Generate the bool mask: `False` is keep, `True` is remove.
        # mask - (batch_size, emb_len)
        mask = torch.ones((batch_size, emb_len), dtype=torch.bool).to(device=E.device); mask[:,:keep_len] = False
        # Un-shuffle to get the bool mask.
        mask = torch.gather(mask, dim=-1, index=restore_idxs)
        # Re-shape back the bool mask.
        # mask - (batch_size, n_channels, n_segs)
        mask = torch.reshape(mask, shape=(batch_size, n_channels, n_segs))
        # Return the final `mask`.
        return mask

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        Args:
            value: (batch_size, n_labels) - Value of the object.
            target: (batch_size, n_labels) - Target of the object.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the cross-entropy loss.
        # loss - torch.float32
        loss = F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=value, target=target,
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        # Return the final `loss`.
        return loss

# def labram_popt_cls class
class labram_popt_cls(nn.Module):
    """
    LaBraM model for classification task introduced by Jiang et al. 2024.

    [1] Jiang W B, Zhao L M, Lu B L. Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI[C].
        International Conference on Learning Representations (ICLR), 2024.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `labram_popt_cls` object.

        Args:
            params: DotDict - Model parameters initialized by labram_popt_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(labram_popt_cls, self).__init__(**kwargs)

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
        # Initialize tokenizer block.
        # tokenizer - (batch_size, n_channels, n_segs, seg_len) -> (batch_size, n_channels, n_segs, d_model)
        self.tokenizer = PatchTokenizer(params=self.params.tokenizer)
        # Initialize encoder block.
        # encoder - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_model)
        self.encoder = nn.Sequential(OrderedDict([
            ("emb_chan", nn.Sequential(
                LambdaLayer(func=(lambda x: [torch.permute(x[0], dims=[0,2,1,3]), x[1]])),
                ChannelEmbedding(d_model=self.params.encoder.d_model, n_axes=self.params.encoder.n_axes),
                LambdaLayer(func=(lambda x: torch.permute(x, dims=[0,2,1,3]))),
            )),
            ("emb_time", TimeEmbedding(d_model=self.params.encoder.d_model, max_segs=self.params.encoder.max_segs)),
            ("xfmr_stack", nn.Sequential(
                LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], -1, x.shape[-1])))),
                TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
                LambdaLayer(func=(lambda x: torch.reshape(x, shape=(x.shape[0], self.params.encoder.n_channels, -1, x.shape[-1])))),
            )),
        ]))
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_codex)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize classification block.
        # cls_block - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_labels)
        self.cls_block = LabelCLSHead(params=self.params.cls)

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
    load funcs
    """
    # def load_weight func
    def load_weight(self, path_ckpt):
        """
        Load model weights from the specified checkpoint path.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*tokenizers": "tokenizers",
            "([^.]*\.)*encoder": "encoder",
        }
        for parameter_name_i in ckpt_dict.keys():
            for module_src_i, module_trg_i in module_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) in models.labram.labram_popt_cls."
        ).format(modules))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `labram_popt_cls` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,y_true,channel_pos].

        Returns:
            y_pred: (batch_size, n_labels) - The output labels.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, n_segs, seg_len); y_true - (batch_size, n_labels)
        # channel_pos - (batch_size, n_channels, n_axes)
        X = inputs[0]; y_true = inputs[1]; channel_pos = inputs[2]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, n_channels), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = self.tokenizer(X)
        # Forward encoder block to get the transformed embedding sequence.
        # E - (batch_size, n_channels, n_segs, d_model)
        E = self.encoder((T, channel_pos))
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, n_channels, n_segs, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Forward classification block to get the corresponding prediction.
        # y_pred - (batch_size, n_labels)
        y_pred = self.cls_block(E)
        # Calculate the binary cross entropy loss.
        # loss_cls - torch.float32
        loss_cls = self._loss_cls(y_pred, y_true)
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
        })
        # Return the final `y_pred` & `loss`.
        return y_pred, loss

    """
    loss funcs
    """
    # def _loss_cls func
    def _loss_cls(self, value, target):
        """
        Calculates classification loss between tensors value and target.
        Get mean over last dimension to keep losses of different batches separate.

        Args:
            value: (batch_size, n_labels) - Value of the object.
            target: (batch_size, n_labels) - Target of the object.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the cross-entropy loss.
        # loss - torch.float32
        loss = F.cross_entropy(
            # Modified `cross_entropy` function arguments.
            input=value, target=target,
            # Default `cross_entropy` function arguments.
            weight=None, size_average=None, ignore_index=-100,
            reduce=None, reduction="mean", label_smoothing=0.
        )
        # Return the final `loss`.
        return loss

if __name__ == "__main__":
    import numpy as np
    # local dep
    import utils.model.torch
    from params.labram_popt_params import labram_popt_vqvae_params, labram_popt_mae_params, labram_popt_cls_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32
    seq_len = 800; seg_len = 200; n_segs = (seq_len // seg_len)
    n_channels = 128; n_labels = 61; n_subjects = 10; n_axes = 3

    # Initialize training process.
    utils.model.torch.set_seeds(42)

    ## Forward labram_popt_vqvae.
    # Instantiate params.
    labram_popt_vqvae_params_inst = labram_popt_vqvae_params(dataset=dataset)
    labram_popt_vqvae_params_inst.model.n_subjects = n_subjects
    labram_popt_vqvae_params_inst.model.encoder.n_channels = labram_popt_vqvae_params_inst.model.n_channels = n_channels
    labram_popt_vqvae_params_inst.model.encoder.n_axes = labram_popt_vqvae_params_inst.model.n_axes = n_axes
    labram_popt_vqvae_params_inst.model.seq_len = seq_len
    labram_popt_vqvae_params_inst.model.encoder.max_segs = (
        labram_popt_vqvae_params_inst.model.seq_len // labram_popt_vqvae_params_inst.model.seg_len
    )
    emb_len = labram_popt_vqvae_params_inst.model.encoder.n_channels * labram_popt_vqvae_params_inst.model.encoder.max_segs
    labram_popt_vqvae_params_inst.model.decoder.emb_len = labram_popt_vqvae_params_inst.model.encoder.emb_len = emb_len
    labram_popt_vqvae_params_inst.model.rgs.n_segs = (
        labram_popt_vqvae_params_inst.model.seq_len // labram_popt_vqvae_params_inst.model.seg_len
    )
    # Initialize input `X` & `Fs`.
    # X - (batch_size, n_channels, n_segs, seg_len); F - (n_channels, n_segs, n_fft), where `n_fft = seg_len`.
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    F_ = torch.fft.fft(X, dim=-1); Fs = [torch.abs(F_), torch.angle(F_)]
    Fs = [((F_i - torch.mean(F_i, dim=-1, keepdims=True)) / torch.std(F_i, dim=-1, keepdims=True)) for F_i in Fs]
    channel_pos = torch.rand((batch_size, n_channels, n_axes), dtype=torch.float32) * 100.
    # Instantiate labram_popt_vqvae.
    labram_popt_vqvae_inst = labram_popt_vqvae(labram_popt_vqvae_params_inst.model); print(labram_popt_vqvae_inst)
    # Forward layers in `labram_popt_vqvae_inst`.
    # Fs_reconstr - (2[list], batch_size, n_channels, n_segs, seg_len); loss - torch.float32
    Fs_reconstr, loss = labram_popt_vqvae_inst((X, Fs, channel_pos))
    # Forward layers before vector-quantizer in `labram_popt_vqvae_inst`.
    # E_vq - (batch_size, n_channels, n_segs, d_model); loss_vq - torch.float32
    # codex_probs - (batch_size, n_channels, n_segs, codex_size)
    E_vq, loss_vq, codex_probs = labram_popt_vqvae_inst.quantize((X, channel_pos))
    ## Forward labram_popt_mae.
    # Instantiate params.
    labram_popt_mae_params_inst = labram_popt_mae_params(dataset=dataset)
    labram_popt_mae_params_inst.model.n_subjects = n_subjects
    labram_popt_mae_params_inst.model.encoder.n_channels = labram_popt_mae_params_inst.model.n_channels = n_channels
    labram_popt_mae_params_inst.model.encoder.n_axes = labram_popt_mae_params_inst.model.n_axes = n_axes
    labram_popt_mae_params_inst.model.seq_len = seq_len
    labram_popt_mae_params_inst.model.encoder.max_segs = (
        labram_popt_mae_params_inst.model.seq_len // labram_popt_mae_params_inst.model.seg_len
    )
    labram_popt_mae_params_inst.model.encoder.emb_len = (
        labram_popt_mae_params_inst.model.encoder.n_channels * labram_popt_mae_params_inst.model.encoder.max_segs
    )
    # Initialize input `X` & `c_true`.
    # X - (batch_size, n_channels, n_segs, seg_len); c_true - (batch_size, n_channels, n_segs, d_model)
    codex_size = labram_popt_mae_params_inst.model.vq.codex_size
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    c_true = torch.tensor(np.eye(codex_size)[np.random.randint(0, codex_size,
        size=(batch_size, n_channels, n_segs))], dtype=torch.float32)
    channel_pos = torch.rand((batch_size, n_channels, n_axes), dtype=torch.float32) * 100.
    # Instantiate labram_popt_mae.
    labram_popt_mae_inst = labram_popt_mae(labram_popt_mae_params_inst.model); print(labram_popt_mae_inst)
    # Forward layers in `labram_popt_mae_inst`.
    # c_pred - (batch_size, n_channels, n_segs, d_model); loss - torch.float32
    c_pred, loss = labram_popt_mae_inst((X, c_true, channel_pos))
    ## Forward labram_popt_cls.
    # Instantiate params.
    labram_popt_cls_params_inst = labram_popt_cls_params(dataset=dataset)
    labram_popt_cls_params_inst.model.n_subjects = n_subjects
    labram_popt_cls_params_inst.model.encoder.n_channels = labram_popt_cls_params_inst.model.n_channels = n_channels
    labram_popt_cls_params_inst.model.encoder.n_axes = labram_popt_cls_params_inst.model.n_axes = n_axes
    labram_popt_cls_params_inst.model.seq_len = seq_len
    labram_popt_cls_params_inst.model.encoder.max_segs = (
        labram_popt_cls_params_inst.model.seq_len // labram_popt_cls_params_inst.model.seg_len
    )
    labram_popt_cls_params_inst.model.encoder.emb_len = (
        labram_popt_cls_params_inst.model.encoder.n_channels * labram_popt_cls_params_inst.model.encoder.max_segs
    )
    labram_popt_cls_params_inst.model.cls.d_feature = (
        labram_popt_cls_params_inst.model.encoder.emb_len * labram_popt_cls_params_inst.model.encoder.d_model
    )
    labram_popt_cls_params_inst.model.cls.n_labels = labram_popt_cls_params_inst.model.n_labels = n_labels
    # Initialize input `X_s` & `y_true` & `X_n`.
    # X - (batch_size, n_channels, n_segs, seg_len); y_true - (batch_size, n_labels)
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    channel_pos = torch.rand((batch_size, n_channels, n_axes), dtype=torch.float32) * 100.
    # Instantiate labram_popt_cls.
    labram_popt_cls_inst = labram_popt_cls(labram_popt_cls_params_inst.model); print(labram_popt_cls_inst)
    # Forward layers in `labram_popt_cls_inst`.
    # y_pred - (batch_size, n_labels); loss - torch.float32
    y_pred, loss = labram_popt_cls_inst((X, y_true, channel_pos))

