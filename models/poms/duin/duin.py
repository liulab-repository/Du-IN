#!/usr/bin/env python3
"""
Created on 21:51, Jan. 19th, 2024

@author: Norbert Zheng
"""
import re, torch
import copy as cp
import torch.nn as nn
import torch.nn.functional as F
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from layers import *
else:
    from .layers import *
from utils import DotDict

__all__ = [
    "duin_vqvae",
    "duin_mae",
    "duin_cls",
]

# def duin_vqvae class
class duin_vqvae(nn.Module):
    """
    Du-IN model for neural signal prediction.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_vqvae` object.

        Args:
            params: DotDict - Model parameters initialized by duin_vqvae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_vqvae, self).__init__(**kwargs)

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
        # tokenizer - (batch_size, seq_len, n_channels) -> (batch_size, token_len, d_model)
        self.tokenizers = nn.ModuleList(modules=[PatchTokenizer(params=self.params.tokenizer) for _ in range(self.params.n_subjects)])
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None) and (self.params.decoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize decoder block.
        # decoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.decoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.decoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize regression block.
        # rgs_block - (batch_size, token_len, d_model) -> (batch_size, seq_len, n_channels)
        self.rgs_blocks = nn.ModuleList(modules=[PatchRGSHead(params=self.params.rgs) for _ in range(self.params.n_subjects)])

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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.poms.duin.duin_vqvae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_vqvae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,subj_id,channel_mask].

        Returns:
            X_reconstr: (batch_size, seq_len, n_channels) - The reconstructed signals.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects); channel_mask - (batch_size, n_channels)
        X = inputs[0]; subj_id = inputs[1]; channel_mask = inputs[2]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, n_channels), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (n_subjects, batch_size, token_len, d_model)
        T = torch.stack([self.tokenizers[subj_idx](X) for subj_idx in range(self.params.n_subjects)], dim=0)
        # T - (batch_size, token_len, d_model)
        T = torch.sum((torch.permute(T, dims=[1,0,2,3]) * subj_id[...,None,None]), dim=1); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Forward decoder & regression block to get the corresponding reconstructon.
        # TODO: Support subject-layer in `rgs_block`, we do not reconstruct the intermediate `X_h_reconstr`.
        # T_reconstr - (batch_size, token_len, d_model)
        T_reconstr = torch.reshape(self.decoder(E_vq), shape=token_shape)
        # X_reconstr - (n_subjects, batch_size, seq_len, n_channels)
        X_reconstr = torch.stack([self.rgs_blocks[subj_idx](T_reconstr) for subj_idx in range(self.params.n_subjects)], dim=0)
        # X_reconstr - (batch_size, seq_len, n_channels)
        X_reconstr = torch.sum((torch.permute(X_reconstr, dims=[1,0,2,3]) * subj_id[...,None,None]), dim=1)
        # Calculate the regression loss.
        # loss_rgs - torch.float32
        loss_rgs = self._loss_rgs(X_reconstr, X, weight=channel_mask.to(dtype=X.dtype))
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
            "rgs": loss_rgs,
        })
        # Return the final `X_reconstr` & `loss`.
        return X_reconstr, loss

    # def quantize func
    def quantize(self, inputs):
        """
        Forward `duin_vqvae` to get the quantized embeddings.

        Args:
            inputs: tuple - The input data, including [X,subj_id].

        Returns:
            E_vq: (batch_size, emb_len, d_model) - The quantized embeddings.
            loss_vq: torch.float32 - The vector-quantizer loss.
            codex_probs: (batch_size, emb_len, codex_size) - The one-hot probabilities of the embeddings.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects)
        X = inputs[0]; subj_id = inputs[1]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, n_channels), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (n_subjects, batch_size, token_len, d_model)
        T = torch.stack([self.tokenizers[subj_idx](X) for subj_idx in range(self.params.n_subjects)], dim=0)
        # T - (batch_size, token_len, d_model)
        T = torch.sum((torch.permute(T, dims=[1,0,2,3]) * subj_id[...,None,None]), dim=1); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32; codex_probs - (batch_size, emb_len, codex_size)
        E_vq, loss_vq, codex_probs = self.vq_block(E)
        # Return the final `E_vq` & `loss_vq` & `codex_probs`.
        return E_vq, loss_vq, codex_probs

    """
    loss funcs
    """
    # def _loss_rgs func
    def _loss_rgs(self, value, target, weight=None):
        """
        Calculate regresion error between (list of) tensors value and target. Include a factor
        0.5 to squared error by convention. Set `keepdims` to false, then get sum over last dimension to keep
        losses of different batches separate.

        Args:
            value: (batch_size, seq_len, n_channels) - Value of the object.
            target: (batch_size, seq_len, n_channels) - Traget of the object.
            weight: (batch_size, n_channels) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the regression loss.
        # loss - (batch_size, seq_len, n_channels)
        loss = torch.square(target - value)
        # Average over all locations.
        # loss - (batch_size, n_channels)
        loss = torch.mean(torch.flatten(torch.permute(loss, dims=[0,-1,*range(1, len(loss.shape)-1)]), start_dim=2, end_dim=-1), dim=-1)
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

# def duin_mae class
class duin_mae(nn.Module):
    """
    Du-IN model for neural token prediction.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_mae` object.

        Args:
            params: DotDict - Model parameters initialized by duin_mae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_mae, self).__init__(**kwargs)

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
        # tokenizer - (batch_size, seq_len, n_channels) -> (batch_size, token_len, d_model)
        self.tokenizers = nn.ModuleList(modules=[PatchTokenizer(params=self.params.tokenizer) for _ in range(self.params.n_subjects)])
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize classification block.
        # cls_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, n_tokens)
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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.poms.duin.duin_mae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_mae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,c_true,subj_id].

        Returns:
            c_pred: (batch_size, emb_len, codex_size) - The predicted codex.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); c_true - (batch_size, emb_len, codex_size); subj_id - (batch_size, n_subjects)
        X = inputs[0]; c_true = inputs[1]; subj_id = inputs[2]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, n_channels), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (n_subjects, batch_size, token_len, d_model)
        T = torch.stack([self.tokenizers[subj_idx](X) for subj_idx in range(self.params.n_subjects)], dim=0)
        # T - (batch_size, token_len, d_model)
        T = torch.sum((torch.permute(T, dims=[1,0,2,3]) * subj_id[...,None,None]), dim=1); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E_init - (batch_size, emb_len, d_model)
        E_init = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Generate mask according to the init embedding `E`.
        # mask - (batch_size, emb_len)
        mask = self.gen_mask(E_init, mask_ratio=self.params.mask_ratio)
        # Get the masked embedding `E_masked` according to `mask`.
        # mask_emb - (batch_size, emb_len, d_model)
        mask_emb = self.mask_emb[None,None,...].expand(*mask.shape, -1)
        # E_masked - (2[list], batch_size, emb_len, d_model)
        E_masked = [
            (E_init * (1. - mask[...,None].to(dtype=E_init.dtype)) + mask_emb * mask[...,None].to(dtype=E_init.dtype)),
            (E_init * mask[...,None].to(dtype=E_init.dtype) + mask_emb * (1. - mask[...,None].to(dtype=E_init.dtype))),
        ]
        # Forward encoder block to get time-aligned token sequence.
        # E - (2[list], batch_size, emb_len, d_model)
        E = [self.encoder(E_i) for E_i in E_masked]
        # Forward classification block to get the corresponding prediction.
        # c_pred - (batch_size, emb_len, codex_size)
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
            E: (batch_size, emb_len, d_model) - The embedding sequence.
            mask_ratio: float - The mask ratio of each embedding item.

        Returns:
            mask: (batch_size, emb_len) - The generated mask.
        """
        # Initialize `batch_size` & `emb_len` & `d_model` from `E`.
        batch_size, emb_len, d_model = E.shape
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

# def duin_cls class
class duin_cls(nn.Module):
    """
    Du-IN model for classification task.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `duin_cls` object.

        Args:
            params: DotDict - Model parameters initialized by duin_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(duin_cls, self).__init__(**kwargs)

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
        # tokenizer - (batch_size, seq_len, n_channels) -> (batch_size, token_len, d_model)
        self.tokenizers = nn.ModuleList(modules=[PatchTokenizer(params=self.params.tokenizer) for _ in range(self.params.n_subjects)])
        # Initialize time embedding block.
        # emb_time - (batch_size, token_len, d_model) -> (batch_size, token_len, d_model)
        assert (self.params.encoder.rot_theta is None)
        self.emb_time = TimeEmbedding(d_model=self.params.encoder.d_model, max_len=self.params.encoder.emb_len, mode="sincos")
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.encoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize vector-quantizer block.
        # vq_block - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.vq_block = LaBraMVectorQuantizer(
            d_model=self.params.vq.d_model, codex_size=self.params.vq.codex_size, d_codex=self.params.vq.d_codex,
            beta=self.params.vq.beta, decay=self.params.vq.decay, init_kmeans=self.params.vq.init_kmeans
        )
        # Initialize contrastive block.
        self.contra_block = ContrastiveBlock(d_model=self.params.contra.d_model,
            d_contra=self.params.contra.d_contra, loss_mode=self.params.contra.loss_mode)
        # Initialize classification block.
        # cls_block - (batch_size, emb_len, d_model) -> (batch_size, n_labels)
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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.poms.duin.duin_cls."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `duin_cls` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,y_true,subj_id].

        Returns:
            y_pred: (batch_size, n_labels) - The output labels.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels); subj_id - (batch_size, n_subjects)
        X = inputs[0]; y_true = inputs[1]; subj_id = inputs[2]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # For example, if `X_h` is 1D-signal of shape (batch_size, data_len, n_channels), the resolution
        # along non-channel axis may be reduced, i.e., `T` is of shape (batch_size, token_len, d_model).
        # Record the shape of tokens before forwarding encoder, so we can reshape after decoder.
        # T - (n_subjects, batch_size, token_len, d_model)
        T = torch.stack([self.tokenizers[subj_idx](X) for subj_idx in range(self.params.n_subjects)], dim=0)
        # T - (batch_size, token_len, d_model)
        T = torch.sum((torch.permute(T, dims=[1,0,2,3]) * subj_id[...,None,None]), dim=1); token_shape = T.shape
        # Reshape tokens to get the init embedding.
        # E - (batch_size, emb_len, d_model)
        E = torch.reshape(T, shape=(token_shape[0], -1, token_shape[-1]))
        # Forward encoder block to get time-aligned token sequence.
        E = self.encoder(E)
        # Forward vector-quantizer block to get vector-quantized token sequence.
        # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32
        E_vq, loss_vq, _ = self.vq_block(E)
        # Calculate the contrastive loss.
        # loss_contra - torch.float32
        loss_contra, _ = self.contra_block(((E, E), (y_true, y_true)))
        # Forward classification block to get the corresponding prediction.
        # y_pred - (batch_size, n_labels)
        y_pred = self.cls_block(E)
        # Calculate the binary cross entropy loss.
        # loss_cls - torch.float32
        loss_cls = self._loss_cls(y_pred, y_true)
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.cls_loss_scale * loss_cls +\
            self.params.contra_loss_scale * loss_contra
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "cls": loss_cls,
            "contra": loss_contra,
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
    from params.poms_params.duin_params import duin_vqvae_params, duin_mae_params, duin_cls_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32; seq_len = 3000; n_channels = 16; n_labels = 61; n_subjects = 10

    # Initialize training process.
    utils.model.torch.set_seeds(42)

    ## Forward duin_vqvae.
    # Instantiate params.
    duin_vqvae_params_inst = duin_vqvae_params(dataset=dataset)
    duin_vqvae_params_inst.model.n_subjects = n_subjects
    duin_vqvae_params_inst.model.n_channels = n_channels
    duin_vqvae_params_inst.model.rgs.n_channels = duin_vqvae_params_inst.model.tokenizer.n_channels = n_channels
    assert seq_len % duin_vqvae_params_inst.model.seg_len == 0; duin_vqvae_params_inst.model.seq_len = seq_len
    token_len = (
        duin_vqvae_params_inst.model.seq_len // duin_vqvae_params_inst.model.tokenizer.seg_len
    ); duin_vqvae_params_inst.model.tokenizer.token_len = token_len
    duin_vqvae_params_inst.model.decoder.emb_len = duin_vqvae_params_inst.model.encoder.emb_len = token_len
    # Initialize input `X` & `subj_id` & `channel_mask`.
    # X - (batch_size, seq_len, n_channels); subj_id - (batch_size, n_subjects); channel_mask - (batch_size, n_channels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    channel_mask = torch.ones((batch_size, n_channels), dtype=torch.bool)
    # Instantiate duin_vqvae.
    duin_vqvae_inst = duin_vqvae(duin_vqvae_params_inst.model); print(duin_vqvae_inst)
    # Forward layers in `duin_vqvae_inst`.
    # X_reconstr - (batch_size, seq_len, n_channels); loss - torch.float32
    X_reconstr, loss = duin_vqvae_inst((X, subj_id, channel_mask))
    # Forward layers before vector-quantizer in `duin_vqvae_inst`.
    # E_vq - (batch_size, emb_len, d_model); loss_vq - torch.float32; codex_probs - (batch_size, emb_len, codex_size)
    E_vq, loss_vq, codex_probs = duin_vqvae_inst.quantize((X, subj_id))
    ## Forward duin_mae.
    # Instantiate params.
    duin_mae_params_inst = duin_mae_params(dataset=dataset)
    duin_mae_params_inst.model.n_subjects = n_subjects
    duin_mae_params_inst.model.tokenizer.n_channels = duin_mae_params_inst.model.n_channels = n_channels
    assert seq_len % duin_mae_params_inst.model.seg_len == 0; duin_mae_params_inst.model.seq_len = seq_len
    token_len = (
        duin_mae_params_inst.model.seq_len // duin_mae_params_inst.model.tokenizer.seg_len
    ); duin_mae_params_inst.model.encoder.emb_len = duin_mae_params_inst.model.tokenizer.token_len = token_len
    # Initialize input `X` & `c_true` & `subj_id`.
    # X - (batch_size, seq_len, n_channels); c_true - (batch_size, emb_len, codex_size); subj_id - (batch_size, n_subjects)
    emb_len = token_len; codex_size = duin_mae_params_inst.model.vq.codex_size
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    c_true = torch.tensor(np.eye(codex_size)[np.random.randint(0, codex_size, size=(batch_size, emb_len))], dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate duin_mae.
    duin_mae_inst = duin_mae(duin_mae_params_inst.model); print(duin_mae_inst)
    # Forward layers in `duin_mae_inst`.
    # c_pred - (batch_size, emb_len, codex_size); loss - torch.float32
    c_pred, loss = duin_mae_inst((X, c_true, subj_id))
    ## Forward duin_cls.
    # Instantiate params.
    duin_cls_params_inst = duin_cls_params(dataset=dataset)
    duin_cls_params_inst.model.n_subjects = n_subjects
    duin_cls_params_inst.model.tokenizer.n_channels = duin_cls_params_inst.model.n_channels = n_channels
    assert seq_len % duin_cls_params_inst.model.seg_len == 0; duin_cls_params_inst.model.seq_len = seq_len
    token_len = (
        duin_cls_params_inst.model.seq_len // duin_cls_params_inst.model.tokenizer.seg_len
    ); duin_cls_params_inst.model.encoder.emb_len = duin_cls_params_inst.model.tokenizer.token_len = token_len
    duin_cls_params_inst.model.cls.d_feature = (
        duin_cls_params_inst.model.encoder.emb_len * duin_cls_params_inst.model.encoder.d_model
    )
    duin_cls_params_inst.model.cls.n_labels = duin_cls_params_inst.model.n_labels = n_labels
    # Initialize input `X` & `y_true` & `subj_id`.
    # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels); subj_id - (batch_size, n_subjects)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    subj_id = torch.tensor(np.eye(n_subjects)[np.random.randint(0, n_subjects, size=(batch_size,))], dtype=torch.float32)
    # Instantiate duin_cls.
    duin_cls_inst = duin_cls(duin_cls_params_inst.model); print(duin_cls_inst)
    # Forward layers in `duin_cls_inst`.
    # y_pred - (batch_size, n_labels); loss - torch.float32
    y_pred, loss = duin_cls_inst((X, y_true, subj_id))

