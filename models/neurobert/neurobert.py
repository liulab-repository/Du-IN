#!/usr/bin/env python3
"""
Created on 20:04, Mar. 28th, 2024

@author: Norbert Zheng
"""
import re, torch
import copy as cp
import torch.nn as nn
import torch.nn.functional as F
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from layers import *
else:
    from .layers import *
from utils import DotDict

__all__ = [
    "neurobert_mae",
    "neurobert_cls",
]

# def neurobert_mae class
class neurobert_mae(nn.Module):
    """
    Neuro-BERT model for neural signal prediction introduced by Song et al. 2022.

    [1] Wu D, Li S, Yang J, et al. Neuro-BERT: Rethinking Masked Autoencoding for Self-Supervised Neurological Pretraining[J].
        IEEE Journal of Biomedical and Health Informatics, 2024.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `neurobert_mae` object.

        Args:
            params: DotDict - Model parameters initialized by neurobert_mae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(neurobert_mae, self).__init__(**kwargs)

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
        self.tokenizer = ConvTokenizer(params=self.params.tokenizer)
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
        # Initialize decoder block.
        # decoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.decoder = nn.Sequential(
            LambdaLayer(func=(lambda x: self.emb_time(x))),
            TransformerStack(self.params.decoder), LambdaLayer(func=(lambda x: x[0])),
        )
        # Initialize regression block.
        # rgs_block - (batch_size, token_len, d_model) -> (batch_size, seq_len, n_channels)
        self.rgs_block = TimeRGSHead(params=self.params.rgs)

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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.neurobert.neurobert_mae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `neurobert_mae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,].

        Returns:
            X_reconstr: (batch_size, seq_len, n_channels) - The reconstructed signals.
            loss: torch.float32 - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels)
        X = inputs[0]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X); E_init = T
        # Generate mask according to the init embedding `E`.
        # mask - (batch_size, emb_len)
        mask = self.gen_mask(E_init, mask_ratio=self.params.mask_ratio)
        # Get the masked embedding `E_masked` according to `mask`.
        # mask_emb - (batch_size, emb_len, d_model)
        mask_emb = self.mask_emb[None,None,...].expand(*mask.shape, -1)
        # E_masked - (batch_size, emb_len, d_model)
        E_masked = (E_init * (1. - mask[...,None].to(dtype=E_init.dtype)) + mask_emb * mask[...,None].to(dtype=E_init.dtype))
        # Forward encoder block to get time-aligned token sequence.
        # E - (batch_size, emb_len, d_model)
        E = self.encoder(E_masked)
        # Forward regression block to get the corresponding reconstructon.
        # X_reconstr - (batch_size, seq_len, n_channels)
        X_reconstr = self.rgs_block(self.decoder(E))
        # Calculate the regression loss.
        # loss_rgs - torch.float32
        loss_rgs = self._loss_rgs(X_reconstr, X, weight=None)
        # Calculate the total loss.
        # loss_total - torch.float32
        loss_total = (
            self.params.rgs_loss_scale * loss_rgs
        )
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss_total,
            "rgs": loss_rgs,
        })
        # Return the final `X_reconstr` & `loss`.
        return X_reconstr, loss

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

# def neurobert_cls class
class neurobert_cls(nn.Module):
    """
    Neuro-BERT model for classification task introduced by Song et al. 2022.

    [1] Wu D, Li S, Yang J, et al. Neuro-BERT: Rethinking Masked Autoencoding for Self-Supervised Neurological Pretraining[J].
        IEEE Journal of Biomedical and Health Informatics, 2024.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `neurobert_cls` object.

        Args:
            params: DotDict - Model parameters initialized by neurobert_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(neurobert_cls, self).__init__(**kwargs)

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
        self.tokenizer = ConvTokenizer(params=self.params.tokenizer)
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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.neurobert.neurobert_cls."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `neurobert_cls` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,y_true].

        Returns:
            y_pred: (batch_size, n_labels) - The output labels.
            loss: DotDict - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels)
        X = inputs[0]; y_true = inputs[1]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # T - (batch_size, token_len, d_model)
        T = self.tokenizer(X)
        # Forward encoder block to get time-aligned embedding sequence.
        # E - (batch_size, emb_len, d_model)
        E = self.encoder(T)
        # Forward classification block to get the predictions.
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
        Calculate classification loss between value and target.

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
    from params.neurobert_params import neurobert_mae_params, neurobert_cls_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32; seq_len = 1200; n_channels = 10; n_labels = 61

    # Initialize random seeds.
    utils.model.torch.set_seeds(42)

    ## Forward neurobert_mae.
    # Instantiate params.
    neurobert_mae_params_inst = neurobert_mae_params(dataset=dataset)
    neurobert_mae_params_inst.model.tokenizer.n_channels = neurobert_mae_params_inst.model.n_channels = n_channels
    neurobert_mae_params_inst.model.rgs.n_channels = n_channels
    neurobert_mae_params_inst.model.seq_len = seq_len
    token_len = (
        seq_len // np.prod(neurobert_mae_params_inst.model.tokenizer.n_strides)
    ); neurobert_mae_params_inst.model.encoder.emb_len = neurobert_mae_params_inst.model.tokenizer.token_len = token_len
    neurobert_mae_params_inst.model.rgs.emb_len = neurobert_mae_params_inst.model.decoder.token_len = token_len
    # Initialize input `X`.
    # X - (batch_size, seq_len, n_channels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    # Instantiate neurobert_mae.
    neurobert_mae_inst = neurobert_mae(params=neurobert_mae_params_inst.model); print(neurobert_mae_inst)
    # Forward layers in `neurobert_mae_inst`.
    # X_reconstr - (batch_size, seq_len, n_channels); loss - DotDict
    X_reconstr, loss = neurobert_mae_inst((X,))
    ## Forward neurobert_cls.
    # Instantiate params.
    neurobert_cls_params_inst = neurobert_cls_params(dataset=dataset)
    neurobert_cls_params_inst.model.tokenizer.n_channels = neurobert_cls_params_inst.model.n_channels = n_channels
    neurobert_cls_params_inst.model.seq_len = seq_len
    token_len = (
        seq_len // np.prod(neurobert_cls_params_inst.model.tokenizer.n_strides)
    ); neurobert_cls_params_inst.model.encoder.emb_len = neurobert_cls_params_inst.model.tokenizer.token_len = token_len
    neurobert_cls_params_inst.model.cls.d_feature = (
        neurobert_cls_params_inst.model.encoder.emb_len * neurobert_cls_params_inst.model.encoder.d_model
    )
    neurobert_cls_params_inst.model.cls.n_labels = neurobert_cls_params_inst.model.n_labels = n_labels
    # Initialize input `X` & `y_true`.
    # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    # Instantiate neurobert_cls.
    neurobert_cls_inst = neurobert_cls(params=neurobert_cls_params_inst.model); print(neurobert_cls_inst)
    # Forward layers in `neurobert_cls_inst`.
    # y_pred - (batch_size, n_labels); loss - DotDict
    y_pred, loss = neurobert_cls_inst((X, y_true))

