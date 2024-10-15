#!/usr/bin/env python3
"""
Created on 15:10, Nov. 30th, 2023

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
    "brainbert_mae",
    "brainbert_cls",
]

class brainbert_mae(nn.Module):
    """
    BrainBERT model for neural signal prediction introduced by Wang et al. 2023.

    [1] Wang C, Subramaniam V, Yaari A U, et al. BrainBERT: Self-supervised representation
        learning for intracranial recordings[J]. arXiv preprint arXiv:2302.14367, 2023.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `brainbert_mae` object.

        Args:
            params: DotDict - Model parameters initialized by brainbert_mae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(brainbert_mae, self).__init__(**kwargs)

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
        # tokenizer - (batch_size, freq_len, n_freqs) -> (batch_size, token_len, d_model)
        self.tokenizer = SpecTokenizer(params=self.params.tokenizer)
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = Encoder(params=self.params.encoder)
        # Initialize regression block.
        # rgs_block - (batch_size, token_len, d_model) -> (batch_size, freq_len, n_freqs)
        self.rgs_block = SpecRGSHead(params=self.params.rgs)

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
        ckpt_dict = torch.load(path_ckpt) if ("brainbert-czlwang" not in path_ckpt) else torch.load(path_ckpt)["model"]
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
            "([^.]*\.)*input_encoding": "tokenizer.input_encoding",
            "([^.]*\.)*transformer": "encoder.transformer",
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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.brainbert.brainbert_mae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `brainbert_rgs`.

        Args:
            inputs: tuple - The input data, including [S,mask,S_masked].

        Returns:
            S_reconstr: (batch_size, n_channels, freq_len, n_freqs) - The reconstruction spectrum series.
            loss: DotDict - The corresponding loss.
        """
        # Initialize `S` & `mask` & `S_masked` from `inputs`.
        # S* - (batch_size, n_channels, freq_len, n_freqs); mask - (batch_size, n_channels, freq_len, n_freqs)
        S = inputs[0]; mask = inputs[1]; S_masked = inputs[2]
        # Initialize `batch_size` & `n_channels` & `freq_len` & `n_freqs` from `S_masked`.
        batch_size, n_channels, freq_len, n_freqs = S_masked.shape
        # Reshape `S_masked` to prepare input for tokenizer.
        # S_masked - (batch_size * n_channels, freq_len, n_freqs)
        S_masked = torch.reshape(S_masked, shape=(-1, *S_masked.shape[2:]))
        # Forward the tokenizer to get the token sequence.
        # T - (batch_size * n_channels, token_len, d_model), where `token_len = freq_len`.
        T = self.tokenizer(S_masked)
        # Forward the encoder block to get the embedding sequence.
        # E - (batch_size * n_channels, emb_len, d_model), where `emb_len = freq_len`.
        E = self.encoder(T)
        # Forward the regression block to get the final regression series.
        # S_reconstr - (batch_size * n_channels, freq_len, n_freqs)
        S_reconstr = self.rgs_block(E)
        # Reshape `S_reconstr` to get the original shape.
        # S_reconstr - (batch_size, n_channels, freq_len, n_freqs)
        S_reconstr = torch.reshape(S_reconstr, shape=(batch_size, n_channels, *S_reconstr.shape[1:]))
        # Calculate the regression loss.
        # loss_rgs - torch.float32
        loss_rgs = self._loss_rgs(S_reconstr, S, weight=mask.to(dtype=S.dtype))
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
        # Return the final `S_reconstr` & `loss`.
        return S_reconstr, loss

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
            value: (batch_size, n_channels, freq_len, n_freqs) - Value of the object.
            target: (batch_size, n_channels, freq_len, n_freqs) - Traget of the object.
            weight: (batch_size, n_channels, freq_len, n_freqs) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the regression loss.
        # loss - (batch_size, n_channels, freq_len, n_freqs)
        loss = torch.square(target - value)
        # Weight loss according to weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

# def brainbert_cls class
class brainbert_cls(nn.Module):
    """
    BrainBERT model for classification task introduced by Wang et al. 2023.

    [1] Wang C, Subramaniam V, Yaari A U, et al. BrainBERT: Self-supervised representation
        learning for intracranial recordings[J]. arXiv preprint arXiv:2302.14367, 2023.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `brainbert_cls` object.

        Args:
            params: DotDict - Model parameters initialized by brainbert_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(brainbert_cls, self).__init__(**kwargs)

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
        # tokenizer - (batch_size, freq_len, n_freqs) -> (batch_size, token_len, d_model)
        self.tokenizer = SpecTokenizer(params=self.params.tokenizer)
        # Initialize encoder block.
        # encoder - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.encoder = Encoder(params=self.params.encoder)
        # Initialize classification block.
        # cls_block - (batch_size, n_channels, emb_len, d_model) -> (batch_size, n_labels)
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
        ckpt_dict = torch.load(path_ckpt) if ("brainbert-czlwang" not in path_ckpt) else torch.load(path_ckpt)["model"]
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder": "encoder",
            "([^.]*\.)*input_encoding": "tokenizer.input_encoding",
            "([^.]*\.)*transformer": "encoder.transformer",
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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.brainbert.brainbert_cls."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `brainbert_cls` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [S,y_true].

        Returns:
            y_pred: (batch_size, n_labels) - The predicted labels.
            loss: DotDict - The corresponding loss.
        """
        # Initialize `S` & `y_true` from `inputs`.
        # S - (batch_size, n_channels, freq_len, n_freqs); y_true - (batch_size, n_labels)
        S = inputs[0]; y_true = inputs[1]
        # Initialize `batch_size` & `n_channels` & `freq_len` & `n_freqs` from `S`.
        batch_size, n_channels, freq_len, n_freqs = S.shape
        # Reshape `S` to prepare input for tokenizer.
        # S - (batch_size * n_channels, freq_len, n_freqs)
        S = torch.reshape(S, shape=(-1, *S.shape[2:]))
        # Forward the tokenizer to get the token sequence.
        # T - (batch_size * n_channels, token_len, d_model)
        T = self.tokenizer(S)
        # Forward the encoder block to get the embedding sequence.
        # E - (batch_size * n_channels, emb_len, d_model)
        E = GradScaler.apply(self.encoder(T), self.params.grad_scale)
        # Reshape `E` to prepare input for classification block.
        # E - (batch_size, n_channels, emb_len, d_model)
        E = torch.reshape(E, shape=(batch_size, n_channels, *E.shape[1:]))
        # Forward the classification block to get the final predictions.
        # y_pred - (batch_size, n_labels)
        emb_len = E.shape[-2]; avg_center = (emb_len // 2)
        avg_left = avg_center + self.params.cls.avg_range[0]
        avg_right = avg_center + self.params.cls.avg_range[1]
        y_pred = self.cls_block(torch.mean(E[...,avg_left:avg_right,:], dim=-2))
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
    from params.brainbert_params import brainbert_mae_params, brainbert_cls_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32; n_channels = 10; freq_len = 160; n_freqs = 40; n_labels = 61
    # Initialize base & pt_ckpt path.
    base = os.path.join(os.getcwd(), os.pardir, os.pardir)
    path_ckpt = os.path.join(base, "pretrains", "brainbert", "brainbert-czlwang", "stft_large_pretrained.pth")

    ## Forward brainbert_mae.
    # Instantiate params.
    brainbert_mae_params_inst = brainbert_mae_params(dataset=dataset)
    brainbert_mae_params_inst.model.n_channels = n_channels
    brainbert_mae_params_inst.model.freq_len = freq_len
    brainbert_mae_params_inst.model.encoder.emb_len = brainbert_mae_params_inst.model.tokenizer.token_len = freq_len
    brainbert_mae_params_inst.model.n_freqs = n_freqs
    brainbert_mae_params_inst.model.rgs.n_freqs = brainbert_mae_params_inst.model.tokenizer.n_freqs = n_freqs
    # Initialize input `S` & `mask` & `S_masked`.
    # S* - (batch_size, n_channels, freq_len, n_freqs); mask - (batch_size, n_channels, freq_len, n_freqs)
    S = torch.rand((batch_size, n_channels, freq_len, n_freqs), dtype=torch.float32)
    mask = torch.tensor((np.random.uniform(low=0., high=1., size=(batch_size, n_channels, freq_len, n_freqs)) < 0.5), dtype=torch.bool)
    S_masked = torch.rand((batch_size, n_channels, freq_len, n_freqs), dtype=torch.float32)
    # Instantiate brainbert_mae.
    brainbert_mae_inst = brainbert_mae(params=brainbert_mae_params_inst.model); print(brainbert_mae_inst)
    # Load weights from specified pt_ckpt.
    brainbert_mae_inst.load_weight(path_ckpt=path_ckpt)
    # Forward layers in `brainbert_mae_inst`.
    # S_reconstr - (batch_size, n_channels, freq_len, n_freqs); loss - DotDict
    S_reconstr, loss = brainbert_mae_inst((S, mask, S_masked))
    ## Forward brainbert_cls.
    # Instantiate params.
    brainbert_cls_params_inst = brainbert_cls_params(dataset=dataset)
    brainbert_cls_params_inst.model.n_channels = n_channels
    brainbert_cls_params_inst.model.cls.d_feature = (
        n_channels * brainbert_cls_params_inst.model.encoder.d_model
    )
    brainbert_cls_params_inst.model.freq_len = freq_len
    brainbert_cls_params_inst.model.encoder.emb_len = brainbert_cls_params_inst.model.tokenizer.token_len = freq_len
    brainbert_cls_params_inst.model.tokenizer.n_freqs = brainbert_cls_params_inst.model.n_freqs = n_freqs
    brainbert_cls_params_inst.model.cls.n_labels = brainbert_cls_params_inst.model.n_labels = n_labels
    # Initialize input `S` & `y_true`.
    # S - (batch_size, n_channels, freq_len, n_freqs); y_true - (batch_size, n_labels)
    S = torch.rand((batch_size, n_channels, freq_len, n_freqs), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    # Instantiate brainbert_cls.
    brainbert_cls_inst = brainbert_cls(params=brainbert_cls_params_inst.model); print(brainbert_cls_inst)
    # Load weights from specified pt_ckpt.
    brainbert_cls_inst.load_weight(path_ckpt=path_ckpt)
    # Forward layers in `brainbert_cls_inst`.
    # y_pred - (batch_size, n_labels); loss - DotDict
    y_pred, loss = brainbert_cls_inst((S, y_true))

