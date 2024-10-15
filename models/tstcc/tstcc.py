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
    "tstcc_contra",
    "tstcc_cls",
]

# def tstcc_contra class
class tstcc_contra(nn.Module):
    """
    TS-TCC model for contrastive pretrain introduced by Eldele et al. 2021.

    [1] Eldele E, Ragab M, Chen Z, et al. Time-series representation learning via temporal and contextual contrasting[J].
        arXiv preprint arXiv:2106.14112, 2021.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `tstcc_contra` object.

        Args:
            params: DotDict - Model parameters initialized by tstcc_contra_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(tstcc_contra, self).__init__(**kwargs)

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
        # Initialize temporal-contrastive block.
        # tc_block - (batch_size, token_len, d_model) -> loss
        self.tc_block = TemporalContrastiveBlock(params=self.params.contra)

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
            "([^.]*\.)*tc_block": "tc_block",
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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.tstcc.tstcc_contra."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `tstcc_contra` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,].

        Returns:
            y_pred: (batch_size, n_labels) - The output labels.
            loss: DotDict - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, seq_len, n_channels)
        X_aug1 = inputs[0]; X_aug2 = inputs[1]
        # Forward tokenizer to get the tokenized tokens, this process may reduce the resolution.
        # T_aug1 - (batch_size, token_len, d_model)
        T_aug1 = self.tokenizer(X_aug1); T_aug2 = self.tokenizer(X_aug2)
        # Forward to get loss.
        loss = self.tc_block((T_aug1, T_aug2))
        # Calculate the final loss.
        # loss - DotDict
        loss = DotDict({
            "total": loss,
            "contra": loss,
        })
        # Return the final loss.
        return loss

# def tstcc_cls class
class tstcc_cls(nn.Module):
    """
    TS-TCC model for classification task introduced by Eldele et al. 2021.

    [1] Eldele E, Ragab M, Chen Z, et al. Time-series representation learning via temporal and contextual contrasting[J].
        arXiv preprint arXiv:2106.14112, 2021.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `tstcc_cls` object.

        Args:
            params: DotDict - Model parameters initialized by tstcc_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(tstcc_cls, self).__init__(**kwargs)

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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.tstcc.tstcc_cls."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `tstcc_cls` to get the final predictions.

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
        # Forward classification block to get the predictions.
        # y_pred - (batch_size, n_labels)
        y_pred = self.cls_block(T)
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
    from params.tstcc_params import tstcc_contra_params, tstcc_cls_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32; seq_len = 300; n_channels = 10; n_labels = 61

    # Initialize random seeds.
    utils.model.torch.set_seeds(42)

    ## Forward tstcc_contra.
    # Instantiate params.
    tstcc_contra_params_inst = tstcc_contra_params(dataset=dataset)
    tstcc_contra_params_inst.model.n_channels = n_channels
    tstcc_contra_params_inst.model.tokenizer.n_channels = n_channels
    tstcc_contra_params_inst.model.seq_len = seq_len
    tstcc_contra_params_inst.model.tokenizer.token_len = (seq_len // (
        np.prod(tstcc_contra_params_inst.model.tokenizer.n_strides) * np.prod(tstcc_contra_params_inst.model.tokenizer.pool_strides)
    ))
    # Initialize input `X` & `y_true`.
    # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    # Instantiate tstcc_contra.
    tstcc_contra_inst = tstcc_contra(params=tstcc_contra_params_inst.model); print(tstcc_contra_inst)
    # Forward layers in `tstcc_contra_inst`.
    # loss - DotDict
    loss = tstcc_contra_inst((X, y_true))
    ## Forward tstcc_cls.
    # Instantiate params.
    tstcc_cls_params_inst = tstcc_cls_params(dataset=dataset)
    tstcc_cls_params_inst.model.n_channels = n_channels
    tstcc_cls_params_inst.model.tokenizer.n_channels = n_channels
    tstcc_cls_params_inst.model.seq_len = seq_len
    tstcc_cls_params_inst.model.tokenizer.token_len = (seq_len // (
        np.prod(tstcc_cls_params_inst.model.tokenizer.n_strides) * np.prod(tstcc_cls_params_inst.model.tokenizer.pool_strides)
    ))
    tstcc_cls_params_inst.model.cls.d_feature = (
        tstcc_cls_params_inst.model.tokenizer.token_len * tstcc_cls_params_inst.model.tokenizer.d_model
    )
    tstcc_cls_params_inst.model.cls.n_labels = tstcc_cls_params_inst.model.n_labels = n_labels
    # Initialize input `X` & `y_true`.
    # X - (batch_size, seq_len, n_channels); y_true - (batch_size, n_labels)
    X = torch.rand((batch_size, seq_len, n_channels), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    # Instantiate tstcc_cls.
    tstcc_cls_inst = tstcc_cls(params=tstcc_cls_params_inst.model); print(tstcc_cls_inst)
    # Forward layers in `tstcc_cls_inst`.
    # y_pred - (batch_size, n_labels); loss - DotDict
    y_pred, loss = tstcc_cls_inst((X, y_true))

