#!/usr/bin/env python3
"""
Created on 01:19, Dec. 3rd, 2023

@author: Norbert Zheng
"""
import torch
import os, re
import copy as cp
import numpy as np
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
    "brant_mae",
    "brant_cls",
]

# def brant_mae class
class brant_mae(nn.Module):
    """
    Brant model for neural signal prediction introduced by Zhang et al. 2023.

    [1] Zhang D, Yuan Z, Yang Y, et al. Brant: Foundation Model for Intracranial Neural Signal[C]
        //Thirty-seventh Conference on Neural Information Processing Systems. 2023.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `brant_mae` object.

        Args:
            params: DotDict - Model parameters initialized by brant_mae_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(brant_mae, self).__init__(**kwargs)

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
        self.tokenizer = LinearTokenizer(params=self.params.tokenizer)
        # Initialize time-encoder block.
        # encoder_t - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_model)
        self.encoder_t = TimeEncoder(params=self.params.encoder.time)
        # Initialize spatial-encoder block.
        # encoder_s - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_model)
        self.encoder_s = SpatialEncoder(params=self.params.encoder.spatial)
        # Initialize regression block.
        # rgs_block - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, seg_len)
        self.rgs_block = LinearRGSHead(params=self.params.rgs)

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
        # Try to load weights from specified checkpoint path.
        try:
            # Load weights according `path_ckpt`.
            if os.path.isfile(path_ckpt):
                self._load_weight_file(path_ckpt)
            else:
                self._load_weight_dir(path_ckpt)
        # Get exception, log information.
        except Exception as e:
            print((
                "ERROR: Fail loading pretrained weights from checkpoint ({}) due to exception ({}) in models.brant.brant_mae."
            ).format(path_ckpt, e))

    # def _load_weight_file func
    def _load_weight_file(self, path_ckpt):
        """
        Load model weights from the specified checkpoint file.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        assert os.path.isfile(path_ckpt)
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder_t": "encoder_t",
            "([^.]*\.)*encoder_s": "encoder_s",
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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.brant.brant_mae."
        ).format(modules, path_ckpt))

    # def _load_weight_dir func
    def _load_weight_dir(self, path_ckpt):
        """
        Load model weights from the specified checkpoint directory.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        assert os.path.isdir(path_ckpt)
        # Initialize `ckpt_*_dict`.
        ckpt_t_dict = torch.load(os.path.join(path_ckpt, "time_encoder.pt"))
        ckpt_s_dict = torch.load(os.path.join(path_ckpt, "channel_encoder.pt"))
        # Construct `model_dict` according to `ckpt_*_dict`.
        model_dict = {}; module_t_map = {
            "([^.]*\.)*module.input_embedding": "tokenizer",
            "([^.]*\.)*module.trans_enc": "encoder_t.trans_enc",
        }; module_s_map = {
            "([^.]*\.)*module.trans_enc": "encoder_s.trans_enc",
        }
        for parameter_name_i in ckpt_t_dict.keys():
            for module_src_i, module_trg_i in module_t_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_t_dict[parameter_name_i]; break
        for parameter_name_i in ckpt_s_dict.keys():
            for module_src_i, module_trg_i in module_s_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_s_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.brant.brant_mae."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward `brant_mae` to get the final predictions.

        Args:
            inputs: tuple - The input data, including [X,S,mask].

        Returns:
            X_reconstr: (batch_size, n_channels, n_segs, seg_len) - The reconstructed signals.
            loss: DotDict - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, n_segs, seg_len); S - (batch_size, n_channels, n_segs, n_bands)
        # mask - (batch_size, n_channels, n_segs)
        X = inputs[0]; S = inputs[1]; mask = inputs[2]
        # Forward tokenizer to get the tokenized tokens.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = self.tokenizer((X, S, mask))
        # Forward time-encoder to get the time-transformed embeddings.
        # E - (batch_size, n_channels, n_segs, d_model)
        E = self.encoder_t(T)
        # Forward spatial-encoder to get the spatial-transformed embeddings.
        # E - (batch_size, n_channels, n_segs, d_model)
        E = self.encoder_s(E)
        # Forward regression block to get the corresponding reconstruction.
        # X_reconstr - (batch_size, n_channels, n_segs, seg_len)
        X_reconstr = self.rgs_block(E)
        # Calculate the regression loss.
        # loss_rgs - torch.float32
        loss_rgs = self._loss_rgs(X_reconstr, X, weight=mask.to(dtype=X.dtype))
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
            value: (batch_size, n_channels, n_segs, seg_len) - Value of the object.
            target: (batch_size, n_channels, n_segs, seg_len) - Traget of the object.
            weight: (batch_size, n_channels, n_segs) - The regression weight.

        Returns:
            loss: torch.float32 - Loss between value and target.
        """
        # Calculate the regression loss.
        # loss - (batch_size, n_channels, n_segs, seg_len)
        loss = torch.square(target - value)
        # Average over all locations.
        # loss - (batch_size, n_channels, n_segs)
        loss = torch.mean(loss, dim=-1)
        # Average loss with uniform weight.
        # loss - torch.float32
        loss = torch.sum(loss * weight) / (torch.sum(weight) + 1e-12)\
            if weight is not None else torch.mean(loss)
        # Return the final `loss`.
        return loss

# def brant_cls class
class brant_cls(nn.Module):
    """
    Brant model for classification task introduced by Zhang et al. 2023.

    [1] Zhang D, Yuan Z, Yang Y, et al. Brant: Foundation Model for Intracranial Neural Signal[C]
        //Thirty-seventh Conference on Neural Information Processing Systems. 2023.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `brant_cls` object.

        Args:
            params: DotDict - Model parameters initialized by brant_cls_params, updated by params.iteration.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(brant_cls, self).__init__(**kwargs)

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
        self.tokenizer = LinearTokenizer(params=self.params.tokenizer)
        # Initialize time-encoder block.
        # encoder_t - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_model)
        self.encoder_t = TimeEncoder(params=self.params.encoder.time)
        # Initialize spatial-encoder block.
        # encoder_s - (batch_size, n_channels, n_segs, d_model) -> (batch_size, n_channels, n_segs, d_model)
        self.encoder_s = SpatialEncoder(params=self.params.encoder.spatial)
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
        # Try to load weights from specified checkpoint path.
        try:
            # Load weights according `path_ckpt`.
            if os.path.isfile(path_ckpt):
                self._load_weight_file(path_ckpt)
            else:
                self._load_weight_dir(path_ckpt)
        # Get exception, log information.
        except Exception as e:
            print((
                "ERROR: Fail loading pretrained weights from checkpoint ({}) due to exception ({}) in models.brant.brant_cls."
            ).format(path_ckpt, e))

    # def _load_weight_file func
    def _load_weight_file(self, path_ckpt):
        """
        Load model weights from the specified checkpoint file.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        assert os.path.isfile(path_ckpt)
        # Initialize `ckpt_dict`.
        ckpt_dict = torch.load(path_ckpt)
        # Construct `model_dict` according to `ckpt_dict`.
        model_dict = {}; module_map = {
            "([^.]*\.)*tokenizer": "tokenizer",
            "([^.]*\.)*encoder_t": "encoder_t",
            "([^.]*\.)*encoder_s": "encoder_s",
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
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.brant.brant_cls."
        ).format(modules, path_ckpt))

    # def _load_weight_dir func
    def _load_weight_dir(self, path_ckpt):
        """
        Load model weights from the specified checkpoint directory.

        Args:
            path_ckpt: str - The path of the spcified checkpoint.

        Returns:
            None
        """
        assert os.path.isdir(path_ckpt)
        # Initialize `ckpt_*_dict`.
        ckpt_t_dict = torch.load(os.path.join(path_ckpt, "time_encoder.pt"))
        ckpt_s_dict = torch.load(os.path.join(path_ckpt, "channel_encoder.pt"))
        # Construct `model_dict` according to `ckpt_*_dict`.
        model_dict = {}; module_t_map = {
            "([^.]*\.)*module.input_embedding": "tokenizer",
            "([^.]*\.)*module.trans_enc": "encoder_t.trans_enc",
        }; module_s_map = {
            "([^.]*\.)*module.trans_enc": "encoder_s.trans_enc",
        }
        for parameter_name_i in ckpt_t_dict.keys():
            for module_src_i, module_trg_i in module_t_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_t_dict[parameter_name_i]; break
        for parameter_name_i in ckpt_s_dict.keys():
            for module_src_i, module_trg_i in module_s_map.items():
                if re.compile(module_src_i).match(parameter_name_i) is not None:
                    parameter_rename_i = re.sub(module_src_i, module_trg_i, parameter_name_i)
                    model_dict[parameter_rename_i] = ckpt_s_dict[parameter_name_i]; break
        for key_i in model_dict.keys():
            assert key_i in self.state_dict().keys()
        assert len(model_dict.keys()) > 0; self.load_state_dict(model_dict, strict=False)
        # Log information related to parameter load.
        modules = sorted(set([key_i.split(".")[0] for key_i in model_dict.keys()]))
        print((
            "INFO: Complete loading pretrained weights of modules ({}) from checkpoint ({}) in models.brant.brant_cls."
        ).format(modules, path_ckpt))

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `brainbert`.

        Args:
            inputs: tuple - The input data, including [X,y_true,S].

        Returns:
            y_pred: (batch_size, n_labels) - The transformed prediction.
            loss: DotDict - The corresponding loss.
        """
        # Initialize components of inputs.
        # X - (batch_size, n_channels, n_segs, seg_len); y_true - (batch_size, n_labels)
        # S - (batch_size, n_channels, n_segs, n_bands)
        X = inputs[0]; y_true = inputs[1]; S = inputs[2]
        # Forward tokenizer to get the tokenized tokens.
        # T - (batch_size, n_channels, n_segs, d_model)
        T = self.tokenizer((X, S, None))
        # Forward time-encoder to get the time-transformed embeddings.
        # E - (batch_size, n_channels, n_segs, d_model)
        E = self.encoder_t(T)
        # Forward spatial-encoder to get the spatial-transformed embeddings.
        # E - (batch_size, n_channels, n_segs, d_model)
        E = self.encoder_s(E)
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
    # local dep
    import utils.model.torch
    from params.brant_params import brant_mae_params, brant_cls_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"; batch_size = 32; seq_len = 6000; n_channels = 10; n_bands = 8; n_labels = 61
    # Initialize base & pt_ckpt path.
    base = os.path.join(os.getcwd(), os.pardir, os.pardir)
    path_ckpt = os.path.join(base, "pretrains", "brant", "brant-yzz673")

    # Initialize random seeds.
    utils.model.torch.set_seeds(42)

    ## Forward brant_mae.
    # Instantiate params.
    brant_mae_params_inst = brant_mae_params(dataset=dataset, model_type="large")
    brant_mae_params_inst.model.encoder.n_channels = brant_mae_params_inst.model.n_channels = n_channels
    brant_mae_params_inst.model.tokenizer.n_bands = brant_mae_params_inst.model.n_bands = n_bands
    brant_mae_params_inst.model.seq_len = seq_len
    n_segs = (
        brant_mae_params_inst.model.seq_len // brant_mae_params_inst.model.seg_len
    ); assert brant_mae_params_inst.model.tokenizer.max_segs >= n_segs
    brant_mae_params_inst.model.encoder.n_segs = brant_mae_params_inst.model.n_segs = n_segs
    # Initialize input `X` & `S` & `mask`.
    # X - (batch_size, n_channels, n_segs, seg_len); S - (batch_size, n_channels, n_segs, n_bands)
    # mask - (batch_size, n_channels, n_segs)
    seg_len = brant_mae_params_inst.model.seg_len
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    S = torch.rand((batch_size, n_channels, n_segs, n_bands), dtype=torch.float32)
    mask = (torch.rand((batch_size, n_channels, n_segs), dtype=torch.float32) < 0.5)
    # Instantiate brant_mae.
    brant_mae_inst = brant_mae(brant_mae_params_inst.model); print(brant_mae_inst)
    # Load weights from specified pt_ckpt.
    brant_mae_inst.load_weight(path_ckpt=path_ckpt)
    # Forward layers in `brant_mae_inst`.
    # X_reconstr - (batch_size, n_channels, n_segs, seg_len); loss - DotDict
    X_reconstr, loss = brant_mae_inst((X, S, mask))
    ## Forward brant_cls.
    # Instantiate params.
    brant_cls_params_inst = brant_cls_params(dataset=dataset, model_type="large")
    brant_cls_params_inst.model.encoder.n_channels = brant_cls_params_inst.model.n_channels = n_channels
    brant_cls_params_inst.model.tokenizer.n_bands = brant_cls_params_inst.model.n_bands = n_bands
    brant_cls_params_inst.model.seq_len = seq_len
    n_segs = (
        brant_cls_params_inst.model.seq_len // brant_cls_params_inst.model.seg_len
    ); assert brant_cls_params_inst.model.tokenizer.max_segs >= n_segs
    brant_cls_params_inst.model.encoder.n_segs = brant_cls_params_inst.model.n_segs = n_segs
    brant_cls_params_inst.model.cls.d_feature = (
        brant_cls_params_inst.model.encoder.n_channels *\
        brant_cls_params_inst.model.encoder.n_segs *\
        brant_cls_params_inst.model.encoder.d_model
    )
    brant_cls_params_inst.model.cls.n_labels = brant_cls_params_inst.model.n_labels = n_labels
    # Initialize input `X` & `y_true`.
    # X - (batch_size, n_channels, n_segs, seg_len); y_true - (batch_size, n_labels)
    # S - (batch_size, n_channels, n_segs, n_bands)
    seg_len = brant_cls_params_inst.model.seg_len
    X = torch.rand((batch_size, n_channels, n_segs, seg_len), dtype=torch.float32)
    y_true = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    S = torch.rand((batch_size, n_channels, n_segs, n_bands), dtype=torch.float32)
    # Instantiate brant_cls.
    brant_cls_inst = brant_cls(brant_cls_params_inst.model); print(brant_cls_inst)
    # Load weights from specified pt_ckpt.
    brant_cls_inst.load_weight(path_ckpt=path_ckpt)
    # Forward layers in `brant_cls_inst`.
    # y_pred - (batch_size, n_labels); loss - DotDict
    y_pred, loss = brant_cls_inst((X, y_true, S))

