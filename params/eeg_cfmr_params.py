#!/usr/bin/env python3
"""
Created on 14:56, Mar. 28th, 2024

@author: Norbert Zheng
"""
import torch
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
from utils import DotDict

__all__ = [
    "eeg_cfmr_params",
]

# def eeg_cfmr_params class
class eeg_cfmr_params(DotDict):
    """
    This contains single object that generates a dictionary of parameters,
    which is provided to `eeg_cfmr` on initialization.
    """
    # Initialize macro parameters.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `eeg_cfmr_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(eeg_cfmr_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = eeg_cfmr_params._gen_model_params(dataset)
        # -- Train parameters
        self.train = eeg_cfmr_params._gen_train_params(dataset)

        ## Do init iteration.
        eeg_cfmr_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## -- Train parameters
        # Calculate current learning rate.
        lr_min, lr_max = self.train.lr_factors
        # If `iteration` is smaller than `params.train.warmup_epochs`, gradually increase `lr`.
        if iteration < self.train.warmup_epochs:
            self.train.lr_i = lr_max * ((iteration + 1) / self.train.warmup_epochs)
        # After `config.warmup_epochs`, decay the learning rate with half-cycle cosine after warmup.
        else:
            self.train.lr_i = lr_min + (lr_max - lr_min) * 0.5 *\
                (1. + np.cos(np.pi * (iteration - self.train.warmup_epochs) / (self.train.n_epochs - self.train.warmup_epochs)))

    """
    generate funcs
    """
    ## def _gen_model_* funcs
    # def _gen_model_params func
    @staticmethod
    def _gen_model_params(dataset):
        """
        Generate model parameters.
        """
        # Initialize `model_params`.
        model_params = DotDict()

        ## -- Normal parameters
        # The type of dataset.
        model_params.dataset = dataset
        # The device of model.
        model_params.device = torch.device("cpu")
        # Precision parameter.
        model_params.precision = getattr(torch, eeg_cfmr_params._precision)\
            if hasattr(torch, eeg_cfmr_params._precision) else torch.float32
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of channels.
            model_params.n_channels = 10
            # The length of element sequence.
            model_params.seq_len = 3000
            # The number of labels.
            model_params.n_labels = 61
        # Normal parameters related to other dataset.
        else:
            # The number of channels.
            model_params.n_channels = 8
            # The length of element sequence.
            model_params.seq_len = 100
            # The number of labels.
            model_params.n_labels = 10
        ## -- Tokenizer parameters
        model_params.tokenizer = eeg_cfmr_params._gen_model_tokenizer_params(model_params)
        ## -- Encoder parameters
        model_params.encoder = eeg_cfmr_params._gen_model_encoder_params(model_params)
        ## -- Classification parameters
        model_params.cls = eeg_cfmr_params._gen_model_cls_params(model_params)
        ## -- Additional parameters
        # The scale factor of cls loss.
        model_params.cls_loss_scale = 1.

        # Return the final `model_params`.
        return model_params

    # def _gen_model_tokenizer_params func
    @staticmethod
    def _gen_model_tokenizer_params(model_params):
        """
        Generate model.tokenizer parameters.
        """
        # Initialize `model_tokenizer_params`.
        model_tokenizer_params = DotDict()

        ## -- Normal parameters
        # The number of channels.
        model_tokenizer_params.n_channels = model_params.n_channels
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of convolution filters.
            model_tokenizer_params.n_filters = [128,]
            # The size of convolution kernel.
            model_tokenizer_params.kernel_sizes = [25,]
            # The number of convolution strides.
            model_tokenizer_params.n_strides = [1,]
            # The size of pooling kernel.
            model_tokenizer_params.pool_size = 75
            # The number of pooling strides.
            model_tokenizer_params.pool_stride = 15
            # The ratio of dropout.
            model_tokenizer_params.dropout = 0.
        # Normal parameters related to other dataset.
        else:
            # The number of convolution filters.
            model_tokenizer_params.n_filters = [128,]
            # The size of convolution kernel.
            model_tokenizer_params.kernel_sizes = [25,]
            # The number of convolution strides.
            model_tokenizer_params.n_strides = [1,]
            # The size of pooling kernel.
            model_tokenizer_params.pool_size = 75
            # The number of pooling strides.
            model_tokenizer_params.pool_stride = 15
            # The ratio of dropout.
            model_tokenizer_params.dropout = 0.
        # The dimensions of model embedding.
        model_tokenizer_params.d_model = model_tokenizer_params.n_filters[-1]
        # The length of token sequence.
        model_tokenizer_params.token_len = (
            model_params.seq_len // (np.prod(model_tokenizer_params.n_strides) * model_tokenizer_params.pool_stride)
        )

        # Return the final `model_tokenizer_params`.
        return model_tokenizer_params

    # def _gen_model_encoder_params func
    @staticmethod
    def _gen_model_encoder_params(model_params):
        """
        Generate model.encoder parameters.
        """
        # Initialize `model_encoder_params`.
        model_encoder_params = DotDict()

        ## -- Normal parameters
        # The dimensions of model embedding.
        model_encoder_params.d_model = model_params.tokenizer.d_model
        # The length of embedding sequence.
        model_encoder_params.emb_len = model_params.tokenizer.token_len
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of attention blocks.
            model_encoder_params.n_blocks = 4
            # The flag that indicates whether enable residual attention.
            model_encoder_params.res_attn = False
            # The number of attention heads.
            model_encoder_params.n_heads = 8
            # The dimensions of attention head.
            model_encoder_params.d_head = 64
            # The power base of rotation angle.
            model_encoder_params.rot_theta = None
            # The dropout probability of attention score.
            model_encoder_params.attn_dropout = 0.2
            # The dropout probability of attention projection.
            model_encoder_params.proj_dropout = 0.
            # The dimensions of the hidden layer in ffn.
            model_encoder_params.d_ff = model_encoder_params.d_model * 2
            # The dropout probability of the hidden layer in ffn.
            model_encoder_params.ff_dropout = [0.2, 0.]
            # The flag that indicates whether execute normalization first.
            model_encoder_params.norm_first = False
        # Normal parameters related to other dataset.
        else:
            # The number of attention blocks.
            model_encoder_params.n_blocks = 4
            # The flag that indicates whether enable residual attention.
            model_encoder_params.res_attn = False
            # The number of attention heads.
            model_encoder_params.n_heads = 8
            # The dimensions of attention head.
            model_encoder_params.d_head = 64
            # The power base of rotation angle.
            model_encoder_params.rot_theta = None
            # The dropout probability of attention score.
            model_encoder_params.attn_dropout = 0.2
            # The dropout probability of attention projection.
            model_encoder_params.proj_dropout = 0.
            # The dimensions of the hidden layer in ffn.
            model_encoder_params.d_ff = model_encoder_params.d_model * 2
            # The dropout probability of the hidden layer in ffn.
            model_encoder_params.ff_dropout = [0.2, 0.]
            # The flag that indicates whether execute normalization first.
            model_encoder_params.norm_first = False

        # Return the final `model_encoder_params`.
        return model_encoder_params

    # def _gen_model_cls_params func
    @staticmethod
    def _gen_model_cls_params(model_params):
        """
        Generate model.cls parameters.
        """
        # Initialize `model_cls_params`.
        model_cls_params = DotDict()

        ## -- Normal parameters
        # The dimensions of feature embedding.
        model_cls_params.d_feature = (
            model_params.encoder.emb_len * model_params.encoder.d_model
        )
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The dimensions of hidden layers.
            model_cls_params.d_hidden = [128,]
            # The dropout ratio after hidden layers.
            model_cls_params.dropout = 0.
        # Normal parameters related to other dataset.
        else:
            # The dimensions of hidden layers.
            model_cls_params.d_hidden = [128,]
            # The dropout ratio after hidden layers.
            model_cls_params.dropout = 0.
        # The number of labels.
        model_cls_params.n_labels = model_params.n_labels

        # Return the final `model_cls_params`.
        return model_cls_params

    ## def _gen_train_* funcs
    # def _gen_train_params func
    @staticmethod
    def _gen_train_params(dataset):
        """
        Generate train parameters.
        """
        # Initialize `train_params`.
        train_params = DotDict()

        ## -- Normal parameters
        # The type of dataset.
        train_params.dataset = dataset
        # The base path of project.
        train_params.base = None
        # The name of subject.
        train_params.subj = "011"
        # Precision parameter.
        train_params.precision = getattr(torch, eeg_cfmr_params._precision)\
            if hasattr(torch, eeg_cfmr_params._precision) else torch.float32
        # Whether use graph mode or eager mode.
        train_params.use_graph_mode = False
        # The ratio of train dataset. The rest is test dataset.
        train_params.train_ratio = 0.8
        # Size of buffer used in shuffle.
        train_params.buffer_size = int(1e4)
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if train_params.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            train_params.n_epochs = 200
            # Number of warmup epochs.
            train_params.warmup_epochs = 20
            # Number of batch size used in training process.
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (2e-5, 2e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (5e-6, 2e-4)

        # Return the final `train_params`.
        return train_params

if __name__ == "__main__":
    # Instantiate `eeg_cfmr_params`.
    eeg_cfmr_params_inst = eeg_cfmr_params(dataset="seeg_he2023xuanwu")

