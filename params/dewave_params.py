#!/usr/bin/env python3
"""
Created on 20:01, Aug. 1st, 2024

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
    "dewave_params",
]

# def dewave_params class
class dewave_params(DotDict):
    """
    This contains single object that generates a dictionary of parameters,
    which is provided to `dewave` on initialization.
    """
    # Initialize macro parameters.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `dewave_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(dewave_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = dewave_params._gen_model_params(dataset)
        # -- Train parameters
        self.train = dewave_params._gen_train_params(dataset)

        ## Do init iteration.
        dewave_params.iteration(self, 0)

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
        model_params.precision = getattr(torch, dewave_params._precision)\
            if hasattr(torch, dewave_params._precision) else torch.float32
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of input channels.
            model_params.n_channels = 10
            # The length of element sequence.
            model_params.seq_len = 1500
            # The number of output classes.
            model_params.n_labels = 61
        # Normal parameters related to other dataset.
        else:
            # The number of input channels.
            model_params.n_channels = 32
            # The length of element sequence.
            model_params.seq_len = 100
            # The number of output classes.
            model_params.n_labels = 10
        ## -- Tokenizer parameters
        model_params.tokenizer = dewave_params._gen_model_tokenizer_params(model_params)
        ## -- Encoder parameters
        model_params.encoder = dewave_params._gen_model_encoder_params(model_params)
        ## -- Classification parameters
        model_params.cls = dewave_params._gen_model_cls_params(model_params)
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
            # The number of filters of each convolution block.
            model_tokenizer_params.n_filters = [128, 128, 128, 128, 128]
            # The size of kernel of each convolution block.
            model_tokenizer_params.kernel_sizes = [9, 3, 3, 3, 3]
            # The number of strides of each convolution block.
            model_tokenizer_params.n_strides = [3, 2, 2, 2, 2]
            # The dilation rate of each convolution block.
            model_tokenizer_params.dilation_rates = [1, 1, 1, 1, 1]
            # The flag that indicates whether use batch-norm.
            model_tokenizer_params.use_bn = [True, True, True, True, True]
            # The flag that indicates whether use residual connection.
            model_tokenizer_params.use_res = [False, False, False, False, False]
            # The size of pooling of each convolution block.
            model_tokenizer_params.pool_sizes = [1, 1, 1, 1, 1]
        # Normal parameters related to other dataset.
        else:
            # The number of filters of each convolution block.
            model_tokenizer_params.n_filters = [512, 512, 512, 512, 512]
            # The size of kernel of each convolution block.
            model_tokenizer_params.kernel_sizes = [10, 3, 3, 3, 2]
            # The number of strides of each convolution block.
            model_tokenizer_params.n_strides = [3, 2, 2, 2, 2]
            # The dilation rate of each convolution block.
            model_tokenizer_params.dilation_rates = [1, 1, 1, 1, 1]
            # The flag that indicates whether use batch-norm.
            model_tokenizer_params.use_bn = [True, True, True, True, True]
            # The flag that indicates whether use residual connection.
            model_tokenizer_params.use_res = [False, False, False, False, False]
            # The size of pooling of each convolution block.
            model_tokenizer_params.pool_sizes = [1, 1, 1, 1, 1]
        # The dimensions of the embedding.
        model_tokenizer_params.d_model = model_tokenizer_params.n_filters[-1]
        # The length of token sequence.
        model_tokenizer_params.token_len = model_params.seq_len
        for n_stride_i in model_tokenizer_params.n_strides: model_tokenizer_params.token_len //= n_stride_i
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
        # The dimensions of the embedding.
        model_encoder_params.d_model = model_params.tokenizer.d_model
        # The length of embedding sequence.
        model_encoder_params.emb_len = model_params.tokenizer.token_len
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of attention blocks.
            model_encoder_params.n_blocks = 6
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
            model_encoder_params.d_ff = model_encoder_params.d_model * 4
            # The dropout probability of the hidden layer in ffn.
            model_encoder_params.ff_dropout = [0.2, 0.]
            # The flag that indicates whether execute normalization first.
            model_encoder_params.norm_first = False
        # Normal parameters related to other dataset.
        else:
            # The number of attention blocks.
            model_encoder_params.n_blocks = 2
            # The flag that indicates whether enable residual attention.
            model_encoder_params.res_attn = False
            # The number of attention heads.
            model_encoder_params.n_heads = 8
            # The dimensions of attention head.
            model_encoder_params.d_head = 64
            # The power base of rotation angle.
            model_encoder_params.rot_theta = None
            # The dropout probability of attention score.
            model_encoder_params.attn_dropout = 0.
            # The dropout probability of attention projection.
            model_encoder_params.proj_dropout = 0.
            # The dimensions of the hidden layer in ffn.
            model_encoder_params.d_ff = model_encoder_params.d_model * 4
            # The dropout probability of the hidden layer in ffn.
            model_encoder_params.ff_dropout = [0., 0.3]
            # The flag that indicates whether execute normalization first.
            model_encoder_params.norm_first = False

        # Return the final `model_encoder_params`.
        return model_encoder_params

    # def _gen_model_cls_params func
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
            # The dimensions of the hidden layers.
            model_cls_params.d_hidden = [128,]
            # The dropout probability after the hidden layers.
            model_cls_params.dropout = 0.
        # Normal parameters related to other dataset.
        else:
            # The dimensions of the hidden layers.
            model_cls_params.d_hidden = [128,]
            # The dropout probability after the hidden layers.
            model_cls_params.dropout = 0.
        # The dimensions of classification layer.
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
        # Precision parameter.
        train_params.precision = getattr(torch, dewave_params._precision)\
            if hasattr(torch, dewave_params._precision) else torch.float32
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
            train_params.lr_factors = (5e-6, 2e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            train_params.n_epochs = 200
            # Number of warmup epochs.
            train_params.warmup_epochs = 20
            # Number of batch size used in training process.
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (5e-6, 2e-4)

        # Return the final `train_params`.
        return train_params

if __name__ == "__main__":
    # Instantiate `dewave_params`.
    dewave_params_inst = dewave_params(dataset="seeg_he2023xuanwu")

