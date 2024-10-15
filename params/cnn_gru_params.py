#!/usr/bin/env python3
"""
Created on 23:33, Mar. 29th, 2024

@author: Norbert Zheng
"""
import torch
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.pardir)
from utils import DotDict

__all__ = [
    "cnn_gru_params",
]

# def cnn_gru_params class
class cnn_gru_params(DotDict):
    """
    This contains single object that generates a dictionary of parameters,
    which is provided to `cnn_gru` on initialization.
    """
    # Initialize macro parameters.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `cnn_gru_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(cnn_gru_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = cnn_gru_params._gen_model_params(dataset)
        # -- Train parameters
        self.train = cnn_gru_params._gen_train_params(dataset)

        ## Do init iteration.
        cnn_gru_params.iteration(self, 0)

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
        model_params.precision = getattr(torch, cnn_gru_params._precision)\
            if hasattr(torch, cnn_gru_params._precision) else torch.float32
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
        ## -- CNN parameters
        model_params.cnn = cnn_gru_params._gen_model_cnn_params(model_params)
        ## -- GRU parameters
        model_params.gru = cnn_gru_params._gen_model_gru_params(model_params)
        ## -- Classification parameters
        model_params.cls = cnn_gru_params._gen_model_cls_params(model_params)
        ## -- Additional parameters
        # The scale factor of cls loss.
        model_params.cls_loss_scale = 1.

        # Return the final `model_params`.
        return model_params

    # def _gen_model_cnn_params func
    @staticmethod
    def _gen_model_cnn_params(model_params):
        """
        Generate model.cnn parameters.
        """
        # Initialize `model_cnn_params`.
        model_cnn_params = DotDict()

        ## -- Normal parameters
        # The number of channels.
        model_cnn_params.n_channels = model_params.n_channels
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of convolution filters.
            model_cnn_params.n_filters = [128,]
            # The size of convolution kernel.
            model_cnn_params.kernel_sizes = [5,]
            # The number of convolution strides.
            model_cnn_params.n_strides = [3,]
        # Normal parameters related to other dataset.
        else:
            # The number of convolution filters.
            model_cnn_params.n_filters = [128,]
            # The size of convolution kernel.
            model_cnn_params.kernel_sizes = [5,]
            # The number of convolution strides.
            model_cnn_params.n_strides = [3,]
        # The dimensions of model embedding.
        model_cnn_params.d_model = model_cnn_params.n_filters[-1]
        # The length of token sequence.
        model_cnn_params.token_len = (
            model_params.seq_len // np.prod(model_cnn_params.n_strides)
        )

        # Return the final `model_cnn_params`.
        return model_cnn_params

    # def _gen_model_gru_params func
    @staticmethod
    def _gen_model_gru_params(model_params):
        """
        Generate model.gru parameters.
        """
        # Initialize `model_gru_params`.
        model_gru_params = DotDict()

        ## -- Normal parameters
        # The dimensions of model embedding.
        model_gru_params.d_model = model_params.cnn.d_model
        # The length of token sequence.
        model_gru_params.token_len = model_params.cnn.token_len
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of gru layers.
            model_gru_params.n_layers = 2
            # The dropout ratio of gru layers.
            model_gru_params.dropout = 0.2
            # The flag that indicates whether use BiGRU.
            model_gru_params.use_bigru = True
        # Normal parameters related to other dataset.
        else:
            # The number of gru layers.
            model_gru_params.n_layers = 2
            # The dropout ratio of gru layers.
            model_gru_params.dropout = 0.2
            # The flag that indicates whether use BiGRU.
            model_gru_params.use_bigru = True

        # Return the final `model_gru_params`.
        return model_gru_params

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
        model_cls_params.d_feature = model_params.gru.d_model if not model_params.gru.use_bigru else (2 * model_params.gru.d_model)
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
        train_params.precision = getattr(torch, cnn_gru_params._precision)\
            if hasattr(torch, cnn_gru_params._precision) else torch.float32
        # Whether use graph mode or eager mode.
        train_params.use_graph_mode = False
        # The ratio of train dataset. The rest is test dataset.
        train_params.train_ratio = 0.8
        # Size of buffer used in shuffle.
        train_params.buffer_size = int(1e4)
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if train_params.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            train_params.n_epochs = 500
            # Number of warmup epochs.
            train_params.warmup_epochs = 0
            # Number of batch size used in training process.
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (1e-3, 1e-3)
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
    # Instantiate `cnn_gru_params`.
    cnn_gru_params_inst = cnn_gru_params(dataset="seeg_he2023xuanwu")

