#!/usr/bin/env python3
"""
Created on 16:49, Aug. 1st, 2024

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
    "tstcc_params",
    "tstcc_cnotra_params",
    "tstcc_cls_params",
]

# def tstcc_params class
class tstcc_params(DotDict):
    """
    This contains single object that generates a dictionary of parameters,
    which is provided to `tstcc` on initialization.
    """
    # Initialize macro parameters.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `tstcc_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(tstcc_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = tstcc_params._gen_model_params(dataset)
        # -- Train parameters
        self.train = tstcc_params._gen_train_params(dataset)

        ## Do init iteration.
        tstcc_params.iteration(self, 0)

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
        model_params.precision = getattr(torch, tstcc_params._precision)\
            if hasattr(torch, tstcc_params._precision) else torch.float32
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of channels.
            model_params.n_channels = 10
            # The length of element sequence.
            model_params.seq_len = 1200
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
        model_params.tokenizer = tstcc_params._gen_model_tokenizer_params(model_params)
        ## -- Additional parameters
        # The scale factor of cls loss.
        model_params.cls_loss_scale = 0.

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
            model_tokenizer_params.n_filters = [32, 64, 128]
            # The size of convolution kernel.
            model_tokenizer_params.kernel_sizes = [25, 8, 8]
            # The number of convolution strides.
            model_tokenizer_params.n_strides = [1, 1, 1]
            # The size of pooling kernel.
            model_tokenizer_params.pool_sizes = [10, 2, 2]
            # The number of pooling strides.
            model_tokenizer_params.pool_strides = [10, 2, 2]
            # The ratio of dropout.
            model_tokenizer_params.dropout = [0.35, 0., 0.]
        # Normal parameters related to other dataset.
        else:
            # The number of convolution filters.
            model_tokenizer_params.n_filters = [32, 64, 128]
            # The size of convolution kernel.
            model_tokenizer_params.kernel_sizes = [9, 9, 9]
            # The number of convolution strides.
            model_tokenizer_params.n_strides = [1, 1, 1]
            # The size of pooling kernel.
            model_tokenizer_params.pool_sizes = [3, 2, 2]
            # The number of pooling strides.
            model_tokenizer_params.pool_strides = [3, 2, 2]
            # The ratio of dropout.
            model_tokenizer_params.dropout = [0.35, 0., 0.]
        # The dimensions of model embedding.
        model_tokenizer_params.d_model = model_tokenizer_params.n_filters[-1]
        # The length of token sequence.
        model_tokenizer_params.token_len = (
            model_params.seq_len // (np.prod(model_tokenizer_params.n_strides) * np.prod(model_tokenizer_params.pool_strides))
        )

        # Return the final `model_tokenizer_params`.
        return model_tokenizer_params

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
        # The specified subject.
        train_params.subj = "028"
        # local rank.
        train_params.local_rank = 0
        # Precision parameter.
        train_params.precision = getattr(torch, tstcc_params._precision)\
            if hasattr(torch, tstcc_params._precision) else torch.float32
        # Whether use graph mode or eager mode.
        train_params.use_graph_mode = False
        # The ratio of train dataset. The rest is test dataset.
        train_params.train_ratio = 0.8
        # Size of buffer used in shuffle.
        train_params.buffer_size = int(1e4)
        # The number of samples used to plot reconstruction.
        train_params.n_samples = 5
        # The iteration of epochs to save model.
        train_params.i_save = 5
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if train_params.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            train_params.batch_size = 64
            # The learning rate factors of training process.
            train_params.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            train_params.batch_size = 16
            # The learning rate factors of training process.
            train_params.lr_factors = (1e-5, 3e-4)

        # Return the final `train_params`.
        return train_params

# def tstcc_contra_params class
class tstcc_contra_params(tstcc_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `tstcc_contra` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `tstcc_contra_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(tstcc_contra_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        tstcc_contra_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(tstcc_contra_params, self).iteration(iteration)
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

    ## def _update_model_* funcs
    # def _update_model_params func
    def _update_model_params(self):
        """
        Update model parameters.
        """
        ## -- Normal parameters
        # The scale factor of cls loss.
        self.model.cls_loss_scale = 0.
        # The scale factor of contra loss.
        self.model.contra_loss_scale = 1.
        ## -- Classification parameters
        self._update_model_contra_params()

    # def _update_model_contra_params func
    def _update_model_contra_params(self):
        """
        Update model.contra parameters.
        """
        # Initialize `model_contra_params`.
        self.model.contra = DotDict()
        ## -- Normal parameters
        self.model.contra.d_token = self.model.tokenizer.d_model
        self.model.contra.d_model = 64
        self.model.contra.n_steps = 20

    ## def _update_train_* funcs
    # def _update_train_params func
    def _update_train_params(self):
        """
        Update train parameters.
        """
        ## -- Normal parameters
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.train.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            self.train.n_epochs = 200
            # Number of warmup epochs.
            self.train.warmup_epochs = 20
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)

# def tstcc_cls_params class
class tstcc_cls_params(tstcc_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `tstcc_cls` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `tstcc_cls_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(tstcc_cls_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        tstcc_cls_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(tstcc_cls_params, self).iteration(iteration)
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

    ## def _update_model_* funcs
    # def _update_model_params func
    def _update_model_params(self):
        """
        Update model parameters.
        """
        ## -- Normal parameters
        # The scale factor of cls loss.
        self.model.cls_loss_scale = 1.
        ## -- Classification parameters
        self._update_model_cls_params()

    # def _update_model_cls_params func
    def _update_model_cls_params(self):
        """
        Update model.cls parameters.
        """
        # Initialize `model_cls_params`.
        self.model.cls = DotDict()
        ## -- Normal parameters
        # The dimensions of feature embedding.
        self.model.cls.d_feature = self.model.tokenizer.token_len * self.model.tokenizer.d_model
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = []
            # The dropout probability after the hidden layers.
            self.model.cls.dropout = 0.
        # Normal parameters related to other dataset.
        else:
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = [128,]
            # The dropout probability after the hidden layers.
            self.model.cls.dropout = 0.
        # The dimensions of classification layer.
        self.model.cls.n_labels = self.model.n_labels

    ## def _update_train_* funcs
    # def _update_train_params func
    def _update_train_params(self):
        """
        Update train parameters.
        """
        ## -- Normal parameters
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.train.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            self.train.n_epochs = 200
            # Number of warmup epochs.
            self.train.warmup_epochs = 20
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)

if __name__ == "__main__":
    # Instantiate `tstcc_params`.
    tstcc_params_inst = tstcc_params(dataset="seeg_he2023xuanwu")
    # Instantiate `tstcc_contra_params`.
    tstcc_contra_params_inst = tstcc_contra_params(dataset="seeg_he2023xuanwu")
    # Instantiate `tstcc_cls_params`.
    tstcc_cls_params_inst = tstcc_cls_params(dataset="seeg_he2023xuanwu")

