#!/usr/bin/env python3
"""
Created on 20:22, Nov. 30th, 2023

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
    "brainbert_params",
    "brainbert_mae_params",
    "brainbert_cls_params",
]

# def brainbert_params class
class brainbert_params(DotDict):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `brainbert` on initialization.
    """
    # Initialize macro parameter.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `brainbert_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(brainbert_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = brainbert_params._gen_model_params(dataset)
        # -- Train parameters
        self.train = brainbert_params._gen_train_params(dataset)

        ## Do init iteration.
        brainbert_params.iteration(self, 0)

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
        model_params.precision = getattr(torch, brainbert_params._precision)\
            if hasattr(torch, brainbert_params._precision) else torch.float32
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The length of spectrum sequence.
            model_params.freq_len = 160
        # Normal parameters related to other dataset.
        else:
            # The length of spectrum sequence.
            model_params.freq_len = 100
        # The type of encoder model. `small` refers to `BrainBERT-small`;
        # `base` refers to `BrainBERT-base`; `large` refers to `BrainBERT-large`.
        model_params.model_type = ["small", "base", "large"][-1]
        # The number of frequencies.
        model_params.n_freqs = 40
        # Normal parameters related to small model.
        if model_params.model_type == "small":
            # The dimension of model.
            model_params.d_model = 384
        # Normal parameters related to base model.
        elif model_params.model_type == "base":
            # The dimension of model.
            model_params.d_model = 768
        # Normal parameters related to large model.
        elif model_params.model_type == "large":
            # The dimension of model.
            model_params.d_model = 768
        # Get unknown model type, raise error.
        else:
            raise ValueError("ERROR: Get unknown model type {} in params.brainbert_params.".format(model_params.model_type))
        ## -- Tokenizer parameters
        model_params.tokenizer = brainbert_params._gen_model_tokenizer_params(model_params)
        ## -- Encoder parameters
        model_params.encoder = brainbert_params._gen_model_encoder_params(model_params)
        ## -- Additional parameters
        # The scale factor of cls loss.
        model_params.cls_loss_scale = 0.
        # The scale factor of rgs loss.
        model_params.rgs_loss_scale = 0.

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
        # The number of frequencies.
        model_tokenizer_params.n_freqs = model_params.n_freqs
        # The dimensions of model embedding.
        model_tokenizer_params.d_model = model_params.d_model
        # The maximum length of embedding sequence.
        model_tokenizer_params.max_len = 5000
        # The ratio of dropout after projection.
        model_tokenizer_params.dropout = 0.1
        # The length of token sequence.
        model_tokenizer_params.token_len = model_params.freq_len

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
        # The length of embedding sequence.
        model_encoder_params.emb_len = model_params.tokenizer.token_len
        # The dimensions of model embedding.
        model_encoder_params.d_model = model_params.tokenizer.d_model
        # The number of attention heads.
        model_encoder_params.n_heads = 12
        # The type of activation.
        model_encoder_params.activation = ["relu", "gelu"][-1]
        # Normal parameters related to small model.
        if model_params.model_type == "small":
            # The dimensions of the hidden layer in ffn.
            model_encoder_params.d_ff = 1200
            # The number of encoder blocks.
            model_encoder_params.n_blocks = 3
        # Normal parameters related to base model.
        elif model_params.model_type == "base":
            # The dimensions of the hidden layer in ffn.
            model_encoder_params.d_ff = 3072
            # The number of encoder blocks.
            model_encoder_params.n_blocks = 3
        # Normal parameters related to large model.
        elif model_params.model_type == "large":
            # The dimensions of the hidden layer in ffn.
            model_encoder_params.d_ff = 3072
            # The number of encoder blocks.
            model_encoder_params.n_blocks = 6
        # Get unknown model type, raise error.
        else:
            raise ValueError("ERROR: Get unknown model type {} in params.brainbert_params.".format(model_params.model_type))

        # Return the final `model_encoder_params`.
        return model_encoder_params

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
        # The rank of distributed device.
        train_params.local_rank = 0
        # The list of subjects.
        train_params.subjs = ["023",]
        # Precision parameter.
        train_params.precision = getattr(torch, brainbert_params._precision)\
            if hasattr(torch, brainbert_params._precision) else torch.float32
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
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (5e-5, 3e-4)

        # Return the final `train_params`.
        return train_params

# def brainbert_mae_params class
class brainbert_mae_params(brainbert_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `brainbert_mae` on initialization.
    """
    # Internal macro parameter.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `brainbert_mae_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(brainbert_mae_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        brainbert_mae_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(brainbert_mae_params, self).iteration(iteration)
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
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The number of channels.
            self.model.n_channels = 10
        # Normal parameters related to other dataset.
        else:
            # The number of channels.
            self.model.n_channels = 8
        ## -- Regression parameters
        self._update_model_rgs_parmas()
        ## -- Additional parameters
        # The scale factor of cls loss.
        self.model.cls_loss_scale = 0.
        # The scale factor of rgs loss.
        self.model.rgs_loss_scale = 1.

    # def _update_model_rgs_parmas func
    def _update_model_rgs_parmas(self):
        """
        Update model.rgs parameters.
        """
        # Initialize `model_rgs_params`.
        self.model.rgs = DotDict()
        ## -- Normal parameters
        # The dimensions of model embedding.
        self.model.rgs.d_model = self.model.encoder.d_model
        # The number of frequencies.
        self.model.rgs.n_freqs = self.model.n_freqs

    ## def _update_train_* funcs
    # def _update_train_params func
    def _update_train_params(self):
        """
        Update train parameters.
        """
        ## -- Normal parameters
        # Normal parmeters related to seeg_he2023xuanwu dataset.
        if self.train.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-5, 3e-4)

# def brainbert_cls_params class
class brainbert_cls_params(brainbert_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `brainbert_cls` on initialization.
    """
    # Internal macro parameter.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `brainbert_cls_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(brainbert_cls_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        brainbert_cls_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(brainbert_cls_params, self).iteration(iteration)
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
        # The scale factor of gradient flow.
        self.model.grad_scale = 0.1
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The number of channels.
            self.model.n_channels = 10
            # The number of labels.
            self.model.n_labels = 61
        # Normal parameters related to other dataset.
        else:
            # The number of channels.
            self.model.n_channels = 8
            # The number of labels.
            self.model.n_labels = 10
        ## -- Classification parameters
        self._update_model_cls_parmas()
        ## -- Additional parameters
        # The scale factor of cls loss.
        self.model.cls_loss_scale = 1.
        # The scale factor of rgs loss.
        self.model.rgs_loss_scale = 0.

    # def _update_model_cls_parmas func
    def _update_model_cls_parmas(self):
        """
        Update model.cls parameters.
        """
        # Initialize `model_cls_params`.
        self.model.cls = DotDict()
        ## -- Normal parameters
        # The range to average embeddings from the center.
        self.model.cls.avg_range = [-5, 5]
        # The dimensions of feature embedding.
        self.model.cls.d_feature = (
            self.model.n_channels * self.model.encoder.d_model
        )
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = [128,]
            # The dropout rate of dropout layer.
            self.model.cls.dropout = 0.
        # Normal parameters related to other dataset.
        else:
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = [128,]
            # The dropout rate of dropout layer.
            self.model.cls.dropout = 0.
        # The number of labels.
        self.model.cls.n_labels = self.model.n_labels

    ## def _update_train_* funcs
    # def _update_train_params func
    def _update_train_params(self):
        """
        Update train parameters.
        """
        ## -- Normal parameters
        # Normal parmeters related to seeg_he2023xuanwu dataset.
        if self.train.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)

if __name__ == "__main__":
    # Initialize `brainbert_params`.
    brainbert_params_inst = brainbert_params(dataset="seeg_he2023xuanwu")
    # Initialize `brainbert_mae_params`.
    brainbert_mae_params_inst = brainbert_mae_params(dataset="seeg_he2023xuanwu")
    # Initialize `brainbert_cls_params`.
    brainbert_cls_params_inst = brainbert_cls_params(dataset="seeg_he2023xuanwu")

