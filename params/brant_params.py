#!/usr/bin/env python3
"""
Created on 20:21, Dec. 2nd, 2023

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
    "brant_params",
    "brant_mae_params",
    "brant_cls_params",
]

# def brant_params class
class brant_params(DotDict):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `brant` on initialization.
    """
    # Initialize macro parameter.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu", model_type=None):
        """
        Initialize `brant_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(brant_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = brant_params._gen_model_params(dataset, model_type=model_type)
        # -- Train parameters
        self.train = brant_params._gen_train_params(dataset)

        ## Do init iteration.
        brant_params.iteration(self, 0)

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
    def _gen_model_params(dataset, model_type=None):
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
        model_params.precision = getattr(torch, brant_params._precision)\
            if hasattr(torch, brant_params._precision) else torch.float32
        # The type of encoder model. `tiny` refers to `Brant-tiny`; `small` refers to `Brant-small`;
        # `medium` refers to `Brant-medium`; `large` refers to `Brant`.
        model_params.model_type = ["tiny", "small", "medium", "large"][0] if model_type is None else model_type
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of input channels.
            model_params.n_channels = 62
            # The number of frequency bands.
            model_params.n_bands = 8
            # The length of element sequence.
            model_params.seq_len = 800 if model_type != "large" else 6000
            # The length of element segment.
            model_params.seg_len = 200 if model_type != "large" else 1500
        # Normal parameters related to other dataset.
        else:
            # The number of input channels.
            model_params.n_channels = 32
            # The number of frequency bands.
            model_params.n_bands = 8
            # The length of element sequence.
            model_params.seq_len = 100
            # The length of element segment.
            model_params.seg_len = 10
        # The number of time segments.
        assert model_params.seq_len % model_params.seg_len == 0
        model_params.n_segs = model_params.seq_len // model_params.seg_len
        # Normal parameters related to tiny model.
        if model_params.model_type == "tiny":
            # The dimension of model.
            model_params.d_model = 768
        # Normal parameters related to small model.
        elif model_params.model_type == "small":
            # The dimension of model.
            model_params.d_model = 1024
        # Normal parameters related to medium model.
        elif model_params.model_type == "medium":
            # The dimension of model.
            model_params.d_model = 1280
        # Normal parameters related to large model.
        elif model_params.model_type == "large":
            # The dimension of model.
            model_params.d_model = 2048
        # Get unknown model type, raise error.
        else:
            raise ValueError("ERROR: Get unknown model type {} in params.brant_params.".format(model_params.model_type))
        ## -- Tokenizer parameters
        model_params.tokenizer = brant_params._gen_model_tokenizer_params(model_params)
        ## -- Encoder parameters
        model_params.encoder = brant_params._gen_model_encoder_params(model_params)
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
        # The maximum number of time segments.
        model_tokenizer_params.max_segs = 15
        # The number of frequency bands.
        model_tokenizer_params.n_bands = model_params.n_bands
        # The length of element segment.
        model_tokenizer_params.seg_len = model_params.seg_len
        # The dimensions of model embedding.
        model_tokenizer_params.d_model = model_params.d_model

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

        ## -- Normal parmaeters
        # The dimensions of model embedding.
        model_encoder_params.d_model = model_params.d_model
        # The number of channels along spatial axis.
        model_encoder_params.n_channels = model_params.n_channels
        # The number of segments along temporal axis.
        model_encoder_params.n_segs = model_params.n_segs
        ## -- Time Encoder parameters
        model_encoder_params.time = brant_params._gen_model_encoder_time_params(model_params)
        ## -- Spatial Encoder parameters
        model_encoder_params.spatial = brant_params._gen_model_encoder_spatial_params(model_params)

        # Return the final `model_encoder_params`.
        return model_encoder_params

    # def _gen_model_encoder_time_params func
    @staticmethod
    def _gen_model_encoder_time_params(model_params):
        """
        Generate model.encoder.time parameters.
        """
        # Initialize `model_encoder_time_params`.
        model_encoder_time_params = DotDict()

        ## -- Normal parameters
        # The number of attention heads.
        model_encoder_time_params.n_heads = 16
        # The dimensions of model embedding.
        model_encoder_time_params.d_model = model_params.d_model
        # Normal parameters related to tiny model.
        if model_params.model_type == "tiny":
            # The number of attention blocks.
            model_encoder_time_params.n_blocks = 8
            # The dimensions of the hidden layer in ffn.
            model_encoder_time_params.d_ff = 2048
        # Normal parameters related to small model.
        elif model_params.model_type == "small":
            # The number of attention blocks.
            model_encoder_time_params.n_blocks = 8
            # The dimensions of the hidden layer in ffn.
            model_encoder_time_params.d_ff = 2048
        # Normal parameters related to medium model.
        elif model_params.model_type == "medium":
            # The number of attention blocks.
            model_encoder_time_params.n_blocks = 12
            # The dimensions of the hidden layer in ffn.
            model_encoder_time_params.d_ff = 3072
        # Normal parameters related to large model.
        elif model_params.model_type == "large":
            # The number of attention blocks.
            model_encoder_time_params.n_blocks = 12
            # The dimensions of the hidden layer in ffn.
            model_encoder_time_params.d_ff = 3072
        # Get unknown model type, raise error.
        else:
            raise ValueError("ERROR: Get unknown model type {} in params.brant_params.".format(model_params.model_type))

        # Return the final `model_encoder_time_params`.
        return model_encoder_time_params

    # def _gen_model_encoder_spatial_params func
    @staticmethod
    def _gen_model_encoder_spatial_params(model_params):
        """
        Generate model.encoder.spatial parameters.
        """
        # Initialize `model_encoder_spatial_params`.
        model_encoder_spatial_params = DotDict()

        ## -- Normal parameters
        # The number of attention heads.
        model_encoder_spatial_params.n_heads = 16
        # The dimensions of model embedding.
        model_encoder_spatial_params.d_model = model_params.d_model
        # Normal parameters related to tiny model.
        if model_params.model_type == "tiny":
            # The number of attention blocks.
            model_encoder_spatial_params.n_blocks = 4
            # The dimensions of the hidden layer in ffn.
            model_encoder_spatial_params.d_ff = 2048
        # Normal parameters related to small model.
        elif model_params.model_type == "small":
            # The number of attention blocks.
            model_encoder_spatial_params.n_blocks = 4
            # The dimensions of the hidden layer in ffn.
            model_encoder_spatial_params.d_ff = 2048
        # Normal parameters related to medium model.
        elif model_params.model_type == "medium":
            # The number of attention blocks.
            model_encoder_spatial_params.n_blocks = 5
            # The dimensions of the hidden layer in ffn.
            model_encoder_spatial_params.d_ff = 3072
        # Normal parameters related to large model.
        elif model_params.model_type == "large":
            # The number of attention blocks.
            model_encoder_spatial_params.n_blocks = 5
            # The dimensions of the hidden layer in ffn.
            model_encoder_spatial_params.d_ff = 3072
        # Get unknown model type, raise error.
        else:
            raise ValueError("ERROR: Get unknown model type {} in params.brant_params.".format(model_params.model_type))

        # Return the final `model_encoder_spatial_params`.
        return model_encoder_spatial_params

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
        # The name of subject.
        train_params.subj = "011"
        # Precision parameter.
        train_params.precision = getattr(torch, brant_params._precision)\
            if hasattr(torch, brant_params._precision) else torch.float32
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
            train_params.batch_size = 128
            # The learning rate factors of training process.
            train_params.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            train_params.batch_size = 128
            # The learning rate factors of training process.
            train_params.lr_factors = (5e-5, 3e-4)

        # Return the final `train_params`.
        return train_params

# def brant_mae_params class
class brant_mae_params(brant_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `brant_mae` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu", model_type=None):
        """
        Initialize `brant_mae_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(brant_mae_params, self).__init__(dataset=dataset, model_type=model_type)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        brant_mae_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(brant_mae_params, self).iteration(iteration)
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
        # The ratio of random mask.
        self.model.mask_ratio = 0.4
        ## -- Regression parameters
        self._update_model_rgs_params()
        ## -- Additional parameters
        # The scale factor of cls loss.
        self.model.cls_loss_scale = 0.
        # The scale factor of rgs loss.
        self.model.rgs_loss_scale = 1.

    # def _update_model_rgs_params func
    def _update_model_rgs_params(self):
        """
        Update model.rgs parameters.
        """
        # Initialize `model_rgs_params`.
        self.model.rgs = DotDict()
        ## -- Normal parameters
        # The dimensions of model embedding.
        self.model.rgs.d_model = self.model.encoder.d_model
        # The dimensions of the hidden layers.
        self.model.rgs.d_hidden = []
        # The dropout probability of the hidden layer.
        self.model.rgs.dropout = 0.
        # The length of element segment.
        self.model.rgs.seg_len = self.model.seg_len

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
            self.train.batch_size = 128
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 3e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 128
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 3e-4)

# def brant_cls_params class
class brant_cls_params(brant_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `brant_cls` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu", model_type=None):
        """
        Initialize `brant_cls_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(brant_cls_params, self).__init__(dataset=dataset, model_type=model_type)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        brant_cls_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(brant_cls_params, self).iteration(iteration)
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
        ## -- Dataset-specific parameters
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The length of element sequence.
            self.model.seq_len = 600 if self.model.model_type != "large" else 4500
            # The number of labels.
            self.model.n_labels = 61
        # Normal parameters related to other dataset.
        else:
            # The length of element sequence.
            self.model.seq_len = 100
            # The number of labels.
            self.model.n_labels = 10
        # The number of time segments.
        assert self.model.seq_len % self.model.seg_len == 0
        self.model.n_segs = self.model.seq_len // self.model.seg_len
        ## -- Encoder parameters
        self._update_model_encoder_params()
        ## -- Classification parameters
        self._update_model_cls_params()
        ## -- Additional parameters
        # The scale factor of cls loss.
        self.model.cls_loss_scale = 1.
        # The scale factor of rgs loss.
        self.model.rgs_loss_scale = 0.

    # def _update_model_encoder_params func
    def _update_model_encoder_params(self):
        """
        Update model.encoder parameters.
        """
        ##-- Normal parameters
        # The number of segments along temporal axis.
        self.model.encoder.n_segs = self.model.n_segs

    # def _update_model_cls_params func
    def _update_model_cls_params(self):
        """
        Update model.cls parameters.
        """
        # Initialize `model_cls_params`.
        self.model.cls = DotDict()
        ## -- Normal parameters
        # The dimensions of feature embedding.
        self.model.cls.d_feature = (
            self.model.encoder.n_channels * self.model.encoder.n_segs * self.model.encoder.d_model
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
            self.train.n_epochs = 200
            # Number of warmup epochs.
            self.train.warmup_epochs = 20
            # Number of batch size used in training process.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 5e-5)
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
    # Initialize `brant_params`.
    brant_params_inst = brant_params(dataset="seeg_he2023xuanwu")
    # Initialize `brant_mae_params`.
    brant_mae_params_inst = brant_mae_params(dataset="seeg_he2023xuanwu")
    # Initialize `brant_cls_params`.
    brant_cls_params_inst = brant_cls_params(dataset="seeg_he2023xuanwu")

