#!/usr/bin/env python3
"""
Created on 21:47, Aug. 2rd, 2024

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
    "neurobert_params",
    "neurobert_cls_params",
]

# def neurobert_params class
class neurobert_params(DotDict):
    """
    This contains single object that generates a dictionary of parameters,
    which is provided to `neurobert` on initialization.
    """
    # Initialize macro parameters.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `neurobert_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(neurobert_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = neurobert_params._gen_model_params(dataset)
        # -- Train parameters
        self.train = neurobert_params._gen_train_params(dataset)

        ## Do init iteration.
        neurobert_params.iteration(self, 0)

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
        model_params.precision = getattr(torch, neurobert_params._precision)\
            if hasattr(torch, neurobert_params._precision) else torch.float32
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of input channels.
            model_params.n_channels = 10
            # The length of element sequence.
            model_params.seq_len = 1600
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
        model_params.tokenizer = neurobert_params._gen_model_tokenizer_params(model_params)
        ## -- Encoder parameters
        model_params.encoder = neurobert_params._gen_model_encoder_params(model_params)
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
        # The number of channels.
        model_tokenizer_params.n_channels = model_params.n_channels
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of filters of each convolution block.
            model_tokenizer_params.n_filters = [128,]
            # The size of kernel of each convolution block.
            model_tokenizer_params.kernel_sizes = [40,]
            # The number of strides of each convolution block.
            model_tokenizer_params.n_strides = [40,]
        # Normal parameters related to other dataset.
        else:
            # The number of convolution filters.
            model_tokenizer_params.n_filters = [128,]
            # The size of convolution kernels.
            model_tokenizer_params.kernel_sizes = [3,]
            # The number of convolution strides.
            model_tokenizer_params.n_strides = [1,]
        # The dimensions of the embedding.
        model_tokenizer_params.d_model = model_tokenizer_params.n_filters[-1]
        # The length of token sequence.
        assert model_params.seq_len % np.prod(model_tokenizer_params.n_strides) == 0
        model_tokenizer_params.token_len = model_params.seq_len // np.prod(model_tokenizer_params.n_strides)
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
        # The specified subject.
        train_params.subj = "028"
        # Precision parameter.
        train_params.precision = getattr(torch, neurobert_params._precision)\
            if hasattr(torch, neurobert_params._precision) else torch.float32
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

# def neurobert_mae_params class
class neurobert_mae_params(neurobert_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `neurobert_mae` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `neurobert_mae_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(neurobert_mae_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        neurobert_mae_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(neurobert_mae_params, self).iteration(iteration)
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
        # The scale factor of rgs loss.
        self.model.rgs_loss_scale = 1.
        ## -- Decoder parameters
        self._update_model_decoder_params()
        ## -- Regression parameters
        self._update_model_rgs_params()
        ## -- Additional parameters
        # The mask ratio of random mask.
        self.model.mask_ratio = 0.5

    # def _update_model_decoder_params func
    def _update_model_decoder_params(self):
        """
        Generate model.decoder parameters.
        """
        # Initialize `model_decoder_params`.
        self.model.decoder = DotDict()
        ## -- Normal parameters
        # The dimensions of the embedding.
        self.model.decoder.d_model = self.model.encoder.d_model
        # The length of embedding sequence.
        self.model.decoder.emb_len = self.model.encoder.emb_len
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The number of attention blocks.
            self.model.decoder.n_blocks = 4
            # The flag that indicates whether enable residual attention.
            self.model.decoder.res_attn = False
            # The number of attention heads.
            self.model.decoder.n_heads = 8
            # The dimensions of attention head.
            self.model.decoder.d_head = 64
            # The power base of rotation angle.
            self.model.decoder.rot_theta = None
            # The dropout probability of attention score.
            self.model.decoder.attn_dropout = 0.2
            # The dropout probability of attention projection.
            self.model.decoder.proj_dropout = 0.
            # The dimensions of the hidden layer in ffn.
            self.model.decoder.d_ff = self.model.decoder.d_model * 4
            # The dropout probability of the hidden layer in ffn.
            self.model.decoder.ff_dropout = [0.2, 0.]
            # The flag that indicates whether execute normalization first.
            self.model.decoder.norm_first = False
        # Normal parameters related to other dataset.
        else:
            # The number of attention blocks.
            self.model.decoder.n_blocks = 2
            # The flag that indicates whether enable residual attention.
            self.model.decoder.res_attn = False
            # The number of attention heads.
            self.model.decoder.n_heads = 8
            # The dimensions of attention head.
            self.model.decoder.d_head = 64
            # The power base of rotation angle.
            self.model.decoder.rot_theta = None
            # The dropout probability of attention score.
            self.model.decoder.attn_dropout = 0.
            # The dropout probability of attention projection.
            self.model.decoder.proj_dropout = 0.
            # The dimensions of the hidden layer in ffn.
            self.model.decoder.d_ff = self.model.decoder.d_model * 4
            # The dropout probability of the hidden layer in ffn.
            self.model.decoder.ff_dropout = [0., 0.3]
            # The flag that indicates whether execute normalization first.
            self.model.decoder.norm_first = False

    # def _update_model_rgs_params func
    def _update_model_rgs_params(self):
        """
        Update model.rgs parameters.
        """
        # Initialize `model_rgs_params`.
        self.model.rgs = DotDict()
        ## -- Normal parameters
        # The length of embedding sequence.
        self.model.rgs.emb_len = self.model.decoder.emb_len
        # The dimensions of model embedding.
        self.model.rgs.d_model = self.model.decoder.d_model
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The number of filters of each deconvolution block.
            self.model.rgs.n_filters = [128, 128, 128]
            # The size of kernel of each deconvolution block.
            self.model.rgs.kernel_sizes = [3, 3, 19]
            # The number of strides of each deconvolution block.
            self.model.rgs.n_strides = [2, 2, 10]
            # The dimensions of the hidden layers after deconvolution.
            self.model.rgs.d_hidden = []
        # Normal parameters related to other dataset.
        else:
            # The number of filters of each deconvolution block.
            self.model.rgs.n_filters = [128, 128]
            # The size of kernel of each deconvolution block.
            self.model.rgs.kernel_sizes = [3, 3]
            # The number of strides of each deconvolution block.
            self.model.rgs.n_strides = [1, 1]
            # The dimensions of the hidden layers after deconvolution.
            self.model.rgs.d_hidden = [128,]
        # The number of channels.
        self.model.rgs.n_channels = self.model.tokenizer.n_channels

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
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-5, 3e-4)

# def neurobert_cls_params class
class neurobert_cls_params(neurobert_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `neurobert_cls` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `neurobert_cls_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(neurobert_cls_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        neurobert_cls_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(neurobert_cls_params, self).iteration(iteration)
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
        # The scale factor of rgs loss.
        self.model.rgs_loss_scale = 0.
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
        self.model.cls.d_feature = (
            self.model.encoder.emb_len * self.model.encoder.d_model
        )
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = [128,]
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
    # Instantiate `neurobert_params`.
    neurobert_params_inst = neurobert_params(dataset="seeg_he2023xuanwu")
    # Instantiate `neurobert_mae_params`.
    neurobert_mae_params_inst = neurobert_mae_params(dataset="seeg_he2023xuanwu")
    # Instantiate `neurobert_cls_params`.
    neurobert_cls_params_inst = neurobert_cls_params(dataset="seeg_he2023xuanwu")

