#!/usr/bin/env python3
"""
Created on 00:34, Feb. 8th, 2024

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
    "labram_params",
    "labram_vqvae_params",
    "labram_mae_params",
    "labram_cls_params",
]

# def labram_params class
class labram_params(DotDict):
    """
    This contains single object that generates a dictionary of parameters,
    which is provided to `labram` on initialization.
    """
    # Initialize macro parameters.
    _precision = "float32"

    def __init__(self, dataset="eeg_zhou2023cibr"):
        """
        Initialize `labram_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(labram_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = labram_params._gen_model_params(dataset)
        # -- Train parameters
        self.train = labram_params._gen_train_params(dataset)

        ## Do init iteration.
        labram_params.iteration(self, 0)

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
        model_params.precision = getattr(torch, labram_params._precision)\
            if hasattr(torch, labram_params._precision) else torch.float32
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if model_params.dataset == "eeg_zhou2023cibr":
            # The number of subjects.
            model_params.n_subjects = 1
            # The number of input channels.
            model_params.n_channels = 55
            # The length of element sequence.
            model_params.seq_len = 800
            # The length of patch segment.
            model_params.seg_len = 200
            # The number of output classes.
            model_params.n_labels = 15
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif model_params.dataset == "seeg_he2023xuanwu":
            # The number of subjects.
            model_params.n_subjects = 1
            # The number of input channels.
            model_params.n_channels = 10
            # The length of element sequence.
            model_params.seq_len = 800
            # The length of patch segment.
            model_params.seg_len = 200
            # The number of output classes.
            model_params.n_labels = 61
        # Normal parameters related to other dataset.
        else:
            # The number of subjects.
            model_params.n_subjects = 1
            # The number of input channels.
            model_params.n_channels = 32
            # The length of element sequence.
            model_params.seq_len = 800
            # The length of patch segment.
            model_params.seg_len = 200
            # The number of output classes.
            model_params.n_labels = 10
        ## -- Tokenizer parameters
        model_params.tokenizer = labram_params._gen_model_tokenizer_params(model_params)
        ## -- Encoder parameters
        model_params.encoder = labram_params._gen_model_encoder_params(model_params)
        ## -- Vector-Quantizer parameters
        model_params.vq = labram_params._gen_model_vq_params(model_params)
        ## -- Additional parameters
        # The scale factor of cls loss.
        model_params.cls_loss_scale = 0.
        # The scale factor of rgs loss.
        model_params.rgs_loss_scale = 0.
        # The scale factor of vq loss.
        model_params.vq_loss_scale = 0.

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
        # The dimensions of model embedding.
        model_tokenizer_params.d_model = model_params.seg_len
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if model_params.dataset == "eeg_zhou2023cibr":
            # The number of filters of each convolution block.
            model_tokenizer_params.n_filters = [8, 8, 8]
            # The size of kernel of each deconvolution block.
            model_tokenizer_params.kernel_sizes = [15, 3, 3]
            # The number of strides of each deconvolution block.
            model_tokenizer_params.n_strides = [8, 1, 1]
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif model_params.dataset == "seeg_he2023xuanwu":
            # The number of filters of each convolution block.
            model_tokenizer_params.n_filters = [8, 8, 8]
            # The size of kernel of each deconvolution block.
            model_tokenizer_params.kernel_sizes = [15, 3, 3]
            # The number of strides of each deconvolution block.
            model_tokenizer_params.n_strides = [8, 1, 1]
        # Normal parameters related to other dataset.
        else:
            # The number of filters of each convolution block.
            model_tokenizer_params.n_filters = [8, 8, 8]
            # The size of kernel of each deconvolution block.
            model_tokenizer_params.kernel_sizes = [15, 3, 3]
            # The number of strides of each deconvolution block.
            model_tokenizer_params.n_strides = [8, 1, 1]
        # Make sure that the output of `*_blocks` have the same dimensions!
        if len(model_tokenizer_params.n_filters) > 0:
            assert np.prod(model_tokenizer_params.n_strides) == model_tokenizer_params.n_filters[-1]

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
        # The number of input channels.
        model_encoder_params.n_channels = model_params.n_channels
        # The maximum number of segments.
        model_encoder_params.max_segs = model_params.seq_len // model_params.seg_len
        # The length of embedding sequence.
        model_encoder_params.emb_len = model_encoder_params.n_channels * model_encoder_params.max_segs
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if model_params.dataset == "eeg_zhou2023cibr":
            # The number of attention blocks.
            model_encoder_params.n_blocks = 8
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
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif model_params.dataset == "seeg_he2023xuanwu":
            # The number of attention blocks.
            model_encoder_params.n_blocks = 8
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

    # def _gen_model_vq_params func
    @staticmethod
    def _gen_model_vq_params(model_params):
        """
        Generate model.vq parameters.
        """
        # Initialize `model_vq_params`.
        model_vq_params = DotDict()

        ## -- Normal parameters
        # The dimensions of model embedding.
        model_vq_params.d_model = model_params.encoder.d_model
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if model_params.dataset == "eeg_zhou2023cibr":
            # The number of discrete embeddings per group.
            model_vq_params.codex_size = 8192
            # The dimensions of codex embedding.
            model_vq_params.d_codex = 64
            # The scale factor of commitment loss (which is a part of vq loss).
            model_vq_params.beta = 1.
            # The decay factor used to update embeddings according to `decay * weight_old + (1 - decay) * weight_new`.
            model_vq_params.decay = 0.99
            # The flag that indicates whether use kmeans to initialize weight.
            model_vq_params.init_kmeans = True
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif model_params.dataset == "seeg_he2023xuanwu":
            # The number of discrete embeddings per group.
            model_vq_params.codex_size = 8192
            # The dimensions of codex embedding.
            model_vq_params.d_codex = 64
            # The scale factor of commitment loss (which is a part of vq loss).
            model_vq_params.beta = 1.
            # The decay factor used to update embeddings according to `decay * weight_old + (1 - decay) * weight_new`.
            model_vq_params.decay = 0.99
            # The flag that indicates whether use kmeans to initialize weight.
            model_vq_params.init_kmeans = True
        # Normal parameters related to other dataset.
        else:
            # The number of discrete embeddings per group.
            model_vq_params.codex_size = 8192
            # The dimensions of codex embedding.
            model_vq_params.d_codex = 32
            # The scale factor of commitment loss (which is a part of vq loss).
            model_vq_params.beta = 1.
            # The decay factor used to update embeddings according to `decay * weight_old + (1 - decay) * weight_new`.
            model_vq_params.decay = 0.99
            # The flag that indicates whether use kmeans to initialize weight.
            model_vq_params.init_kmeans = True

        # Return the final `model_vq_params`.
        return model_vq_params

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
        # Precision parameter.
        train_params.precision = getattr(torch, labram_params._precision)\
            if hasattr(torch, labram_params._precision) else torch.float32
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
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if train_params.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (1e-5, 5e-5)
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif train_params.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (1e-5, 5e-5)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            train_params.batch_size = 32
            # The learning rate factors of training process.
            train_params.lr_factors = (1e-5, 5e-5)

        # Return the final `train_params`.
        return train_params

# def labram_vqvae_params class
class labram_vqvae_params(labram_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `labram_vqvae` on initialization.
    """

    def __init__(self, dataset="eeg_zhou2023cibr"):
        """
        Initialize `labram_vqvae_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(labram_vqvae_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        labram_vqvae_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(labram_vqvae_params, self).iteration(iteration)
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
        # The scale factor of vq loss.
        self.model.vq_loss_scale = 1.
        ## -- Decoder parameters
        self._update_model_decoder_params()
        ## -- Regression parameters
        self._update_model_rgs_params()

    # def _update_model_decoder_params func
    def _update_model_decoder_params(self):
        """
        Update model.decoder parameters.
        """
        # Initialize `model_decoder_params`.
        self.model.decoder = DotDict()
        ## -- Normal parameters
        # The dimensions of the embedding.
        self.model.decoder.d_model = self.model.encoder.d_model
        # The length of embedding sequence.
        self.model.decoder.emb_len = self.model.encoder.emb_len
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if self.model.dataset == "eeg_zhou2023cibr":
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
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif self.model.dataset == "seeg_he2023xuanwu":
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
        # The dimensions of model embedding.
        self.model.rgs.d_model = self.model.decoder.d_model
        # The dimensions of hidden layers.
        self.model.rgs.d_hidden = [self.model.rgs.d_model,]
        # The size of fourier transform.
        self.model.rgs.n_fft = self.model.seg_len

    ## def _update_train_* funcs
    # def _update_train_params func
    def _update_train_params(self):
        """
        Update train parameters.
        """
        ## -- Normal parameters
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if self.train.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 5e-5)
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif self.train.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 5e-5)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 5e-5)

# def labram_mae_params class
class labram_mae_params(labram_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `labram_mae` on initialization.
    """

    def __init__(self, dataset="eeg_zhou2023cibr"):
        """
        Initialize `labram_mae_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(labram_mae_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        labram_mae_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(labram_mae_params, self).iteration(iteration)
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
        # The scale factor of vq loss.
        self.model.vq_loss_scale = 0.
        ## -- Classification parameters
        self._update_model_cls_params()
        ## -- Additional parameters
        # The mask ratio of random mask.
        self.model.mask_ratio = 0.5

    # def _update_model_cls_params func
    def _update_model_cls_params(self):
        """
        Update model.cls parameters.
        """
        # Initialize `model_cls_params`.
        self.model.cls = DotDict()
        ## -- Normal parameters
        # The dimensions of model embedding.
        self.model.cls.d_model = self.model.encoder.d_model
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if self.model.dataset == "eeg_zhou2023cibr":
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = []
            # The dropout probability after the hidden layers.
            self.model.cls.dropout = 0.
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif self.model.dataset == "seeg_he2023xuanwu":
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
        self.model.cls.n_tokens = self.model.vq.codex_size

    ## def _update_train_* funcs
    # def _update_train_params func
    def _update_train_params(self):
        """
        Update train parameters.
        """
        ## -- Normal parameters
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if self.train.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 5e-4)
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif self.train.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 5e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 5e-4)

# def labram_cls_params class
class labram_cls_params(labram_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `labram_cls` on initialization.
    """

    def __init__(self, dataset="eeg_zhou2023cibr"):
        """
        Initialize `labram_cls_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(labram_cls_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        labram_cls_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(labram_cls_params, self).iteration(iteration)
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
        # The scale factor of vq loss.
        self.model.vq_loss_scale = 0.
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
        self.model.cls.d_feature = self.model.encoder.emb_len * self.model.encoder.d_model
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if self.model.dataset == "eeg_zhou2023cibr":
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = [128,]
            # The dropout probability after the hidden layers.
            self.model.cls.dropout = 0.
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif self.model.dataset == "seeg_he2023xuanwu":
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
        # Normal parameters related to eeg_zhou2023cibr dataset.
        if self.train.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 128
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-6, 5e-4)
        # Normal parameters related to seeg_he2023xuanwu dataset.
        elif self.train.dataset == "seeg_he2023xuanwu":
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-6, 5e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            self.train.batch_size = 128
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-6, 5e-4)

if __name__ == "__main__":
    # Instantiate `labram_params`.
    labram_params_inst = labram_params(dataset="eeg_zhou2023cibr")
    # Instantiate `labram_vqvae_params`.
    labram_vqvae_params_inst = labram_vqvae_params(dataset="eeg_zhou2023cibr")
    # Instantiate `labram_mae_params`.
    labram_mae_params_inst = labram_mae_params(dataset="eeg_zhou2023cibr")
    # Instantiate `labram_cls_params`.
    labram_cls_params_inst = labram_cls_params(dataset="eeg_zhou2023cibr")

