#!/usr/bin/env python3
"""
Created on 17:08, Jan. 16th, 2024

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
    "duin_params",
    "duin_vqvae_params",
    "duin_mae_params",
    "duin_cls_params",
    "duin_llm_params",
]

# def duin_params class
class duin_params(DotDict):
    """
    This contains single object that generates a dictionary of parameters,
    which is provided to `duin` on initialization.
    """
    # Initialize macro parameters.
    _precision = "float32"

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `duin_params`.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(duin_params, self).__init__()

        ## Generate all parameters hierarchically.
        # -- Model parameters
        self.model = duin_params._gen_model_params(dataset)
        # -- Train parameters
        self.train = duin_params._gen_train_params(dataset)

        ## Do init iteration.
        duin_params.iteration(self, 0)

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
        model_params.precision = getattr(torch, duin_params._precision)\
            if hasattr(torch, duin_params._precision) else torch.float32
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of subjects.
            model_params.n_subjects = 10
            # The number of input channels.
            model_params.n_channels = 16
            # The length of element sequence.
            model_params.seq_len = 4000
            # The length of element segment.
            model_params.seg_len = 100
            # The number of output classes.
            model_params.n_labels = 61
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif model_params.dataset == "eeg_zhou2023cibr":
            # The number of subjects.
            model_params.n_subjects = 10
            # The number of input channels.
            model_params.n_channels = 55
            # The length of element sequence.
            model_params.seq_len = 400
            # The length of element segment.
            model_params.seg_len = 10
            # The number of output classes.
            model_params.n_labels = 15
        # Normal parameters related to other dataset.
        else:
            # The number of subjects.
            model_params.n_subjects = 10
            # The number of input channels.
            model_params.n_channels = 32
            # The length of element sequence.
            model_params.seq_len = 100
            # The length of element segment.
            model_params.seg_len = 10
            # The number of output classes.
            model_params.n_labels = 10
        ## -- Subject parameters
        model_params.subj = duin_params._gen_model_subj_params(model_params)
        ## -- Tokenizer parameters
        model_params.tokenizer = duin_params._gen_model_tokenizer_params(model_params)
        ## -- Encoder parameters
        model_params.encoder = duin_params._gen_model_encoder_params(model_params)
        ## -- Vector-Quantizer parameters
        model_params.vq = duin_params._gen_model_vq_params(model_params)
        ## -- Additional parameters
        # The scale factor of cls loss.
        model_params.cls_loss_scale = 0.
        # The scale factor of contrastive loss.
        model_params.contra_loss_scale = 0.
        # The scale factor of rgs loss.
        model_params.rgs_loss_scale = 0.
        # The scale factor of vq loss.
        model_params.vq_loss_scale = 0.

        # Return the final `model_params`.
        return model_params

    # def _gen_model_subj_params func
    @staticmethod
    def _gen_model_subj_params(model_params):
        """
        Generate model.subj parameters.
        """
        # Initialize `model_subj_params`.
        model_subj_params = DotDict()

        ## -- Normal parameters
        # The number of subjects.
        model_subj_params.n_subjects = model_params.n_subjects
        # The dimensions of input embedding.
        model_subj_params.d_input = model_params.n_channels
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The dimensions of output embedding.
            model_subj_params.d_output = 16
            # The flag that indicates whether enable embedding shift.
            model_subj_params.use_bias = True
            # The flag that indicates whether enable projection layer.
            model_subj_params.use_proj = False
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif model_params.dataset == "eeg_zhou2023cibr":
            # The dimensions of output embedding.
            model_subj_params.d_output = 64
            # The flag that indicates whether enable embedding shift.
            model_subj_params.use_bias = True
            # The flag that indicates whether enable projection layer.
            model_subj_params.use_proj = False
        # Normal parameters related to other dataset.
        else:
            # The dimensions of output embedding.
            model_subj_params.d_output = 128
            # The flag that indicates whether enable embedding shift.
            model_subj_params.use_bias = False
            # The flag that indicates whether enable projection layer.
            model_subj_params.use_proj = True

        # Return the final `model_subj_params`.
        return model_subj_params

    # def _gen_model_tokenizer_params func
    @staticmethod
    def _gen_model_tokenizer_params(model_params):
        """
        Generate model.tokenizer parameters.
        """
        # Initialize `model_tokenizer_params`.
        model_tokenizer_params = DotDict()

        ## -- Normal parameters
        # The scale factor of gradient flow.
        model_tokenizer_params.grad_scale = 1.
        # The dimensions of common hidden neural space.
        model_tokenizer_params.d_neural = model_params.subj.d_output
        # The length of element segment.
        model_tokenizer_params.seg_len = model_params.seg_len
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of filters of each convolution block.
            model_tokenizer_params.n_filters = [128, 128, 16]
            # The size of kernel of each convolution block.
            model_tokenizer_params.kernel_sizes = [19, 3, 3]
            # The number of strides of each convolution block.
            model_tokenizer_params.n_strides = [10, 1, 1]
            # The dilation rate of each convolution block.
            model_tokenizer_params.dilation_rates = [1, 1, 1]
            # The flag that indicates whether use batch-norm.
            model_tokenizer_params.use_bn = [True, True, True]
            # The flag that indicates whether use residual connection.
            model_tokenizer_params.use_res = [False, False, False]
            # The size of pooling of each convolution block.
            model_tokenizer_params.pool_sizes = [1, 1, 1]
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif model_params.dataset == "eeg_zhou2023cibr":
            ## -- Normal parameters (related to ConvNDBlock)
            # The number of filters of each ConvNDBlock.
            model_tokenizer_params.n_filters = [256, 128]
            # The size of kernel of each ConvNDBlock.
            model_tokenizer_params.kernel_sizes = [3, 3]
            # The number of strides of each ConvNDBlock.
            model_tokenizer_params.n_strides = [2, 2]
            # The dilation rate of each ConvNDBlock.
            model_tokenizer_params.dilation_rates = [1, 1]
            # The flag that indicates whether use batch-norm.
            model_tokenizer_params.use_bn = [True, True]
            # The flag that indicates whether use residual connection.
            model_tokenizer_params.use_res = [False, False]
            # The size of pooling of each ConvNDBlock.
            model_tokenizer_params.pool_sizes = [1, 1]
        # Normal parameters related to other dataset.
        else:
            # The number of convolution filters.
            model_tokenizer_params.n_filters = [128, 128]
            # The size of convolution kernels.
            model_tokenizer_params.kernel_sizes = [3, 3]
            # The number of convolution strides.
            model_tokenizer_params.n_strides = [1, 1]
            # The dilation rate of each convolution block.
            model_tokenizer_params.dilation_rates = [1, 1]
            # The flag that indicates whether use bias in convolution.
            model_tokenizer_params.use_bias = [True, True]
            # The dropout ratio after convolution.
            model_tokenizer_params.dropout = [0., 0.]
            # The type of normalization after convolution.
            # Note: `norm_type` should be one of [None,layer_norm,group_norm].
            model_tokenizer_params.norm_type = [None, None]
        # The dimensions of the embedding.
        assert model_tokenizer_params.seg_len % np.prod(model_tokenizer_params.n_strides) == 0
        model_tokenizer_params.d_model = (
            model_tokenizer_params.n_filters[-1] * (model_tokenizer_params.seg_len // np.prod(model_tokenizer_params.n_strides))
        )
        # The length of token sequence.
        model_tokenizer_params.token_len = model_params.seq_len // model_tokenizer_params.seg_len
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
            model_encoder_params.d_ff = model_encoder_params.d_model * 2
            # The dropout probability of the hidden layer in ffn.
            model_encoder_params.ff_dropout = [0.2, 0.]
            # The flag that indicates whether execute normalization first.
            model_encoder_params.norm_first = False
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif model_params.dataset == "eeg_zhou2023cibr":
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
            model_encoder_params.d_ff = model_encoder_params.d_model * 2
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
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if model_params.dataset == "seeg_he2023xuanwu":
            # The number of discrete embeddings per group.
            model_vq_params.codex_size = 2048
            # The dimensions of codex embedding.
            model_vq_params.d_codex = 64
            # The scale factor of commitment loss (which is a part of vq loss).
            model_vq_params.beta = 1.
            # The decay factor used to update embeddings according to `decay * weight_old + (1 - decay) * weight_new`.
            model_vq_params.decay = 0.99
            # The flag that indicates whether use kmeans to initialize weight.
            model_vq_params.init_kmeans = True
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif model_params.dataset == "eeg_zhou2023cibr":
            # The number of discrete embeddings per group.
            model_vq_params.codex_size = 2048
            # The dimensions of codex embedding.
            model_vq_params.d_codex = 32
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
        # The list of subjects.
        train_params.subjs = ["023",]
        # Precision parameter.
        train_params.precision = getattr(torch, duin_params._precision)\
            if hasattr(torch, duin_params._precision) else torch.float32
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
            # Note: `64` is best for one subject. All subjects share one batch.
            train_params.batch_size = 64
            # The learning rate factors of training process.
            train_params.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif train_params.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            train_params.n_epochs = 100
            # Number of warmup epochs.
            train_params.warmup_epochs = 10
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
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

# def duin_vqvae_params class
class duin_vqvae_params(duin_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `duin_vqvae` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `duin_vqvae_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(duin_vqvae_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        duin_vqvae_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(duin_vqvae_params, self).iteration(iteration)
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
        # The scale factor of contrastive loss.
        self.model.contra_loss_scale = 0.
        # The scale factor of rgs loss.
        self.model.rgs_loss_scale = 1.
        # The scale factor of vq loss.
        self.model.vq_loss_scale = 1.
        ## -- Decoder parameters
        self._update_model_decoder_params()
        ## -- Regression parameters
        self._update_model_rgs_params()
        ## -- De-Subject parameters
        self._update_model_desubj_params()

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
            self.model.decoder.d_ff = self.model.decoder.d_model * 2
            # The dropout probability of the hidden layer in ffn.
            self.model.decoder.ff_dropout = [0.2, 0.]
            # The flag that indicates whether execute normalization first.
            self.model.decoder.norm_first = False
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.model.dataset == "eeg_zhou2023cibr":
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
            self.model.decoder.d_ff = self.model.decoder.d_model * 2
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
        # The length of element segment.
        self.model.rgs.seg_len = self.model.seg_len
        # The dimensions of model embedding.
        self.model.rgs.d_model = self.model.decoder.d_model
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The number of filters of each deconvolution block.
            self.model.rgs.n_filters = [128, 128, 128, 128, 16]
            # The size of kernel of each deconvolution block.
            self.model.rgs.kernel_sizes = [3, 3, 10, 9, 19]
            # The number of strides of each deconvolution block.
            self.model.rgs.n_strides = [1, 1, 10, 1, 10]
            # The dilation rate of each deconvolution block.
            self.model.rgs.dilation_rates = [1, 1, 1, 1, 1]
            # The dimensions of the hidden layers after deconvolution.
            self.model.rgs.d_hidden = []
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.model.dataset == "eeg_zhou2023cibr":
            ## -- Normal parameters (related to ConvTNDBlock)
            # The number of filters of each ConvTNDBlock.
            self.model.rgs.n_filters = [256, 64]
            # The size of kernel of each ConvTNDBlock.
            self.model.rgs.kernel_sizes = [3, 3]
            # The number of strides of each ConvTNDBlock.
            self.model.rgs.n_strides = [2, 2]
            # The dilation rate of each ConvTNDBlock.
            self.model.rgs.dilation_rates = [1, 1]
            ## -- Normal parameters (related to Dense)
            # The dimensions of the hidden layer in classification block.
            self.model.rgs.d_hidden = []
        # Normal parameters related to other dataset.
        else:
            # The number of filters of each deconvolution block.
            self.model.rgs.n_filters = [128, 128]
            # The size of kernel of each deconvolution block.
            self.model.rgs.kernel_sizes = [3, 3]
            # The number of strides of each deconvolution block.
            self.model.rgs.n_strides = [1, 1]
            # The dilation rate of each deconvolution block.
            self.model.rgs.dilation_rates = [1, 1]
            # The dimensions of the hidden layers after deconvolution.
            self.model.rgs.d_hidden = [128,]
        # The dimensions of common hidden neural space.
        self.model.rgs.d_neural = self.model.tokenizer.d_neural

    # def _update_model_desubj_params func
    def _update_model_desubj_params(self):
        """
        Update model.desubj parameters.
        """
        # Initialize `model_desubj_params`.
        self.model.desubj = DotDict()
        ## -- Normal parameters
        # The number of subjects.
        self.model.desubj.n_subjects = self.model.n_subjects
        # The dimensions of input embedding.
        self.model.desubj.d_input = self.model.rgs.d_neural
        # The dimensions of output embedding.
        self.model.desubj.d_output = self.model.n_channels
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The flag that indicates whether enable embedding shift.
            self.model.desubj.use_bias = True
            # The flag that indicates whether enable projection layer.
            self.model.desubj.use_proj = False
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.model.dataset == "eeg_zhou2023cibr":
            # The flag that indicates whether enable embedding shift.
            self.model.desubj.use_bias = True
            # The flag that indicates whether enable projection layer.
            self.model.desubj.use_proj = False
        # Normal parameters related to other dataset.
        else:
            # The flag that indicates whether enable embedding shift.
            self.model.desubj.use_bias = False
            # The flag that indicates whether enable projection layer.
            self.model.desubj.use_proj = True

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
            self.train.n_epochs = 800
            # Number of warmup epochs.
            self.train.warmup_epochs = 80
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.train.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            self.train.n_epochs = 400
            # Number of warmup epochs.
            self.train.warmup_epochs = 40
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 256
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 16
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 3e-4)

# def duin_mae_params class
class duin_mae_params(duin_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `duin_mae` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `duin_mae_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(duin_mae_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        duin_mae_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(duin_mae_params, self).iteration(iteration)
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
        # The scale factor of contrastive loss.
        self.model.contra_loss_scale = 0.
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
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = []
            # The dropout probability after the hidden layers.
            self.model.cls.dropout = 0.
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.model.dataset == "eeg_zhou2023cibr":
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
        self.model.cls.n_tokens = self.model.vq.codex_size

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
            self.train.n_epochs = 400
            # Number of warmup epochs.
            self.train.warmup_epochs = 40
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.train.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            self.train.n_epochs = 400
            # Number of warmup epochs.
            self.train.warmup_epochs = 40
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 256
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-5, 3e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 16
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 3e-4)

# def duin_cls_params class
class duin_cls_params(duin_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `duin_cls` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `duin_cls_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(duin_cls_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        duin_cls_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(duin_cls_params, self).iteration(iteration)
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
        # The scale factor of contrastive loss.
        self.model.contra_loss_scale = 0.
        # The scale factor of rgs loss.
        self.model.rgs_loss_scale = 0.
        # The scale factor of vq loss.
        self.model.vq_loss_scale = 0.
        ## -- Contrastive parameters
        self._update_model_contra_params()
        ## -- Classification parameters
        self._update_model_cls_params()

    # def _update_model_contra_params func
    def _update_model_contra_params(self):
        """
        Update model.contra parameters.
        """
        # Initialize `model_contra_params`.
        self.model.contra = DotDict()
        ## -- Normal parameters
        # The dimensions of model embedding.
        self.model.contra.d_model = self.model.encoder.d_model
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The dimension of contrastive layer.
            self.model.contra.d_contra = 32
            # The mode of contrastive loss calculation.
            self.model.contra.loss_mode = ["clip", "clip_orig", "unicl"][-1]
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.model.dataset == "eeg_zhou2023cibr":
            # The dimension of contrastive layer.
            self.model.contra.d_contra = 32
            # The mode of contrastive loss calculation.
            self.model.contra.loss_mode = ["clip", "clip_orig", "unicl"][-1]
        # Normal parameters related to other dataset.
        else:
            # The dimension of contrastive layer.
            self.model.contra.d_contra = 32
            # The mode of contrastive loss calculation.
            self.model.contra.loss_mode = ["clip", "clip_orig", "unicl"][-1]

    # def _update_model_cls_params func
    def _update_model_cls_params(self):
        """
        Update model.cls parameters.
        """
        # Initialize `model_cls_params`.
        self.model.cls = DotDict()
        ## -- Normal parameters
        # The dimensions of feature embedding.
        self.model.cls.d_feature = self.model.encoder.d_model * self.model.encoder.emb_len
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = [128,]
            # The dropout probability after the hidden layers.
            self.model.cls.dropout = 0.
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.model.dataset == "eeg_zhou2023cibr":
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
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 32
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.train.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 16
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 3e-4)

# def duin_llm_params class
class duin_llm_params(duin_params):
    """
    This contains one single object that generates a dictionary of parameters,
    which is provided to `duin_llm` on initialization.
    """

    def __init__(self, dataset="seeg_he2023xuanwu"):
        """
        Initialize `duin_llm_params` object.
        """
        ## First call super class init function to set up `DotDict`
        ## style object and inherit it's functionality.
        super(duin_llm_params, self).__init__(dataset=dataset)

        ## Update all parameters hierarchically.
        # -- Model parameters
        self._update_model_params()
        # -- Train parameters
        self._update_train_params()

        ## Do init iteration.
        duin_llm_params.iteration(self, 0)

    """
    update funcs
    """
    # def iteration func
    def iteration(self, iteration):
        """
        Update parameters at every backpropagation iteration/gradient update.
        """
        ## Iterate super parameters.
        super(duin_llm_params, self).iteration(iteration)
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
        # The scale factor of contrastive loss.
        self.model.contra_loss_scale = 0.
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
        # The dimensions of model embedding.
        self.model.cls.d_model = self.model.encoder.d_model
        # Normal parameters related to seeg_he2023xuanwu dataset.
        if self.model.dataset == "seeg_he2023xuanwu":
            # The dimensions of the hidden layers.
            self.model.cls.d_hidden = [128,]
            # The dropout probability after the hidden layers.
            self.model.cls.dropout = 0.
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.model.dataset == "eeg_zhou2023cibr":
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
        # The number of initials.
        self.model.cls.n_initials = 22
        # The number of finals.
        self.model.cls.n_finals = 36

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
            self.train.lr_factors = (5e-5, 2e-4)
        # Normal parameters related to eeg_zhou2023cibr dataset.
        elif self.train.dataset == "eeg_zhou2023cibr":
            # Number of epochs used in training process.
            self.train.n_epochs = 50
            # Number of warmup epochs.
            self.train.warmup_epochs = 5
            # Number of batch size used in training process.
            # Note: `64` is best for one subject. All subjects share one batch.
            self.train.batch_size = 64
            # The learning rate factors of training process.
            self.train.lr_factors = (5e-6, 2e-4)
        # Normal parameters related to other dataset.
        else:
            # Number of epochs used in training process.
            self.train.n_epochs = 100
            # Number of warmup epochs.
            self.train.warmup_epochs = 10
            # Number of batch size used in training process.
            self.train.batch_size = 16
            # The learning rate factors of training process.
            self.train.lr_factors = (1e-5, 3e-4)

if __name__ == "__main__":
    # Instantiate `duin_params`.
    duin_params_inst = duin_params(dataset="seeg_he2023xuanwu")
    # Instantiate `duin_vqvae_params`.
    duin_vqvae_params_inst = duin_vqvae_params(dataset="seeg_he2023xuanwu")
    # Instantiate `duin_mae_params`.
    duin_mae_params_inst = duin_mae_params(dataset="seeg_he2023xuanwu")
    # Instantiate `duin_cls_params`.
    duin_cls_params_inst = duin_cls_params(dataset="seeg_he2023xuanwu")
    # Instantiate `duin_llm_params`.
    duin_llm_params_inst = duin_llm_params(dataset="seeg_he2023xuanwu")

