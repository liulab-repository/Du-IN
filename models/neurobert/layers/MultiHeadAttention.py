#!/usr/bin/env python3
"""
Created on 20:39, Jan. 20th, 2024

@author: Norbert Zheng
"""
import torch
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
import utils.model.torch

__all__ = [
    "MultiHeadAttention",
]

# def MultiHeadAttention class
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention used to calculate the scaled multi-head attention.
    """

    def __init__(self, d_model, n_heads, d_head, attn_dropout=0., proj_dropout=0., emb_rotary=None, use_bias=True, **kwargs):
        """
        Initialize `MultiHeadAttention` object.

        Args:
            d_model: int - The dimensions of model embedding.
            n_heads: int - The number of attention heads.
            d_head: int - The dimensions of attention head.
            attn_dropout: float - The probability of attention score dropout.
            proj_dropout: float - The probability of projection dropout.
            emb_rotary: K.layers.Layer - The rotary embedding layer.
            use_bias: bool - The flag indicates whether use bias.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(MultiHeadAttention, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_model = d_model; self.n_heads = n_heads; self.d_head = d_head
        self.attn_dropout = attn_dropout; self.proj_dropout = proj_dropout
        self.emb_rotary = emb_rotary; self.use_bias = use_bias

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
        # Initialize the query & key & value transformation matrices (perhaps w. bias).
        # W_[q,k,v] - (batch_size, emb_len, d_model) -> (batch_size, emb_len, n_heads, d_head)
        self.W_q = MHAMatrix(d_model=self.d_model, n_heads=self.n_heads, d_head=self.d_head, use_bias=self.use_bias)
        self.W_k = MHAMatrix(d_model=self.d_model, n_heads=self.n_heads, d_head=self.d_head, use_bias=self.use_bias)
        self.W_v = MHAMatrix(d_model=self.d_model, n_heads=self.n_heads, d_head=self.d_head, use_bias=self.use_bias)
        # Initialize the normalization layer for query & key embeddings.
        self.norm_q = nn.LayerNorm(
            # Modified `LayerNorm` layer parameters.
            normalized_shape=(self.d_head,),
            # Default `LayerNorm` layer parameters.
            eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None
        )
        self.norm_k = nn.LayerNorm(
            # Modified `LayerNorm` layer parameters.
            normalized_shape=(self.d_head,),
            # Default `LayerNorm` layer parameters.
            eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None
        )
        # Initialize the scaled dot-product attention layer.
        self.attention = ScaledDotProductAttention(d_head=self.d_head, attn_dropout=self.attn_dropout, scale_trainable=False)
        # Initialize the projection layer.
        self.proj = nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=(self.n_heads * self.d_head), out_features=self.d_model,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ),
            nn.Dropout(p=self.proj_dropout, inplace=False),
        )

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for `norm_q`.
        for module_i in self.norm_q.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)
        # Initialize weights for `norm_k`.
        for module_i in self.norm_k.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)
        # Initialize weights for `proj`.
        for module_i in self.proj.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, embs, attn_score=None, attn_mask=None, key_padding_mask=None):
        """
        Forward layers in `MultiHeadAttention` to get the multi-head attention embeddings.

        Args:
            embs: tuple - The embeddings containing emb_[q,k,v], each element is (batch_size, emb_len, d_input).
            attn_score: (batch_size, n_heads, emb_len, emb_len) - The attention score from the previous layer.
            attn_mask: (emb_len, emb_len) - The pre-defined attention mask within sequence.
            key_padding_mask: (batch_size, emb_len) - The pre-defined key mask within sequence.

        Returns:
            emb: (batch_size, emb_len, d_model) - The multi-head attention embeddings.
            attn_weight: (batch_size, n_heads, emb_len, emb_len) - The attention weight.
            attn_score: (batch_size, n_heads, emb_len, emb_len) - The attention score.
        """
        # Initialize `emb_q` & `emb_k` & `emb_v` from `embs`.
        # emb_[q,k,v] - (batch_size, emb_len, d_model)
        emb_q, emb_k, emb_v = embs
        # Prepare query & key & value for attention computation.
        # TODO: Use l2-normalization, instead of layer norm, to further limit the scale,
        # thus avoiding numerical explotion when calling softmax over `attn_score`.
        # emb_[q,k,v] - (batch_size, n_heads, emb_len, d_head)
        emb_q = self.norm_q(torch.permute(self.W_q(emb_q), dims=[0,2,1,3]))
        emb_k = self.norm_k(torch.permute(self.W_k(emb_k), dims=[0,2,1,3]))
        emb_v = torch.permute(self.W_v(emb_v), dims=[0,2,1,3])
        # If `emb_rotary` is not Nnoe, further embed query & key.
        if self.emb_rotary is not None:
            emb_q = self.emb_rotary(emb_q); emb_k = self.emb_rotary(emb_k)
        # Calculate attention results from `emb_*`.
        # emb - (batch_size, n_heads, emb_len, d_head)
        # attn_weight - (batch_size, n_heads, emb_len, emb_len)
        # attn_score - (batch_size, n_heads, emb_len, emb_len)
        emb, attn_weight, attn_score = self.attention((emb_q, emb_k, emb_v),
            attn_score=attn_score, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        # Transpose `emb` to the original dimensions.
        # emb - (batch_size, emb_len, n_heads, d_head)
        emb = torch.permute(emb, dims=[0,2,1,3])
        # Concatenate multiple heads.
        # emb - (batch_size, emb_len, n_heads * d_head)
        emb = torch.reshape(emb, shape=(*emb.shape[:-2], -1))
        # Project `emb` to the original dimensions.
        # emb - (batch_size, emb_len, d_model)
        emb = self.proj(emb)
        # Return the final `emb` & `attn_weight` & `attn_score`.
        return emb, attn_weight, attn_score

# def MHAMatrix class
class MHAMatrix(nn.Module):
    """
    Multi-Head Attention Matrix does a linear transformation and splits the vector into given number of heads
    for multi-head attention. This is used to transform key, query, and value vectors.
    """

    def __init__(self, d_model, n_heads, d_head, use_bias=True, **kwargs):
        """
        Initialize `MHAMatrix` object.

        Args:
            d_model: int - The dimensions of model embedding.
            n_heads: int - The number of attention heads.
            d_head: int - The dimensions of attention head.
            use_bias: bool - The flag indicates whether use bias.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(MHAMatrix, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_model = d_model; self.n_heads = n_heads; self.d_head = d_head; self.use_bias = use_bias

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
        # Initialize the transformation matrix (perhaps w. bias).
        # W - (batch_size, emb_len, d_input) -> (batch_size, emb_len, n_heads * d_head)
        self.W = nn.Linear(
            # Modified `Linear` layer parameters.
            in_features=self.d_model, out_features=(self.n_heads * self.d_head), bias=self.use_bias,
            # Default `Linear` layer parameters.
            device=None, dtype=None
        )

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for model.
        for module_i in self.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, emb):
        """
        Forward layers in `MHAMatrix` to get the linear-transformed embeddings.

        Args:
            emb: (batch_size, emb_len, d_input) - The input embeddings.

        Returns:
            emb: (batch_size, emb_len, n_heads, d_head) - The linear-transformed embeddings.
        """
        # Get the shape of head from `emb`.
        # head_shape - tuple, should be (batch_size, emb_len)
        head_shape = emb.shape[:-1]
        # Linearly transform `emb` using `W`.
        # emb - (batch_size, emb_len, n_heads, d_head)
        emb = torch.reshape(self.W(emb), shape=(*head_shape, self.n_heads, self.d_head))
        # Return the final `emb`.
        return emb

# def ScaledDotProductAttention class
class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with
    optional residual attention from previous layer (Realformer: Transformer likes residual attention by He et al, 2020)
    and locality self sttention (Vision Transformer for Small-Size Datasets by Lee et al, 2021).
    """

    def __init__(self, d_head, attn_dropout=0., scale_trainable=False, **kwargs):
        """
        Initialize `ScaledDotProductAttention` object.

        Args:
            d_head: int - The dimensions of attention head.
            attn_dropout: float - The probability of dropout.
            scale_trainable: bool - The flag that indicates whether scale factor is trainable.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(ScaledDotProductAttention, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_head = d_head; self.attn_dropout = attn_dropout; self.scale_trainable = scale_trainable

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
        # Initialize the dropout layer.
        self.dropout = nn.Dropout(p=self.attn_dropout, inplace=False)
        # Initialize scale factor.
        self.scale = nn.Parameter(torch.tensor(1. / np.sqrt(self.d_head), dtype=torch.float32), requires_grad=self.scale_trainable)

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
    network funcs
    """
    # def forward func
    def forward(self, embs, attn_score=None, attn_mask=None, key_padding_mask=None):
        """
        Forward layers in `ScaledDotProductAttention` to get the attention embeddings.

        Args:
            embs: tuple - The embeddings containing emb_[q,k,v], each element is (batch_size, n_heads, emb_len, d_head).
            attn_score: (batch_size, n_heads, emb_len, emb_len) - The attention score from the previous layer.
            attn_mask: (emb_len, emb_len) - The pre-defined attention mask within sequence.
            key_padding_mask: (batch_size, emb_len) - The pre-defined key mask within sequence.

        Returns:
            emb: (batch_size, n_heads, emb_len, d_head) - The attention embeddings.
            attn_weight: (batch_size, n_heads, emb_len, emb_len) - The attention weight.
            attn_score: (batch_size, n_heads, emb_len, emb_len) - The attention score.
        """
        # Initialize `emb_q` & `emb_k` & `emb_v` from `embs`.
        # emb_[q,k,v] - (batch_size, n_heads, emb_len, d_head)
        emb_q, emb_k, emb_v = embs
        # Calculate scaled similarity score for all pairs of positions in an input sequence.
        # Here, we support residual attention in Realformer by He et al, 2020.
        # attn_score - (batch_size, n_heads, emb_len, emb_len)
        attn_score = (torch.matmul(emb_q, torch.permute(emb_k, dims=[0,1,3,2])) * self.scale) if attn_score is None else\
            (torch.matmul(emb_q, torch.permute(emb_k, dims=[0,1,3,2])) * self.scale) + attn_score
        # Use pre-defined attention mask to introduce inductive bias.
        # Here, we support locality self attention in Small-Size Dataset ViT by Lee et al, 2021.
        if attn_mask is not None:
            attn_score = torch.where(torch.unsqueeze(torch.unsqueeze(attn_mask, dim=0), dim=0), -np.inf, attn_score)
        # Use pre-defined key padding mask to ignore some keys and their corresponding values.
        if key_padding_mask is not None:
            attn_score = torch.where(torch.unsqueeze(torch.unsqueeze(key_padding_mask, dim=1), dim=2), -np.inf, attn_score)
        # Normalize the attention score to get attention weight.
        # attn_weight - (batch_size, n_heads, emb_len, emb_len)
        attn_weight = self.dropout(utils.model.torch.softmax(attn_score, dim=-1))
        # Calculate the attention embedding.
        # emb - (batch_size, n_heads, emb_len, d_head)
        emb = torch.matmul(attn_weight, emb_v)
        # Return the final `emb` & `attn_weight` & `attn_score`.
        return emb, attn_weight, attn_score

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; seq_len = 80; d_model = 128
    n_heads = 16; d_head = 64; attn_dropout = 0.4; proj_dropout = 0.; use_bias = True

    # Initialize input `embs`.
    # embs - (3[list], batch_size, emb_len, d_model)
    embs = [torch.rand((batch_size, seq_len, d_model), dtype=torch.float32) for _ in range(3)]
    # Instantiate `MultiHeadAttention`.
    mha_inst = MultiHeadAttention(d_model=d_model, n_heads=n_heads, d_head=d_head,
        attn_dropout=attn_dropout, proj_dropout=proj_dropout, use_bias=use_bias)
    # Forward `mha_inst` with random input.
    emb = mha_inst(embs)

