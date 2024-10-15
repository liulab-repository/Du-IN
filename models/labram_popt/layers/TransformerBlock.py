#!/usr/bin/env python3
"""
Created on 23:23, Jan. 20th, 2024

@author: Norbert Zheng
"""
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from MultiHeadAttention import MultiHeadAttention
    from FeedForward import FeedForward
    from Embedding import RotaryEmbedding
else:
    from models.labram_popt.layers.MultiHeadAttention import MultiHeadAttention
    from models.labram_popt.layers.FeedForward import FeedForward
    from models.labram_popt.layers.Embedding import RotaryEmbedding

__all__ = [
    "TransformerBlock",
]

# def TransformerBlock class
class TransformerBlock(nn.Module):
    """
    Transformer Block acts as an encoder layer or decoder layer.
    """

    def __init__(self, d_model, n_heads, d_head, attn_dropout, proj_dropout,
        d_ff, ff_dropout, rot_theta=None, norm_first=False, **kwargs):
        """
        Initialize `TransformerBlock` object.

        Args:
            d_model: int - The dimensions of model embedding.
            n_heads: int - The number of attention heads in `mha` block.
            d_head: int - The dimensions of attention head in `mha` block.
            attn_dropout: float - The dropout probability of attention score in `mha` block.
            proj_dropout: float - The dropout probability of projection in `mha` block.
            d_ff: int - The dimensions of the hidden layer in `ffn` block.
            ff_dropout: (2[list],) - The dropout probabilities in `ffn` block.
            rot_theta: float - The power base of rotation angle, default as `None`.
            norm_first: bool - The flag that indicates whether normalize data first.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(TransformerBlock, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_model = d_model; self.n_heads = n_heads; self.d_head = d_head
        self.attn_dropout = attn_dropout; self.proj_dropout = proj_dropout
        self.d_ff = d_ff; self.ff_dropout = ff_dropout
        self.rot_theta = rot_theta; self.norm_first = norm_first

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
        # Initialize `mha` block.
        # mha - (batch_size, emb_len, d_model) -> (batch_size, emb_len, n_heads * d_head)
        emb_rotary = RotaryEmbedding(d_model=self.d_head, theta=self.rot_theta) if self.rot_theta is not None else None
        self.mha = MultiHeadAttention(
            # Modified `MultiHeadAttention` layer parameters.
            d_model=self.d_model, n_heads=self.n_heads, d_head=self.d_head,
            attn_dropout=self.attn_dropout, proj_dropout=self.proj_dropout, emb_rotary=emb_rotary,
            # Default `MultiHeadAttention` layer parameters.
            use_bias=True
        )
        # Initialize the normalization layer of `mha` block.
        self.norm_mha = nn.LayerNorm(
            # Modified `LayerNorm` layer parameters.
            normalized_shape=(self.d_model,),
            # Default `LayerNorm` layer parameters.
            eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None
        )
        # Initialize `ffn` block.
        # ffn - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_model)
        self.ffn = FeedForward(d_model=self.d_model, d_ff=self.d_ff, ff_dropout=self.ff_dropout) if self.d_ff is not None else None
        # Initialize the normalization layer of `ffn` block.
        self.norm_ffn = nn.LayerNorm(
            # Modified `LayerNorm` layer parameters.
            normalized_shape=(self.d_model,),
            # Default `LayerNorm` layer parameters.
            eps=1e-5, elementwise_affine=True, bias=True, device=None, dtype=None
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
        # Initialize weights for `norm_mha`.
        for module_i in self.norm_mha.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)
        # Initialize weights for `norm_ffn`.
        for module_i in self.norm_ffn.modules():
            if isinstance(module_i, nn.LayerNorm):
                if module_i.weight is not None: nn.init.ones_(module_i.weight)
                if module_i.bias is not None: nn.init.zeros_(module_i.bias)

    """
    network funcs
    """
    # def forward func
    def forward(self, emb, attn_score=None, attn_mask=None, key_padding_mask=None):
        """
        Forward layers in `TransformerBlock` to get the mha-ffn transformed embeddings.

        Args:
            emb: (batch_size, emb_len, d_model) - The input embeddings.
            attn_score: (batch_size, n_heads, emb_len, emb_len) - The attention score from the previous layer.
            attn_mask: (emb_len, emb_len) - The pre-defined attention mask within sequence.
            key_padding_mask: (batch_size, emb_len) - The pre-defined key mask within sequence.

        Returns:
            emb: (batch_size, emb_len, d_model) - The mha-ffn transformed embeddings.
            attn_weight: (batch_size, n_heads, emb_len, emb_len) - The attention weight.
            attn_score: (batch_size, n_heads, emb_len, emb_len) - The attention score.
        """
        # Get the mha transformed embeddings.
        # emb - (batch_size, emb_len, d_model)
        # attn_weight - (batch_size, n_heads, emb_len, emb_len)
        # attn_score - (batch_size, n_heads, emb_len, emb_len)
        if self.norm_first: emb = self.norm_mha(emb) if self.norm_mha is not None else emb
        attn_emb, attn_weight, attn_score = self.mha((emb, emb, emb),
            attn_score=attn_score, attn_mask=attn_mask, key_padding_mask=key_padding_mask); emb = attn_emb + emb
        if not self.norm_first: emb = self.norm_mha(emb) if self.norm_mha is not None else emb
        # Get the ffn transformed embedding.
        # emb - (batch_size, emb_len, d_model)
        if self.norm_first: emb = self.norm_ffn(emb) if self.norm_ffn is not None else emb
        emb = self.ffn(emb) + emb if self.ffn is not None else emb
        if not self.norm_first: emb = self.norm_ffn(emb) if self.norm_ffn is not None else emb
        # Return the final `emb` & `attn_weight` & `attn_score`.
        return emb, attn_weight, attn_score

if __name__ == "__main__":
    import torch

    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128; n_heads = 8; d_head = 64; attn_dropout = 0.2
    proj_dropout = 0.; d_ff = d_model * 4; ff_dropout = [0.2, 0.]; rot_theta = None; norm_first = False

    # Initialize input `emb` & `attn_score` & `attn_mask` & `key_padding_mask`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # attn_score - (batch_size, n_heads, emb_len, emb_len)
    attn_score = torch.rand((batch_size, n_heads, emb_len, emb_len), dtype=torch.float32)
    # attn_mask - (emb_len, emb_len)
    attn_mask = torch.rand((emb_len, emb_len), dtype=torch.float32) < 0.5
    # key_padding_mask - (batch_size, emb_len)
    key_padding_mask = torch.rand((batch_size, emb_len), dtype=torch.float32) < 0.5
    # Instantiate TransformerBlock.
    tb_inst = TransformerBlock(d_model=d_model, n_heads=n_heads, d_head=d_head, attn_dropout=attn_dropout,
        proj_dropout=proj_dropout, d_ff=d_ff, ff_dropout=ff_dropout, rot_theta=rot_theta, norm_first=norm_first)
    # Forward layers in `tb_inst`.
    # emb - (batch_size, emb_len, d_model)
    # attn_weight - (batch_size, n_heads, emb_len, emb_len)
    # attn_score - (batch_size, n_heads, emb_len, emb_len)
    emb, attn_weight, attn_score = tb_inst(emb, attn_score=attn_score, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

