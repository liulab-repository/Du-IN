#!/usr/bin/env python3
"""
Created on 22:57, Jan. 20th, 2024

@author: Norbert Zheng
"""
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))

__all__ = [
    "FeedForward",
]

# def FeedForward class
class FeedForward(nn.Module):
    """
    Position-wise feedforward network. FFN consists of two fully connected layers.
    Number of dimensions in the hidden layer $d_{ff}$, is generally set to around
    four times that of the token embedding $d_{model}$. So it is sometime also
    called the expand-and-contract network.
    """

    def __init__(self, d_model, d_ff, ff_dropout, use_bias=[True, True], use_bias_gate=None, **kwargs):
        """
        Initialize `FeedForward` object.

        Args:
            d_model: int - The dimensions of model embedding.
            d_ff: int - The number of features in the hidden layer of the FFN.
            ff_dropout: (2[list],) - The dropout probabilities for the fully connected layers.
            use_bias: (2[list],) - The flags indicate whether the fully connected layers have a learnable bias.
            use_bias_gate: bool - The flag indicates whether the fully connected layer for the gate have a learnable bias.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(FeedForward, self).__init__(**kwargs)

        # Initialize parameters.
        self.d_model = d_model; self.d_ff = d_ff; self.ff_dropout = ff_dropout; self.use_bias = use_bias
        self.is_gated = (use_bias_gate is not None); self.use_bias_gate = use_bias_gate

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
        # Initialize the fully connected layers.
        # fc1 - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_ff)
        self.fc1 = nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.d_model, out_features=self.d_ff, bias=self.use_bias[0],
                # Default `Linear` layer parameters.
                device=None, dtype=None
            ),
            nn.ReLU(inplace=False),
        )
        # fc2 - (batch_size, emb_len, d_ff) -> (batch_size, emb_len, d_model)
        self.fc2 = nn.Sequential(
            nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.d_ff, out_features=self.d_model, bias=self.use_bias[1],
                # Default `Linear` layer parameters.
                device=None, dtype=None
            ),
        )
        # Initialize the dropout layer.
        self.dropout1 = nn.Dropout(p=self.ff_dropout[0], inplace=False)
        self.dropout2 = nn.Dropout(p=self.ff_dropout[1], inplace=False)
        # Initialize the gate layer.
        # gate - (batch_size, emb_len, d_model) -> (batch_size, emb_len, d_ff)
        self.gate = nn.Linear(
            # Modified `Linear` layer parameters.
            in_features=self.d_model, out_features=self.d_ff, bias=self.use_bias_gate,
            # Default `Linear` layer parameters.
            device=None, dtype=None
        ) if self.is_gated else None

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
        Forward layers in `FeedForward` to get the MLP-transformed embeddings.

        Args:
            emb: (batch_size, emb_len, d_model) - The input embeddings.

        Returns:
            emb: (batch_size, emb_len, d_model) - The MLP-transformed embeddings.
        """
        # Get the activation of the hidden layer.
        # emb - (batch_size, emb_len, d_ff)
        emb = self.fc1(emb) * self.gate(emb) if self.is_gated else self.fc1(emb)
        # Apply dropout the hidden layer.
        emb = self.dropout1(emb)
        # Get the activation of the final layer.
        # emb - (batch_size, emb_len, d_model)
        emb = self.fc2(emb)
        # Apply dropout the final layer.
        emb = self.dropout2(emb)
        # Return the final `emb`.
        return emb

if __name__ == "__main__":
    import torch
    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128
    d_ff = d_model * 4; ff_dropout = [0.2, 0.]; use_bias = [True, True]

    # Initialize input `emb`.
    # emb - (batch_size, emb_len, d_model)
    emb = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Instantiate FeedForward.
    ff_inst = FeedForward(d_model=d_model, d_ff=d_ff, ff_dropout=ff_dropout, use_bias=use_bias)
    # Forward layers in `ff_inst`.
    emb = ff_inst(emb)

