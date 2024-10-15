#!/usr/bin/env python3
"""
Created on 17:28, Jan. 20th, 2024

@author: Norbert Zheng
"""
import torch
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
import utils.model.torch

__all__ = [
    "ContrastiveBlock",
]

# def ContrastiveBlock class
class ContrastiveBlock(nn.Module):
    """
    Contrastive Block used to calculate contrastive loss.
    """

    def __init__(self, d_model, d_contra, loss_mode, **kwargs):
        """
        Initialize `ContrastiveBlock` object.

        Args:
            d_model: int - The dimensions of model embedding.
            d_contra: int - The dimensions of contrastive space after projection layer.
            loss_mode: str - The mode of loss calculation.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(ContrastiveBlock, self).__init__(**kwargs)

        # Initialize parameters.
        # Note: The dimensions of contrastive space prefers smaller value compared to the original space.
        assert loss_mode in ["clip", "clip_orig", "unicl"], (
            "ERROR: Unknown loss mode {} in layers.ContrastiveBlock."
        ).format(loss_mode)
        self.d_model = d_model; self.d_contra = d_contra; self.loss_mode = loss_mode

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
        # Initialize temperature variables according to `loss_mode`.
        if self.loss_mode == "clip":
            self.tau = nn.Parameter(torch.tensor(0.25, dtype=torch.float32), requires_grad=False)
        elif self.loss_mode == "clip_orig":
            self.t = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=False)
        elif self.loss_mode == "unicl":
            self.t = nn.Parameter(torch.tensor(2.0, dtype=torch.float32), requires_grad=False)
        # Initialize projection layers for `Z` & `Y`.
        # proj_z - (batch_size, *, d_model) -> (batch_size, ? * d_contra)
        self.proj_z = nn.Sequential()
        if self.d_contra is not None:
            self.proj_z.append(nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.d_model, out_features=self.d_contra,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ))
        self.proj_z.append(nn.Flatten(start_dim=1, end_dim=-1))
        # proj_y - (batch_size, *, d_model) -> (batch_size, ? * d_contra)
        self.proj_y = nn.Sequential()
        if self.d_contra is not None:
            self.proj_y.append(nn.Linear(
                # Modified `Linear` layer parameters.
                in_features=self.d_model, out_features=self.d_contra,
                # Default `Linear` layer parameters.
                bias=True, device=None, dtype=None
            ))
        self.proj_y.append(nn.Flatten(start_dim=1, end_dim=-1))

    # def _init_weight func
    def _init_weight(self):
        """
        Initialize model weights.

        Args:
            None

        Returns:
            None
        """
        # Initialize weights for `proj_z`.
        for module_i in self.proj_z.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)
        # Initialize weights for `proj_y`.
        for module_i in self.proj_y.modules():
            if isinstance(module_i, nn.Linear):
                nn.init.trunc_normal_(module_i.weight, mean=0., std=0.02)
                if module_i.bias is not None: nn.init.constant_(module_i.bias, val=0.)

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `ContrastiveBlock` to get the final result.

        Args:
            inputs: (2[list],) - The input data, including [Z,Y].

        Returns:
            loss: torch.float32 - The corresponding contrastive loss.
            prob_matrix: (batch_size, batch_size) - The un-normalized probability matrix.
        """
        # Initialize `Z` & `Y` from `inputs`.
        # [Z,Y] - (batch_size, *, d_model), label - (batch_size, n_labels)
        X_f, y_true = inputs; Z, Y = X_f; label_z, label_y = y_true
        # Use `proj_*` layers to get the embedding.
        # emb_[z,y] - (batch_size, ? * d_contra)
        emb_z = utils.model.torch.normalize(self.proj_z(Z), p=2., dim=-1, eps=1e-12)
        emb_y = utils.model.torch.normalize(self.proj_z(Y), p=2., dim=-1, eps=1e-12)
        # Calculate `loss` and related matrices according to `loss_mode`.
        if self.loss_mode == "clip":
            # Calculate `loss_matrix` from `emb_z` and `emb_y`.
            # loss_matrix - (batch_size, batch_size)
            loss_matrix = torch.exp(torch.matmul(emb_z, torch.permute(emb_y, dims=[1,0])) / self.tau)
            # Calculate `loss_z` & `loss_y` from `loss_matrix`, which is `z`x`y`.
            # loss_[z,y] - (batch_size,), loss - torch.float32
            labels = torch.eye(loss_matrix.shape[0], dtype=loss_matrix.dtype)
            loss_z = torch.squeeze(torch.subtract(
                torch.log(torch.sum(loss_matrix, dim=0, keepdim=True)),
                torch.log(torch.sum(torch.multiply(loss_matrix, labels), dim=0, keepdim=True))
            ))
            loss_y = torch.squeeze(torch.subtract(
                torch.log(torch.sum(loss_matrix, dim=1, keepdim=True)),
                torch.log(torch.sum(torch.multiply(loss_matrix, labels), dim=1, keepdim=True))
            ))
            loss = (torch.mean(loss_z) + torch.mean(loss_y)) / 2
        elif self.loss_mode == "clip_orig":
            # Calculate `loss_matrix` from `emb_z` and `emb_y`.
            # loss_matrix - (batch_size, batch_size)
            loss_matrix = torch.matmul(emb_z, torch.permute(emb_y, dims=[1,0])) * torch.exp(self.t)
            # Calculate `loss_z` & `loss_y` from `loss_matrix`, which is `z`x`y`.
            # loss_[z,y] - (batch_size,), loss - torch.float32
            labels = torch.eye(loss_matrix.shape[0], dtype=loss_matrix.dtype)
            loss_z = utils.model.torch.cross_entropy(logits=loss_matrix, target=labels, dim=-1)
            loss_y = utils.model.torch.cross_entropy(logits=loss_matrix, target=labels, dim=0)
            loss = (torch.mean(loss_z) + torch.mean(loss_y)) / 2
        elif self.loss_mode == "unicl":
            # Calculate `loss_matrix` from `emb_z` and `emb_y`.
            # loss_matrix - (batch_size, batch_size)
            loss_matrix = torch.matmul(emb_z, torch.permute(emb_y, dims=[1,0])) * torch.exp(self.t)
            # Construct `labels` according to one-hot `labels`.
            # labels - (batch_size, batch_size)
            labels = torch.matmul(label_z, torch.permute(label_y, dims=[1,0]))
            # Calculate `loss_z` & `loss_y` from `loss_matrix`, which is `z`x`y`.
            # loss_[z,y] - (batch_size,), loss - torch.float32
            loss_z = utils.model.torch.cross_entropy(logits=loss_matrix, target=labels, dim=-1)
            loss_y = utils.model.torch.cross_entropy(logits=loss_matrix, target=labels, dim=0)
            loss = (torch.mean(loss_z) + torch.mean(loss_y)) / 2
        # Return the final `loss` & `prob_matrix`.
        return loss, loss_matrix

if __name__ == "__main__":
    import numpy as np

    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128; n_labels = 10
    d_contra = 32; loss_mode = ["clip", "clip_orig", "unicl"][-1]

    # Initialize input `X_f` (including `Z` & `Y`) and `y_true` (including `label_z` & `label_y`).
    # [Z,Y] - (batch_size, emb_len, d_model)
    Z = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    Y = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # label_[z,y] - (batch_size, n_labels)
    label_z = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    label_y = torch.tensor(np.eye(n_labels)[np.random.randint(0, n_labels, size=(batch_size,))], dtype=torch.float32)
    # Instantiate ContrastiveBlock.
    cb_inst = ContrastiveBlock(d_model=d_model, d_contra=d_contra, loss_mode=loss_mode)
    # Forward layers in `cb_inst`.
    # loss - torch.float32, prob_matrix - (batch_size, batch_size)
    loss, prob_matrix = cb_inst(((Z, Y), (label_z, label_y)))

