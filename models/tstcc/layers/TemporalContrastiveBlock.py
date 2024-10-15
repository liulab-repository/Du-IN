#!/usr/bin/env python3
"""
Created on 18:54, Aug. 2rd, 2024

@author: Norbert Zheng
"""
import torch
import copy as cp
import numpy as np
import torch.nn as nn
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
    from attention import Seq_Transformer
    from loss import NTXentLoss
else:
    from models.tstcc.layers.attention import Seq_Transformer
    from models.tstcc.layers.loss import NTXentLoss

__all__ = [
    "TemporalContrastiveBlock",
]

# def TemporalContrastiveBlock class
class TemporalContrastiveBlock(nn.Module):
    """
    Temporal-Contrastive block.
    """

    def __init__(self, params, **kwargs):
        """
        Initialize `TemporalContrastiveBlock` object.

        Args:
            params: DotDict - The parameters of `TemporalContrastiveBlock`.
            kwargs: dict - The arguments related to initialize `nn.Module`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `nn.Module`
        # style model and inherit it's functionality.
        super(TemporalContrastiveBlock, self).__init__(**kwargs)

        # Initialize parameters.
        self.params = cp.deepcopy(params)

        # Initialize weights.
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
        # Initialize transformer stack.
        self.seq_transformer = Seq_Transformer(patch_size=self.params.d_token, dim=self.params.d_model, depth=4, heads=4, mlp_dim=64)
        # Initialize transition-back projection.
        self.Wk = nn.ModuleList([nn.Linear(self.params.d_model, self.params.d_token) for i in range(self.params.n_steps)])
        self.lsoftmax = nn.LogSoftmax()
        # Initialize final projection.
        self.projection_head = nn.Sequential(
            nn.Linear(self.params.d_model, self.params.d_token // 2),
            nn.BatchNorm1d(self.params.d_token // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.params.d_token // 2, self.params.d_token // 4),
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
        pass

    """
    network funcs
    """
    # def forward func
    def forward(self, inputs):
        """
        Forward layers in `TemporalContrastiveBlock` to get the predicted labels.

        Args:
            inputs: tuple - The input data, including [T_aug1,T_aug2].

        Returns:
            y_pred: (batch_size, n_labels) - The predicted labels.
        """
        # Initialize `T_aug*` from `inputs`.
        # T_aug* - (batch_size, token_len, d_token)
        T_aug1 = inputs[0]; T_aug2 = inputs[1]
        # Get loss & feature.
        temp_cont_loss1, temp_cont_lstm_feat1 = self._forward_helper(T_aug1, T_aug2)
        temp_cont_loss2, temp_cont_lstm_feat2 = self._forward_helper(T_aug2, T_aug1)
        # normalize projection feature vectors
        zis = temp_cont_lstm_feat1
        zjs = temp_cont_lstm_feat2
        # compute loss
        lambda1 = 1
        lambda2 = 0.7
        batch_size = T_aug1.shape[0]
        nt_xent_criterion = NTXentLoss(T_aug1.device, batch_size, 0.2,
                                       True)
        loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
        # Return the final `loss`.
        return loss

    # def _forward_helper func
    def _forward_helper(self, T_aug1, T_aug2):
        """
        Forward layers in `TemporalContrastiveBlock` to get the predicted labels.

        Args:
            inputs: tuple - The input data, including [T_aug1,T_aug2].

        Returns:
            y_pred: (batch_size, n_labels) - The predicted labels.
        """
        # Initialize `T_aug*` from `inputs`.
        # T_aug* - (batch_size, token_len, d_token)
        batch_size, token_len, d_token = T_aug1.shape
        # Randomly pick the start time step.
        t_samples = torch.randint(token_len - self.params.n_steps, size=(1,)).long().to(device=T_aug1.device)
        # Initialize `nce`, which is average over time steps and batch.
        # encode_samples - (n_steps, batch_size, d_token)
        nce = 0; encode_samples = torch.empty((self.params.n_steps, batch_size, self.params.d_token)).float().to(device=T_aug1.device)
        # Fill up `encode_samples`.
        for i in np.arange(1, self.params.n_steps + 1):
            encode_samples[i - 1] = T_aug2[:, t_samples + i, :].view(batch_size, self.params.d_token)
        forward_seq = T_aug1[:, :t_samples + 1, :]
        # Forward them.
        c_t = self.seq_transformer(forward_seq)
        pred = torch.empty((self.params.n_steps, batch_size, self.params.d_token)).float().to(device=T_aug1.device)
        for i in np.arange(0, self.params.n_steps):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.params.n_steps):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch_size * self.params.n_steps
        return nce, self.projection_head(c_t)

if __name__ == "__main__":
    print("TemporalContrastiveBlock: Hello World!")

