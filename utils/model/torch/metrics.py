#!/usr/bin/env python3
"""
Created on 15:54, Apr. 6th, 2022

@author: Norbert Zheng
"""
import torch
import os, random
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))

__all__ = [
    # Config Functions.
    "set_seeds",
    # Functional Functions.
    "normalize",
    "softmax",
    "cross_entropy",
    "mse_loss",
]

"""
config funcs
"""
# def set_seeds func
def set_seeds(seed=42):
    """
    Set random seeds to ensure that results can be reproduced.
    :param seed: The random seed.
    """
    # Set random seeds.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); 
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    # Enable tf global determinism.
    torch.backends.cudnn.deterministic = True

"""
functional funcs
"""
# def normalize func
def normalize(x, p=2., dim=-1, eps=1e-12):
    """
    Perform L_{p} normalization over specified dimension.

    Args:
        x: torch.Tensor - The unnormalized value.
        p: float - The exponent value in the norm formulation.
        dim: int - The dimension along which to apply normalization.
        eps: float - The small value to avoid division by zero.

    Returns:
        x_normed: torch.Tensor - The normalized value.
    """
    # Calculate the L_{p}-normalized value.
    # x_normed - torch.Tensor
    x_normed = x / torch.norm(x, p=p, dim=dim, keepdim=True).clamp_min(eps)
    # Return the final `x_normed`.
    return x_normed

# def softmax func
def softmax(logits, dim=-1):
    """
    Apply a softmax function over specified dimension.

    Args:
        logits: torch.Tensor - The unnormalized logits.
        dim: int - The dimension along which to apply softmax.

    Returns:
        probs: torch.Tensor - The normalized probabilities.
    """
    # Calculate the softmax-normalized probabilities.
    # probs - torch.Tensor
    probs = torch.exp(logits) / torch.sum(torch.exp(logits), dim=dim, keepdim=True)
    # Return the final `probs`.
    return probs

# def cross_entropy func
def cross_entropy(logits, target, dim=-1):
    """
    Calculate the cross-entropy loss between logits and target.

    Args:
        logits: torch.Tensor - The unnormalized logits.
        target: torch.Tensor - The ground truth class probabilities.
        dim: int - The dimension along which to calculate cross-entropy loss.

    Returns:
        loss: torch.Tensor - The cross-entropy loss.
    """
    # Get the softmax-normalized probabilities.
    # probs - torch.Tensor
    probs = softmax(logits, dim=dim)
    # Calculate the cross-entropy loss.
    # loss - torch.Tensor
    loss = -torch.sum(target * torch.log(probs + 1e-12), dim=dim, keepdim=True)
    # Return the final `loss`.
    return loss

# def mse_loss func
def mse_loss(value, target, dim=-1):
    """
    Calculate the element-wise mean-squared-error loss between value and target.

    Args:
        value: torch.Tensor - The input value.
        target: torch.Tensor - The ground truth value.
        dim: int - The dimension along which to calculate mean-squared-error loss.

    Return:
        loss: torch.Tensor - The mean-squared-error loss.
    """
    # Calculate the mean-squared-error loss.
    # loss - (batch_size,)
    loss = torch.mean(torch.square(target - value), dim=dim, keepdim=True)
    # Return the final `loss`.
    return loss

if __name__ == "__main__":
    print("torch: Hello World!")

