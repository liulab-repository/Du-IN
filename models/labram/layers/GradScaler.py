#!/usr/bin/env python3
"""
Created on 20:31, Jan. 16th, 2024

@author: Norbert Zheng
"""
import torch
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    "GradScaler",
]

# def GradScaler class
class GradScaler(torch.autograd.Function):
    """
    Scale the gradient-flow during backpropagation.
    """

    # def forward func
    @staticmethod
    def forward(ctx, x, scale):
        """
        Forward operations in `GradScaler`.

        Args:
            ctx: torch.autograd.function.GradScalerBackward - The gradient-flow context.
            x: torch.Tensor - The intermediate vector during forward process.
            scale: float - The scalar to scale the gradient-flow during back-propagation.

        Returns:
            y: torch.Tensor - The intermediate vector during forward process (w/ no change).
        """
        # Initialize operation context, which can be used in the following backward process.
        # Note: The context can be used to store arbitrary data that can be then retrieved
        # during the backward pass. Tensors should not be stored directly on `ctx` (though
        # this is not currently enforced for backward compatibility). Instead, tensors should
        # be saved either with `ctx.save_for_backward()` if they are intended to be used in
        # `backward` (equivalently, `vjp`) or `ctx.save_for_forward()` if they are intended
        # to be used for in `jvp`. More details can be found in:
        # - https://pytorch.org/docs/stable/generated/torch.autograd.Function.forward.html#torch.autograd.Function.forward
        # - https://pytorch.org/docs/stable/notes/extending.html#combining-forward-context
        ctx.scale = scale
        # Return the final `y`.
        return x.new(x)

    # def backward func
    @staticmethod
    def backward(ctx, grad):
        """
        Scale the gradient-flow during backward process.

        Args:
            ctx: torch.autograd.function.GradScalerBackward - The gradient-flow context.
            grad: torch.Tensor - The gradients from upstream, i.e., `dy`.

        Returns:
            grad: torch.Tensor - The gradients at current backward step.
        """
        return (ctx.scale * grad), None

if __name__ == "__main__":
    # Initialize macros.
    batch_size = 32; emb_len = 20; d_model = 128; scale = 1.

    # Initialize input `x`.
    # x - (batch_size, emb_len, d_model)
    x = torch.rand((batch_size, emb_len, d_model), dtype=torch.float32)
    # Forward `GradScaler` to check function.
    # y - (batch_size, emb_len, d_model)
    y = GradScaler.apply(x, scale)

