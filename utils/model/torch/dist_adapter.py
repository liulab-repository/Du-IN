#!/usr/bin/env python3
"""
Created on 16:20, Feb. 20th, 2024

@author: Norbert Zheng
"""
import torch.distributed as distributed
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))
from utils import DotDict

__all__ = [
    # Macros.
    "ReduceOp",
    # State Functions.
    "is_available",
    # Init Functions.
    "init_process_group",
    # Config Functions.
    "get_rank",
    "get_world_size",
    # Sync Functions.
    "barrier",
    "all_gather",
    "all_reduce",
    "reduce",
    "broadcast",
]

"""
macros
"""
# def ReduceOp macro
ReduceOp = DotDict({
    "SUM": distributed.ReduceOp.SUM, "AVG": distributed.ReduceOp.AVG, "PRODUCT": distributed.ReduceOp.PRODUCT,
    "MIN": distributed.ReduceOp.MIN, "MAX": distributed.ReduceOp.MAX, "BAND": distributed.ReduceOp.BAND,
    "BOR": distributed.ReduceOp.BOR, "BXOR": distributed.ReduceOp.BXOR, "PREMUL_SUM": distributed.ReduceOp.PREMUL_SUM,
})

"""
state funcs
"""
# def is_available func
def is_available():
    """
    Check whether the distributed package is available and the default process group has been initialized.

    Args:
        None

    Returns:
        state: bool - The flag that indicates whether distributed training is enabled.
    """
    return (distributed.is_available() and distributed.is_initialized())

"""
init funcs
"""
# def init_process_group func
def init_process_group(backend, init_method):
    """
    Initialize the default distributed process group, and this will also initialize the distributed package.
    There are 2 main ways to initialize a process group:
     - Specify `store`, `rank`, and `world_size` explicitly.
     - Specify `init_method` (a URL string) which indicates where/how to discover peers. Optionally specify `rank`
       and `world_size`, or encode all required parameters in the URL and omit them.
    If neither is specified, `init_method` is assumed to be `env://`.

    Args:
        backend: str or Backend - The backend to use.
        init_method: str - The URL specifying how to initialize the process group.

    Returns:
        None
    """
    return (distributed.init_process_group(backend=backend, init_method=init_method) if is_available() else None)

"""
config funcs
"""
# def get_rank func
def get_rank():
    """
    Get the rank of the current process in the provided process group
    (or the default process group if none is provided).

    Args:
        None

    Returns:
        rank: int - The rank of the process group.
    """
    return (distributed.get_rank() if is_available() else 0)

# def get_world_size func
def get_world_size():
    """
    Get the number of processes in the provided process group
    (or the default process group if none is provided).

    Args:
        None

    Returns:
        world_size: int - The world size of the process group.
    """
    return (distributed.get_world_size() if is_available() else 1)

"""
sync funcs
"""
# def barrier func
def barrier():
    """
    Synchronizes all processes. This collective blocks processes until the whole group enters
    this function, if `async_op` is `False`, or if async work handle is called on `wait()`.

    Args:
        None

    Returns:
        async_handle: func - The async work handle, if `async_op` is set to `True`.
            `None`, if not `async_op` or if not part of the group.
    """
    return (distributed.barrier() if is_available() else None)

# def all_gather func
def all_gather(tensor_list, tensor):
    """
    Gathers tensors from the whole group in a list. Complex tensors are supported.

    Args:
        tensor_list: list - The output list. It should contain correctly-sized tensors to be used for output of the collective.
        tensor: torch.Tensor - The tensor to be broadcast from current process.

    Returns:
        async_handle: func - The async work handle, if `async_op` is set to `True`.
            `None`, if not `async_op` or if not part of the group.
    """
    # If the distributed training is enabled.
    if is_available():
        return distributed.all_gather(tensor_list=tensor_list, tensor=tensor)
    # Otherwise, set the first element of `tensor_list`, then return.
    else:
        tensor_list[0] = tensor; return None

# def all_reduce func
def all_reduce(tensor, op=ReduceOp.SUM):
    """
    Reduce the tensor data across all machines in such a way that all get the final result.
    After the call, `tensor` is going to be bit-wise identical in all processes. Complex tensors are supported.

    Args:
        tensor: torch.Tensor - The input and output of the collective. The function operates in-place.
        op: int - One of the value from `torch.distributed.ReduceOp` enum. Specifies an operation used for element-wise reductions.

    Returns:
        async_handle: func - The async work handle, if `async_op` is set to `True`.
            `None`, if not `async_op` or if not part of the group.
    """
    return (distributed.all_reduce(tensor=tensor, op=op) if is_available() else None)

# def reduce func
def reduce(tensor, dst, op=ReduceOp.SUM):
    """
    Reduce the tensor data across all machines. Only the process with rank `dst` is going to receive the final result.

    Args:
        tensor: torch.Tensor - The input and output of the collective. The function operates in-place.
        dst: int - The destination rank.
        op: int - One of the value from `torch.distributed.ReduceOp` enum. Specifies an operation used for element-wise reductions.

    Returns:
        async_handle: func - The async work handle, if `async_op` is set to `True`.
            `None`, if not `async_op` or if not part of the group.
    """
    return (distributed.reduce(tensor=tensor, dst=dst, op=op) if is_available() else None)

# def broadcast func
def broadcast(tensor, src):
    """
    Broadcast the tensor to the whole group. `tensor` must have the same number of elements
    in all processes participating in the collective.

    Args:
        tensor: torch.Tensor - The data to be sent if `src` is the rank of current process,
            and the tensor to be used to save received data otherwise.
        src: int - The source rank.

    Returns:
        async_handle: func - The async work handle, if `async_op` is set to `True`.
            `None`, if not `async_op` or if not part of the group.
    """
    return (distributed.broadcast(tensor=tensor, src=src) if is_available() else None)

if __name__ == "__main__":
    print("dist_adapter: Hello World!")

