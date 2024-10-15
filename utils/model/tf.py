#!/usr/bin/env python3
"""
Created on 15:54, Apr. 6th, 2022

@author: Norbert Zheng
"""
import os, random
import functools
import numpy as np
import tensorflow as tf
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))

__all__ = [
    "tf_scope",
    "set_seeds",
]

# def tf_scope func
def tf_scope(func):
    """
    Creates a name_scope that contains all ops created by the function.
    The scope will default to the provided name or to the name of the function
    in CamelCase. If the function is a class constructor, it will default to
    the class name. It can also be specified with name="Name" at call time.
    :param func: The pointer of function.
    :return _wrapper: The wrapper of function.
    """
    # Get the name of func.
    func_name = func.__name__
    if func_name == "__init__":
        func_name = func.__class__.__name__
    # Get the wrapper of func.
    @functools.wraps(func)
    def _wrapper(*args, **kwargs):
        with tf.name_scope(func_name):
            return func(*args, **kwargs)
    # Return the final wrapper.
    return _wrapper

# def set_seeds func
def set_seeds(seed=42):
    """
    Set random seeds to ensure that results can be reproduced.
    :param seed: The random seed.
    """
    # Set random seeds.
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)
    # Enable tf global determinism.
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# def get_device func
def get_device(index=0):
    """
    Try to get the `tf.device` corresponding to gpu index. The default device is "/CPU:0".
    :param index: The index of gpu card belonging to current running partition.
    :return device: The `tf.device` corresponding to gpu index. The default device is "/CPU:0".
    """
    if len(tf.config.experimental.list_logical_devices("GPU")) >= index + 1:
        return tf.device("/GPU:{:d}".format(index))
    return tf.device("/CPU:0")

# def avg_grads func
def avg_grads(grads_tower):
    """
    Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    :param grads_tower: A list of lists of (gradient, variable) tuples. The outer list is over
        individual gradients. The inner is over the gradient calculation for each tower.
    :return grads_avg: List of pairs of (gradient, variable) where the gradient has been
        averaged across all towers.
    """
    # Initialize grads_avg.
    grads_avg = []
    for grads_var_i in zip(*grads_tower):
        # Note that each grads_var_i looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ..., (grad0_gpuN, var0_gpuN))
        # The first item of `grads_var_avg` has the same shape with `grad0_gpu0`, averaging over tower dimension.
        # Keep in mind that the Variables are redundant because they are shared across towers.
        # So .. we will just return the first tower's pointer to the Variable.
        grads_var_avg = (tf.reduce_mean(tf.concat(
            [tf.expand_dims(grad_var_i, axis=0) for grad_var_i, _ in grads_var_i]
        , axis=0), axis=0), grads_var_i[0][1])
        # Append the average gradients of current variable to grads_avg.
        grads_avg.append(grads_var_avg)
    # Return the final `grads_avg`.
    return grads_avg

if __name__ == "__main__":
    print("tf: Hello World!")

