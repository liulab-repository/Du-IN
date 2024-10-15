#!/usr/bin/env python3
"""
Created on 16:11, Feb. 20th, 2024

@author: Norbert Zheng
"""
import json, torch
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir))

__all__ = [
    # Init Functions.
    "create_optimizer",
    # Parameter Functions.
    "get_param_groups",
]

"""
init funcs
"""
# def create_optimizer func
def create_optimizer(cfg, model, nodecay_names=None, get_layer_id=None, get_layer_scale=None, **kwargs):
    """
    Create the optimizer according to the optimizer configuration.

    Args:
        cfg: dict - The configuration of optimzier.
        model: nn.Module - The trainable model.
        nodecay_names: list - The list of parameter names to skip, i.e., no decay.
        get_layer_id: func - The map from parameter name to layer id.
        get_layer_scale: func - The map from layer id to layer scale.
        kwargs: dict - The additional arguments to create optimizer.

    Returns:
        optimizer: torch.optim.Optimizer - The created optimizer.
    """
    # Initialize `optim_name` & `weight_decay` from `cfg`.
    optim_name = cfg["name"].lower(); weight_decay = cfg["weight_decay"]
    # Initialize `param_groups` from `model`.
    param_groups = get_param_groups(model=model, weight_decay=weight_decay, nodecay_names=nodecay_names,
        get_layer_id=get_layer_id, get_layer_scale=get_layer_scale, **kwargs)
    # As we already configure the weight decay of specific parameters, do not specify the weight decay of optimizer.
    weight_decay = 0.
    # Initialize the optimizer according to configuration.
    if optim_name == "adam":
        optimizer = torch.optim.Adam(
            # Modified `Adam` optimizer parameters.
            params=param_groups, lr=cfg["lr"], weight_decay=weight_decay,
            # Default `Adam` optimizer parameters.
            betas=(0.9, 0.999), eps=1e-8, amsgrad=False, foreach=None,
            maximize=False, capturable=False, differentiable=False, fused=None
        )
    elif optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            # Modified `AdamW` optimizer parameters.
            params=param_groups, lr=cfg["lr"], weight_decay=weight_decay,
            # Default `AdamW` optimizer parameters.
            betas=(0.9, 0.999), eps=1e-8, amsgrad=False, maximize=False,
            foreach=None, capturable=False, differentiable=False, fused=None
        )
    # Return the final `optimizer`.
    return optimizer

"""
param funcs
"""
# def get_param_groups func
def get_param_groups(model, weight_decay=1e-5, nodecay_names=None, get_layer_id=None, get_layer_scale=None, **kwargs):
    """
    Get the parameter configuration groups used by optimizer initialization.

    Args:
        model: nn.Module - The trainable model.
        weight_decay: float - The weight decay (i.e., L2 penalty).
        nodecay_names: list - The list of parameter names to skip, i.e., no decay.
        get_layer_id: func - The map from parameter name to layer id.
        get_layer_scale: func - The map from layer id to layer scale.
        kwargs: dict - The additional arguments to get parameter groups.

    Returns:
        param_groups: dict - The parameter groups for optimizer initialization.
    """
    # Initialize `nodecay_names` & `ignore_patterns`.
    nodecay_names = nodecay_names if nodecay_names is not None else []
    ignore_patterns = kwargs.get("ignore_patterns", [])
    # Log information related to `nodecay_names`.
    print((
        "INFO: The pre-marked nodecay parameters include {} in utils.model.torch.optim_factory."
    ).format(nodecay_names))
    # Initialize `param_groups_*`.
    param_groups_name = dict(); param_groups_var = dict()
    # Loop over all model.parameters to get the parameter configuration groups.
    for name_i, param_i in model.named_parameters():
        ## Ignore specific parameters.
        # If not trainable, directly ignore it.
        if not param_i.requires_grad: continue
        # If having specific pattern, directly ignore it.
        if len(ignore_patterns) > 0:
            # Loop over `ignore_patterns` to check whether ignore this parameter.
            have_ignore_pattern = False
            for ignore_pattern_i in ignore_patterns:
                if ignore_pattern_i in name_i:
                    have_ignore_pattern = True; print((
                        "INFO: Ignore parameter ({}) because of the ignore_pattern ({}) in utils.model.torch.optim_factory."
                    ).format(name_i, ignore_pattern_i))
            # If matching one of `ignore_patterns`, directly ignore it.
            if have_ignore_pattern: continue
        ## Setup weight decay of specific parameters.
        # If in `nodecay_names` or is bias or is 1D-tensor, no decay.
        if (param_i.ndim <= 1) or name_i.endswith(".bias") or (name_i in nodecay_names):
            group_name_i = "no_decay"; weight_decay_i = 0.
        # Otherwise, decay!
        else:
            group_name_i = "decay"; weight_decay_i = weight_decay
        ## Setup learning rate scale of specific parameters.
        # If `get_layer_*` is not None, update `group_name_i`.
        if (get_layer_id is not None) and (get_layer_scale is not None):
            layer_id_i = get_layer_id(name_i); lr_scale_i = get_layer_scale(layer_id_i)
            group_name_i = "layer_{:d}_{}".format(layer_id_i, group_name_i)
        # Otherwise, no change.
        else:
            lr_scale_i = 1.
        ## Update the parameter configuration groups.
        # If not exists, create a new configuration group.
        if group_name_i not in param_groups_name:
            param_groups_name[group_name_i] = {
                "weight_decay": weight_decay_i, "lr_scale": lr_scale_i, "params": [],
            }
            param_groups_var[group_name_i] = {
                "weight_decay": weight_decay_i, "lr_scale": lr_scale_i, "params": [],
            }
        # Append to the parameter configuration groups.
        param_groups_name[group_name_i]["params"].append(name_i); param_groups_var[group_name_i]["params"].append(param_i)
    # Log information related to the parameter configuration groups.
    print((
        "INFO: Create parameter configuration groups ({}) in utils.model.torch.optim_factory."
    ).format(json.dumps(param_groups_name, indent=2)))
    # Return the final `param_groups`.
    return list(param_groups_var.values())

if __name__ == "__main__":
    import torch.nn as nn
    # local dep
    from utils import DotDict

    # Initialize the configuration of optimizer.
    optim_cfg = DotDict({
        # The name of optimizer.
        "name": "adamw",
        # The learning rate of optimizer.
        "lr": 3e-4,
        # The weight decay (i.e., L2 penalty).
        "weight_decay": 0.01,
    })
    # Instantiate the demo model.
    model = nn.Linear(
        # Modified `Linear` layer parameters.
        in_features=128, out_features=128,
        # Default `Linear` layer parameters.
        bias=True, device=None, dtype=None
    )
    # Instantiate the optimizer.
    optimizer = create_optimizer(cfg=optim_cfg, model=model, nodecay_names=None, get_layer_id=None, get_layer_scale=None)

