#!/usr/bin/env python3
"""
Created on 14:21, Jun. 8th, 2023

@author: Norbert Zheng
"""
import os, re
import numpy as np
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir, os.pardir, os.pardir))
from utils import DotDict
from utils.data import load_pickle

__all__ = [
    # Bi-polar Functions.
    "load_run_bipolar_default",
    "load_subj_bipolar_default",
]

"""
bipolar funcs
"""
# def load_subj_bipolar_default func
def load_subj_bipolar_default(path_subj, ch_names=None, use_align=True):
    """
    Load data from specified subject in bipolar.default mode.

    Args:
        path_subj: str - The path of specified subject.
        ch_names: list - The list of channel names to select.
        use_align: bool - The flag that indicates whether load aligned data.

    Returns:
        dataset: DotDict - The dataset loaded from specified run, containing [ch_names,X_s,X_n,y].
    """
    # Initialize the path of task.
    path_task = os.path.join(path_subj, "word-recitation")
    # Loop over all available runs to get `path_run_datasets`.
    path_run_datasets = []
    for run_i in os.listdir(path_task):
        # Initialize the path of current run and the corresponding dataset.
        path_run_i = os.path.join(path_task, run_i)
        path_run_dataset_i = os.path.join(path_run_i, "dataset.bipolar.default.{}".format(("aligned" if use_align else "unaligned")))
        # Update `path_run_datasets`.
        path_run_datasets.append(path_run_dataset_i)
    # Return the final `X` & `y`.
    return _load_subj_bipolar(path_run_datasets, ch_names=ch_names)

# def load_run_bipolar_default func
def load_run_bipolar_default(path_run, ch_names=None, use_align=True):
    """
    Load data from specified run in bipolar.default mode.

    Args:
        path_run: str - The path of specified run.
        ch_names: list - The list of channel names to select.
        use_align: bool - The flag that indicates whether load aligned data.

    Returns:
        dataset: DotDict - The dataset loaded from specified run, containing [ch_names,X_s,X_n,y].
    """
    # Initialize the path of dataset.
    path_run_dataset = os.path.join(path_run, "dataset.bipolar.default.{}".format(("aligned" if use_align else "unaligned")))
    # Return the final `X` & `y`.
    return _load_run_bipolar(path_run_dataset, ch_names=ch_names)

# def _load_subj_bipolar func
def _load_subj_bipolar(path_run_datasets, ch_names=None):
    """
    Load data from specified subject in bipolar mode.

    Args:
        path_run_datasets: list - The path list of specified subject & bipolar mode.
        ch_names: list - The list of channel names to select.

    Returns:
        dataset: DotDict - The dataset loaded from specified run, containing [ch_names,X_s,X_n,y].
    """
    # Initialize `subj` from `path_run_datasets`.
    pattern_subj = re.compile(r"{}(\d+){}".format(os.sep, os.sep))
    subj = pattern_subj.findall(path_run_datasets[0])[0]
    # Loop over all available runs to get `dataset_data` & `dataset_info`.
    dataset_data = []; dataset_info = []
    for path_run_dataset_i in path_run_datasets:
        # Initialize the path of dataset.data and the corresponding dataset.info.
        path_run_dataset_data_i = os.path.join(path_run_dataset_i, "data")
        path_run_dataset_info_i = os.path.join(path_run_dataset_i, "info")
        # Check whether dataset.data & dataset.info exists.
        if not (os.path.exists(path_run_dataset_data_i) and os.path.exists(path_run_dataset_info_i)): continue
        # Load `dataset_data_i` & `dataset_info_i` from specified run.
        dataset_data_i = load_pickle(path_run_dataset_data_i); np.random.shuffle(dataset_data_i)
        dataset_info_i = load_pickle(path_run_dataset_info_i)
        # Update `dataset_data` & `dataset_info`.
        dataset_data.append(dataset_data_i); dataset_info.append(dataset_info_i)
    # Get the common set of `ch_names`.
    ch_names = ch_names if ch_names is not None else None
    for dataset_info_i in dataset_info:
        ch_names = dataset_info_i.ch_names if ch_names is None else ch_names
        assert set(ch_names).issubset(set(dataset_info_i.ch_names))
    print((
        "INFO: Get {:d} common channels for subj ({}), including ({})."
    ).format(len(ch_names), subj, ch_names))
    # The labels may not be the same, due to bad trials. Use the union of these labels as the final label set.
    label_sets = [set([data_i.name for data_i in dataset_data_i]) for dataset_data_i in dataset_data]; labels = set()
    for label_set_i in label_sets: labels |= label_set_i
    labels = sorted(labels)
    # Select common channels to construct `X_*` & `y`.
    X_s = []; X_n = []; y = []
    for dataset_info_i, dataset_data_i in zip(dataset_info, dataset_data):
        # Use `ch_names` to get available channel indices from `dataset_info_i.ch_names`.
        ch_idxs_i = np.array([dataset_info_i.ch_names.index(ch_name_i) for ch_name_i in ch_names], dtype=np.int64)
        # Use `ch_idxs_i` to get `X_i`.
        # X_* - (n_samples, seq_len, n_channels)
        X_s_i = np.array([data_i.data_s.T for data_i in dataset_data_i], dtype=np.float32)[:,:,ch_idxs_i]
        X_n_i = np.array([data_i.data_n.T for data_i in dataset_data_i], dtype=np.float32)[:,:,ch_idxs_i]
        assert len(ch_names) == X_s_i.shape[-1] == X_n_i.shape[-1]
        # Use `labels` to get `y_i`.
        # y - (n_samples,)
        y_i = np.array([labels.index(data_i.name) for data_i in dataset_data_i], dtype=np.int64)
        # Update `X_*` & `y`.
        X_s.append(X_s_i); X_n.append(X_n_i); y.append(y_i)
    # Get the final `X_*` & `y`.
    # X_* - (n_samples, seq_len, n_channels); y - (n_samples,)
    X_s = np.concatenate(X_s, axis=0); X_n = np.concatenate(X_n, axis=0); y = np.concatenate(y, axis=0)
    # Return the final `dataset`.
    return DotDict({"ch_names":ch_names, "X_s":X_s, "X_n":X_n, "y":y,})

# def _load_run_bipolar func
def _load_run_bipolar(path_run_dataset, ch_names=None):
    """
    Load data from specified run in bipolar mode.

    Args:
        path_run_dataset: str - The path of specified run & bipolar mode.
        ch_names: list - The list of channel names to select.

    Returns:
        dataset: DotDict - The dataset loaded from specified run, containing [ch_names,X_s,X_n,y].
    """
    # Initialize `subj` from `path_run_dataset`.
    pattern_subj = re.compile(r"{}(\d+){}".format(os.sep, os.sep))
    subj = pattern_subj.findall(path_run_dataset)[0]
    # Initialize the path of dataset.data and the corresponding dataset.info.
    path_run_dataset_data = os.path.join(path_run_dataset, "data")
    path_run_dataset_info = os.path.join(path_run_dataset, "info")
    # Check whether dataset.data & dataset.info exists.
    if not (os.path.exists(path_run_dataset_data) and os.path.exists(path_run_dataset_info)): return None
    # Load `dataset_data` & `dataset_info` from specified run.
    dataset_data = load_pickle(path_run_dataset_data); np.random.shuffle(dataset_data)
    dataset_info = load_pickle(path_run_dataset_info)
    # Get the common set of `ch_names`.
    ch_names = ch_names if ch_names is not None else dataset_info.ch_names
    assert set(ch_names).issubset(set(dataset_info.ch_names))
    print((
        "INFO: Get {:d} common channels for subj ({}), including ({})."
    ).format(len(ch_names), subj, ch_names))
    # Get the corresponding label set.
    labels = [data_i.name for data_i in dataset_data]; labels = sorted(set(labels))
    # Use `ch_names` to get available channel indices from `dataset_info.ch_names`.
    ch_idxs = np.array([dataset_info.ch_names.index(ch_name_i) for ch_name_i in ch_names], dtype=np.int64)
    # Use `ch_idxs` to get `X_*`.
    # X_* - (n_samples, seq_len, n_channels)
    X_s = np.stack([data_i.data_s.T for data_i in dataset_data], axis=0)[:,:,ch_idxs]
    X_n = np.stack([data_i.data_n.T for data_i in dataset_data], axis=0)[:,:,ch_idxs]
    assert len(ch_names) == X_s.shape[-1] == X_n.shape[-1]
    # Use `labels` to get `y`.
    # y - (n_samples,)
    y = np.array([labels.index(data_i.name) for data_i in dataset_data], dtype=np.int64)
    # Return the final `dataset`.
    return DotDict({"ch_names":ch_names, "X_s":X_s, "X_n":X_n, "y":y,})

if __name__ == "__main__":
    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir, os.pardir, os.pardir)
    path_subj = os.path.join(base, "data", "seeg.he2023xuanwu", "028")
    path_run = os.path.join(path_subj, "word-recitation", "run1")
    ch_names = ["TI'1", "TI'2", "TI'3", "TI'4", "TI'6", "TI'7", "TI'8", "TI'9"]

    # Load data from specified run.
    dataset = load_run_bipolar_default(path_run, ch_names=ch_names)
    dataset = load_subj_bipolar_default(path_subj, ch_names=ch_names)

