#!/usr/bin/env python3
"""
Created on 00:44, Mar. 6th, 2024

@author: Norbert Zheng
"""
import os, time
import re, torch
import argparse
import copy as cp
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
import utils; import utils.model.torch
from utils.data import save_pickle, load_pickle
from models.brant import brant_mae as brant_model

__all__ = [
    "train",
]

# Global variables.
params = None; paths = None
model = None; optimizer = None

"""
init funcs
"""
# def init func
def init(params_):
    """
    Initialize `brant_mae` training variables.

    Args:
        params_: DotDict - The parameters of current training process.

    Returns:
        None
    """
    global params, paths
    # Initialize params.
    params = cp.deepcopy(params_)
    # Check whether environment has `LOCAL_RANK` macro.
    if "LOCAL_RANK" in os.environ.keys(): params.train.local_rank = int(os.environ["LOCAL_RANK"])
    # If the number of available GPUs is greater than 1, enable distributed training.
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(params.train.local_rank); torch.distributed.init_process_group(backend="nccl")
    # Initialize `device` according to `params.train.local_rank`.
    params.model.device = torch.device("cuda:{:d}".format(params.train.local_rank))\
        if torch.cuda.is_available() else torch.device("cpu"); log_info(params.model.device)
    # Initialize paths.
    paths = utils.Paths(base=params.train.base, params=params)\
        if (params.train.local_rank == 0) and (params.model.device == torch.device("cuda:0")) else None
    if paths is not None: paths.run.logger.tensorboard = SummaryWriter(paths.run.train)
    # Initialize model.
    _init_model()
    # Initialize training process.
    _init_train()

# def _init_model func
def _init_model():
    """
    Initialize model used in the training process.

    Args:
        None

    Returns:
        None
    """
    global params
    ## Initialize torch configuration.
    # Not set random seed, should be done before initializing `model`.
    torch.set_default_dtype(getattr(torch, params._precision))
    # Set the internal precision of float32 matrix multiplications.
    torch.set_float32_matmul_precision("high")

# def _init_train func
def _init_train():
    """
    Initialize the training process.

    Args:
        None

    Returns:
        None
    """
    pass

"""
data funcs
"""
# def load_data func
def load_data(load_params):
    """
    Load data from specified dataset.

    Args:
        load_params: DotDict - The load parameters of specified dataset.

    Returns:
        dataset_train: torch.utils.data.DataLoader - The train dataset, including (*_train).
        dataset_validation: torch.utils.data.DataLoader - The validation dataset, including (*_validation).
        dataset_test: torch.utils.data.DataLoader - The test dataset, including (*_test).
    """
    global params
    # Load data from specified dataset.
    try:
        func = getattr(sys.modules[__name__], "_".join(["_load_data", params.train.dataset]))
        dataset_train, dataset_validation, dataset_test = func(load_params)
    except Exception:
        raise ValueError((
            "ERROR: Unknown dataset type {} in train.brant.brant_mae."
        ).format(params.train.dataset))
    # Return the final `dataset_train` & `dataset_validation` & `dataset_test`.
    return dataset_train, dataset_validation, dataset_test

# def _load_data_seeg_he2023xuanwu func
def _load_data_seeg_he2023xuanwu(load_params):
    """
    Load seeg data from the specified subject in `seeg_he2023xuanwu`.

    Args:
        load_params: DotDict - The load parameters of specified dataset.

    Returns:
        dataset_train: torch.utils.data.DataLoader - The train dataset, including (*_train).
        dataset_validation: torch.utils.data.DataLoader - The validation dataset, including (*_validation).
        dataset_test: torch.utils.data.DataLoader - The test dataset, including (*_test).
    """
    global params, paths
    # Initialize subjs_cfg.
    subjs_cfg = load_params.subjs_cfg
    # Initialize the pattern to extract channel group.
    pattern_group = re.compile(r"([^\d]+)\d+")
    # Initialize the configuration of data items.
    data_items = []
    for subj_idx, subj_cfg_i in enumerate(subjs_cfg):
        # Initialize `subj_i` & `path_subj_i` from `subj_cfg_i`.
        subj_i = subj_cfg_i.name; path_subj_i = subj_cfg_i.path
        # Construct the path of pretrain dataset.
        path_dataset_i = os.path.join(path_subj_i, "pretrain", "dataset.default.200hz")
        # Load dataset.info to get the channel group indices.
        info_i = load_pickle(os.path.join(path_dataset_i, "info")); ch_groups_i = utils.DotDict()
        for ch_idx, ch_name_i in enumerate(info_i.ch_names):
            # Use `pattern_group` to extract `ch_group_i`.
            ch_group_i = pattern_group.findall(ch_name_i)[0]
            # If `ch_groups_i` doesn't have key `ch_group_i`, create a new list.
            if ch_group_i not in ch_groups_i.keys():
                ch_groups_i[ch_group_i] = [ch_idx,]
            # If `ch_groups_i` has attribute `ch_group_i`, append it to that list.
            else:
                ch_groups_i[ch_group_i].append(ch_idx)
        ch_groups_i = [np.array(ch_group_i, dtype=np.int64) for ch_group_i in ch_groups_i.values()]
        # Loop over all task-runs to add data items.
        for root_i, dirs_i, files_i in os.walk(path_dataset_i):
            for file_i in [file_i for file_i in files_i if file_i.endswith(".npy")]:
                data_items.append(utils.DotDict({
                    "subj_idx": subj_idx, "ch_groups": ch_groups_i, "path": os.path.join(root_i, file_i),
                }))
    # Initialize dataset configuration.
    dataset_cfg = utils.DotDict({
        "n_subjects": len(subjs_cfg), "n_channels": params.model.n_channels,
        "seq_len": params.model.seq_len, "seg_len": params.model.seg_len,
        "sfreq": 200, "mask_ratio": params.model.mask_ratio,
    })
    # Split dataset according to split ratio.
    train_ratio = params.train.train_ratio; validation_ratio = test_ratio = (1. - train_ratio) / 2.
    split_ratio = (train_ratio, validation_ratio, test_ratio)
    data_splits = split_dataset(data_items, split_ratio=split_ratio)
    train_splits, validation_splits, test_splits = data_splits
    # Construct train & validation & test dataset.
    dataset_train = PretrainDataset(cfg=dataset_cfg, data_items=train_splits)
    dataset_validation = PretrainDataset(cfg=dataset_cfg, data_items=validation_splits)
    dataset_test = PretrainDataset(cfg=dataset_cfg, data_items=test_splits)
    n_subjects = dataset_cfg.n_subjects; n_channels = dataset_cfg.n_channels; seq_len = dataset_cfg.seq_len
    # Check whether train-set & validation-set & test-set exists.
    msg = (
        "INFO: Data preparation complete, with train-set ({:d} items) & validation-set ({:d} items) & test-set ({:d} items)."
    ).format(len(dataset_train), len(dataset_validation), len(dataset_test)); log_info(msg)
    # Update related hyper-parameters in `params`.
    params.model.n_subjects = n_subjects; params.model.encoder.n_channels = params.model.n_channels = n_channels
    params.model.seq_len = seq_len; params.model.encoder.max_segs = params.model.seq_len // params.model.seg_len
    params.model.encoder.emb_len = (
        params.model.encoder.n_channels * params.model.encoder.max_segs
    )
    # Return the final `dataset_train` & `dataset_validation` & `dataset_test`.
    return dataset_train, dataset_validation, dataset_test

"""
dataset funcs
"""
# def split_dataset func
def split_dataset(data_items, split_ratio=(0.8, 0.1, 0.1)):
    """
    Split dataset according to split ratio.

    Args:
        data_items: list - The list of data items, each of which contains [path,subj_idx].
        split_ratio: list - The list of split ratio.

    Returns:
        data_splits: list - The list of splited data items.
    """
    # Normalize `split_ratio` for future split.
    split_ratio = np.array(split_ratio, dtype=np.float32); split_ratio /= np.sum(split_ratio)
    # Initialize `data_splits` according to `split_ratio`.
    data_splits = [[] for _ in range(len(split_ratio))]
    # Split `data_items` according to subject index.
    subj_idxs = np.array([data_item_i.subj_idx for data_item_i in data_items], dtype=np.int64); n_subjects = len(set(subj_idxs))
    for subj_idx in range(n_subjects):
        # Initialize the indices of data, then split.
        data_idxs = np.where(subj_idxs == subj_idx)[0]; np.random.shuffle(data_idxs)
        split_pos = np.concatenate([np.zeros((1,), dtype=np.int64),
            np.round(np.cumsum(split_ratio) * len(data_idxs)).astype(np.int64)
        ], axis=-1); assert len(data_idxs) == split_pos[-1]
        # Update `data_splits` according to `split_pos`.
        for split_idx in range(len(data_splits)):
            data_splits[split_idx].extend([data_items[data_idx]\
                for data_idx in data_idxs[split_pos[split_idx]:split_pos[split_idx+1]]])
    # Return the final `data_splits`.
    return data_splits

# def PretrainDataset class
class PretrainDataset(torch.utils.data.Dataset):
    """
    Brain signal pretrain dataset.
    """

    def __init__(self, cfg, data_items, **kwargs):
        """
        Initialize `PretrainDataset` object.

        Args:
            cfg: DotDict - The configuration of pretrain dataset.
            data_items: list - The list of data items, each of which contains [path,subj_idx].
            kwargs: dict - The arguments related to initialize `torch.utils.data.Dataset`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `torch.utils.data.Dataset`
        # style model and inherit it's functionality.
        super(PretrainDataset, self).__init__(**kwargs)

        # Initialize parameters.
        self.cfg = cfg; self.data_items = data_items

        # Initialize variables.
        self._init_dataset()

    """
    init funcs
    """
    # def _init_dataset func
    def _init_dataset(self):
        """
        Initialize the configuration of dataset.

        Args:
            None

        Returns:
            None
        """
        pass

    """
    tool funcs
    """
    # def _cal_power func
    def _cal_power(self, X, sfreq=250):
        """
        Calculate the band-wise power spectral density.

        Args:
            X: (n_channels, n_segs, seg_len) - The raw time series.
            sfreq: int - The sampling frequency of time series.

        Returns:
            S: (n_channels, n_segs, n_bands) - The band-wise power spectral density.
        """
        # Initialize default band splits.
        # freq_splits - ((n_bands + 1),)
        freq_splits = [4, 8, 13, 30, 50, 70, 90, 110, 128]
        # Calculate the freq-wise power spectral density.
        # freqs - (n_freqs,); Pxx - (n_channels, n_segs, n_freqs); `n_freqs = ((X.shape[-1] // 2) + 1)`
        freqs, Pxx = sp.signal.periodogram(X, fs=sfreq, axis=-1)
        # Get the frequency indices of each frequency band.
        band_idxs = [np.intersect1d(
            np.where(freq_splits[split_idx] < freqs)[0],
            np.where(freq_splits[split_idx + 1] >= freqs)[0]
        ) for split_idx in range(len(freq_splits) - 1)]
        # Calculate the band-wise power spectral density.
        # S - (n_channels, n_segs, n_bands)
        S_shape = Pxx.shape[:-1] + (len(band_idxs),)
        S = np.reshape(np.concatenate([np.log10(
            np.sum(np.reshape(Pxx, newshape=(-1, Pxx.shape[-1]))[:,band_idxs_i], axis=-1) + 1
        )[:,np.newaxis] for band_idxs_i in band_idxs], axis=-1), newshape=S_shape)
        # Return the final `S`.
        return S

    # def _gen_mask func
    def _gen_mask(self, X, mask_ratio=0.4):
        """
        Generate mask for embedding sequence.

        Args:
            X: (n_channels, n_segs, seg_len) - The raw time series.
            mask_ratio: float - The mask ratio of each segment.

        Returns:
            mask: (n_channels, n_segs) - The generated mask.
        """
        # Initialize `emb_len` from `X`.
        emb_len = np.prod(X.shape[:-1])
        # Initialize the length of keep embedding items.
        keep_len = int(emb_len * (1. - mask_ratio))
        # Initialize the noise for further argsort.
        # noise - (emb_len,)
        noise = np.random.rand(emb_len).astype(dtype=np.float32)
        # Get the corresponding `shuffle_idxs` & `restore_idxs`.
        # Note: `np.argsort` is reversible, we have `shuffle_idxs = np.argsort(restore_idxs)`.
        shuffle_idxs = np.argsort(noise, axis=-1); restore_idxs = np.argsort(shuffle_idxs, axis=-1)
        # Generate the bool mask: `False` is keep, `True` is remove.
        # mask - (emb_len,)
        mask = np.ones((emb_len,), dtype=np.bool_); mask[:keep_len] = False
        # Un-shuffle to get the bool mask.
        mask = mask[restore_idxs]
        # Reshape `mask` to have the similar shape with `X`.
        # mask - (n_channels, n_segs)
        mask = np.reshape(mask, newshape=X.shape[:-1])
        # Return the final `mask`.
        return mask

    """
    dataset funcs
    """
    # def __len__ func
    def __len__(self):
        """
        Get the number of samples of dataset.

        Args:
            None

        Returns:
            n_samples: int - The number of samples of dataset.
        """
        return len(self.data_items)

    # def __getitem__ func
    def __getitem__(self, index):
        """
        Get the data item corresponding to data index.

        Args:
            index: int - The index of data item to get.

        Returns:
            data: dict - The data item dictionary.
        """
        # Initialize `data_item` according to `index`.
        data_item = self.data_items[index]
        # Load data item from `data_item`.
        # X_* - (n_channels_subj, seq_len)
        X_s, X_n = np.load(data_item.path); assert X_s.shape == X_n.shape
        # Randomly select channels from `X_*_i` according to its `ch_groups`.
        group_idxs = np.random.permutation(len(data_item.ch_groups)); ch_idxs = np.array([], dtype=np.int64)
        for group_idx in group_idxs:
            ch_idxs = np.append(ch_idxs, data_item.ch_groups[group_idx])
            if len(ch_idxs) >= self.cfg.n_channels: break
        keep_idxs = np.random.permutation(len(ch_idxs))[:self.cfg.n_channels]; ch_idxs = ch_idxs[keep_idxs]
        # X_* - (n_channels, seq_len)
        X_s = X_s[ch_idxs,:]; X_n = X_n[ch_idxs,:]
        # Randomly select the start index from `X_s.shape[1] - seq_len`.
        shift_i = np.random.randint((X_s.shape[-1] - self.cfg.seq_len)) if (X_s.shape[-1] > self.cfg.seq_len) else 0
        X_s = X_s[:,shift_i:(shift_i+self.cfg.seq_len)]; X_n = X_n[:,shift_i:(shift_i+self.cfg.seq_len)]
        # Add `X_s` & `X_n` to get the original `X`.
        # X - (n_channels, seq_len)
        X = (X_s + X_n) / 2.
        # Split `X` into patch segments.
        # X - (n_channels, n_segs, seg_len)
        X = np.reshape(X, newshape=(*X.shape[:-1], -1, self.cfg.seg_len))
        # Calculate the band-wise spectral density.
        # S - (n_channels, n_segs, n_bands)
        S = self._cal_power(X, sfreq=self.cfg.sfreq)
        # Generate mask according to mask ratio.
        # mask - (n_channels, n_segs)
        mask = self._gen_mask(X, mask_ratio=self.cfg.mask_ratio)
        # Construct the final data dict.
        data = {
            "X": torch.from_numpy(X).to(dtype=torch.float32),
            "S": torch.from_numpy(S).to(dtype=torch.float32),
            "mask": torch.from_numpy(mask).to(dtype=torch.bool),
        }
        # Return the final `data`.
        return data

"""
train funcs
"""
# def train func
def train(params_):
    """
    Train the model.

    Args:
        params_: DotDict - The parameters of current training process.

    Returns:
        None
    """
    global _forward, _train
    global params, paths, model, optimizer
    # Initialize parameters & variables of current training process.
    init(params_)
    # Log the start of current training process.
    msg = "Training started with dataset {}.".format(params.train.dataset); log_info(msg)
    # Initialize load_params. Each load_params_i corresponds to a sub-dataset.
    if params.train.dataset == "seeg_he2023xuanwu":
        # Initialize the configurations of subjects that we want to execute experiments.
        subjs_cfg = utils.DotDict({
            "001": utils.DotDict({
                "name": "001", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "001"),
                "ch_names": ["SM8", "SM9", "SM7", "SM11", "P4", "SM10", "SM6", "P3", "SM5", "CI9"],
            }),
            "002": utils.DotDict({
                "name": "002", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "002"),
                "ch_names": ["TI'2", "TI'3", "TI'1", "TI'6", "TI'4", "TI'7", "ST'3", "ST'2", "ST'4", "FP'4"],
            }),
            "003": utils.DotDict({
                "name": "003", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "003"),
                "ch_names": ["ST3", "ST1", "ST2", "ST9", "TI'4", "TI'3", "ST4", "TI'2", "ST7", "TI'8"] ,
            }),
            "004": utils.DotDict({
                "name": "004", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "004"),
                "ch_names": ["D12", "D13", "C4", "C3", "D11", "D14", "D10", "D9", "D5", "C15"],
            }),
            "005": utils.DotDict({
                "name": "005", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "005"),
                "ch_names": ["E8", "E9", "E6", "E7", "E11", "E12", "E5", "E10", "C10", "E4"],
            }),
            "006": utils.DotDict({
                "name": "006", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "006"),
                "ch_names": ["D3", "D1", "D6", "D2", "D5", "D4", "D7", "D8", "G8", "E13"],
            }),
            "007": utils.DotDict({
                "name": "007", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "007"),
                "ch_names": ["H2", "H4", "H3", "H1", "H6", "H5", "E4", "H7", "C13", "E5"],
            }),
            "008": utils.DotDict({
                "name": "008", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "008"),
                "ch_names": ["TI3", "TI4", "TI2", "TI5", "B9", "TI6", "TI7", "TI9", "TI10", "B5"],
            }),
            "009": utils.DotDict({
                "name": "009", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "009"),
                "ch_names": ["K9", "K8", "K6", "K7", "K11", "K10", "K5", "K4", "K3", "I9"],
            }),
            "010": utils.DotDict({
                "name": "010", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "010"),
                "ch_names": ["PI5", "PI6", "PI7", "PI8", "PI1", "PI9", "PI2", "SM2", "SP3", "PI4"],
            }),
            "011": utils.DotDict({
                "name": "011", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "011"),
                "ch_names": ["T2", "T3", "C9", "T4", "T5", "C7", "C8", "T1", "s1", "C4"],
            }),
            "012": utils.DotDict({
                "name": "012", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "012"),
                "ch_names": ["TI'4", "TI'2", "TI'3", "TI'5", "TI'8", "TI'6", "TI'7", "TO'9", "P'5", "TO'8"],
            }),
        }); subjs_cfg = [subjs_cfg[subj_i] for subj_i in params.train.subjs]
        # Log the configuration of current training process.
        msg = "INFO: The dataset configuration is initialized as {}.".format(subjs_cfg); log_info(msg)
        # `load_params` contains all the experiments that we want to execute for every run.
        load_params = [
            # train-task-all-all-test-task-all-all
            utils.DotDict({
                "name": "train-task-all-all-test-task-all-all",
            }),
        ]
    else:
        raise ValueError("ERROR: Unknown dataset {} in train.brant.run_mae.".format(params.train.dataset))
    # Loop over all the experiments.
    for load_params_idx in range(len(load_params)):
        # Add `subjs_cfg` to `load_params_i`.
        load_params_i = cp.deepcopy(load_params[load_params_idx]); load_params_i.subjs_cfg = subjs_cfg
        # Log the start of current training iteration.
        msg = (
            "Training started with experiment {} with {:d} subjects."
        ).format(load_params_i.name, len(load_params_i.subjs_cfg)); log_info(msg)
        # Load data from specified experiment.
        dataset_train, dataset_validation, dataset_test = load_data(load_params_i)
        # Construct the distributed sampler.
        sampler_train = torch.utils.data.DistributedSampler(dataset_train,
            rank=params.train.local_rank) if torch.cuda.device_count() > 1 else None
        sampler_validation = torch.utils.data.DistributedSampler(dataset_validation,
            rank=params.train.local_rank) if torch.cuda.device_count() > 1 else None
        sampler_test = torch.utils.data.DistributedSampler(dataset_test,
            rank=params.train.local_rank) if torch.cuda.device_count() > 1 else None
        # Shuffle and then batch the dataset.
        dataset_train = torch.utils.data.DataLoader(dataset_train, batch_size=params.train.batch_size,
            shuffle=(sampler_train is None), sampler=sampler_train, drop_last=False, num_workers=0, pin_memory=True)
        dataset_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=params.train.batch_size,
            shuffle=(sampler_validation is None), sampler=sampler_validation, drop_last=False, num_workers=0, pin_memory=True)
        dataset_test = torch.utils.data.DataLoader(dataset_test, batch_size=params.train.batch_size,
            shuffle=(sampler_test is None), sampler=sampler_test, drop_last=False, num_workers=0, pin_memory=True)

        # Train the model for each time segment.
        accuracies_validation = []; accuracies_test = []

        # Reset the iteration information of params.
        params.iteration(iteration=0)
        if paths is not None: save_pickle(os.path.join(paths.run.save, "params"), utils.DotDict(params))
        # Initialize model of current time segment.
        model = brant_model(params.model).to(device=params.model.device)
        if params.train.use_graph_mode: model = torch.compile(model)
        # Summarize model information.
        msg = summary(model, col_names=("num_params", "params_percent", "trainable",)); log_info(msg)
        # If the number of available devices is greater than 1, use data parallel.
        if torch.cuda.device_count() > 1:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[params.train.local_rank,],
                output_device=params.train.local_rank, find_unused_parameters=True)
        # Make an ADAM optimizer for model.
        optim_cfg = utils.DotDict({"name":"adamw","lr":params.train.lr_i,"weight_decay":0.01,})
        optimizer = utils.model.torch.create_optimizer(cfg=optim_cfg, model=model)
        for epoch_idx in range(params.train.n_epochs):
            # If the number of available devices is greater than 1, shuffle the data at every epoch.
            if sampler_train is not None: sampler_train.set_epoch(epoch_idx)
            if sampler_validation is not None: sampler_validation.set_epoch(epoch_idx)
            if sampler_test is not None: sampler_test.set_epoch(epoch_idx)
            # Update params according to `epoch_idx`, then update optimizer.lr.
            params.iteration(iteration=epoch_idx)
            for param_group_i in optimizer.param_groups: param_group_i["lr"] = params.train.lr_i
            # Record the start time of preparing data.
            time_start = time.time()
            # Prepare for model train process.
            loss_train = utils.DotDict(); reconstr_train = []
            # Execute train process.
            for train_batch in dataset_train:
                # Initialize `batch_i` from `train_batch`.
                batch_i = [
                    train_batch["X"].to(device=params.model.device),
                    train_batch["S"].to(device=params.model.device),
                    train_batch["mask"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Train model for current batch.
                X_reconstr_i, loss_i = _train(batch_i)
                # Numpy the outputs of current batch.
                X_i = batch_i[0].detach().cpu().numpy(); X_reconstr_i = X_reconstr_i.detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_train, key_i):
                        loss_train[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_train[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
                X_i = np.transpose(np.reshape(X_i, newshape=(*X_i.shape[:2], -1)), axes=[0,2,1])
                X_reconstr_i = np.transpose(np.reshape(X_reconstr_i, newshape=(*X_reconstr_i.shape[:2], -1)), axes=[0,2,1])
                reconstr_train.append(np.stack([X_i, X_reconstr_i], axis=0))
            # Record information related to train process.
            for key_i, item_i in loss_train.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_train[key_i] = item_i
            reconstr_train = np.concatenate(reconstr_train, axis=1)
            reconstr_train = [(
                reconstr_train[0,sample_idx,...,channel_idx], reconstr_train[1,sample_idx,...,channel_idx]
            ) for sample_idx, channel_idx in zip(
                np.random.randint(low=0, high=reconstr_train.shape[1], size=(params.train.n_samples,)),
                np.random.randint(low=0, high=reconstr_train.shape[-1], size=(params.train.n_samples,)),
            )]
            # Prepare for model validation process.
            loss_validation = utils.DotDict(); reconstr_validation = []
            # Execute validation process.
            for validation_batch in dataset_validation:
                # Initialize `batch_i` from `validation_batch`.
                batch_i = [
                    validation_batch["X"].to(device=params.model.device),
                    validation_batch["S"].to(device=params.model.device),
                    validation_batch["mask"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Validate model for current batch.
                X_reconstr_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                X_i = batch_i[0].detach().cpu().numpy(); X_reconstr_i = X_reconstr_i.detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_validation, key_i):
                        loss_validation[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_validation[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
                X_i = np.transpose(np.reshape(X_i, newshape=(*X_i.shape[:2], -1)), axes=[0,2,1])
                X_reconstr_i = np.transpose(np.reshape(X_reconstr_i, newshape=(*X_reconstr_i.shape[:2], -1)), axes=[0,2,1])
                reconstr_validation.append(np.stack([X_i, X_reconstr_i], axis=0))
            # Record information related to validation process.
            for key_i, item_i in loss_validation.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_validation[key_i] = item_i
            reconstr_validation = np.concatenate(reconstr_validation, axis=1)
            reconstr_validation = [(
                reconstr_validation[0,sample_idx,...,channel_idx], reconstr_validation[1,sample_idx,...,channel_idx]
            ) for sample_idx, channel_idx in zip(
                np.random.randint(low=0, high=reconstr_validation.shape[1], size=(params.train.n_samples,)),
                np.random.randint(low=0, high=reconstr_validation.shape[-1], size=(params.train.n_samples,)),
            )]
            # Prepare for model test process.
            loss_test = utils.DotDict(); reconstr_test = []
            # Execute test process.
            for test_batch in dataset_test:
                # Initialize `batch_i` from `test_batch`.
                batch_i = [
                    test_batch["X"].to(device=params.model.device),
                    test_batch["S"].to(device=params.model.device),
                    test_batch["mask"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Test model for current batch.
                X_reconstr_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                X_i = batch_i[0].detach().cpu().numpy(); X_reconstr_i = X_reconstr_i.detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_test, key_i):
                        loss_test[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_test[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
                X_i = np.transpose(np.reshape(X_i, newshape=(*X_i.shape[:2], -1)), axes=[0,2,1])
                X_reconstr_i = np.transpose(np.reshape(X_reconstr_i, newshape=(*X_reconstr_i.shape[:2], -1)), axes=[0,2,1])
                reconstr_test.append(np.stack([X_i, X_reconstr_i], axis=0))
            # Record information related to test process.
            for key_i, item_i in loss_test.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_test[key_i] = item_i
            reconstr_test = np.concatenate(reconstr_test, axis=1)
            reconstr_test = [(
                reconstr_test[0,sample_idx,...,channel_idx], reconstr_test[1,sample_idx,...,channel_idx]
            ) for sample_idx, channel_idx in zip(
                np.random.randint(low=0, high=reconstr_test.shape[1], size=(params.train.n_samples,)),
                np.random.randint(low=0, high=reconstr_test.shape[-1], size=(params.train.n_samples,)),
            )]
            ## Write progress to summaries.
            # Log information related to current training epoch.
            time_stop = time.time()
            msg = (
                "Finish train epoch {:d} in {:.2f} seconds."
            ).format(epoch_idx, time_stop-time_start); log_info(msg)
            # Log information related to train process.
            loss_keys = list(loss_train.keys())
            msg = "Loss(train): {:.5f} ({})".format(loss_train[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_train[loss_keys[loss_idx]], loss_keys[loss_idx])
            log_info(msg)
            # Log information related to validation process.
            loss_keys = list(loss_validation.keys())
            msg = "Loss(validation): {:.5f} ({})".format(loss_validation[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_validation[loss_keys[loss_idx]], loss_keys[loss_idx])
            log_info(msg)
            # Log information related to test process.
            loss_keys = list(loss_test.keys())
            msg = "Loss(test): {:.5f} ({})".format(loss_test[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_test[loss_keys[loss_idx]], loss_keys[loss_idx])
            log_info(msg)
            ## Write progress to tensorboard.
            if paths is not None:
                # Get the pointer of writer.
                writer = paths.run.logger.tensorboard
                # Log information related to train process.
                for key_i, loss_i in loss_train.items():
                    writer.add_scalar(os.path.join("losses", "train", key_i), loss_i, global_step=epoch_idx)
                # Log information related to validation process.
                for key_i, loss_i in loss_validation.items():
                    writer.add_scalar(os.path.join("losses", "validation", key_i), loss_i, global_step=epoch_idx)
                # Log information related to test process.
                for key_i, loss_i in loss_test.items():
                    writer.add_scalar(os.path.join("losses", "test", key_i), loss_i, global_step=epoch_idx)
            ## Plot reconstruction figure of current epoch.
            if paths is not None:
                _plot_reconstr(reconstr_train, img_fname=os.path.join(paths.run.save,
                    "train-reconstr-{:d}.png".format(epoch_idx)))
                _plot_reconstr(reconstr_validation, img_fname=os.path.join(paths.run.save,
                    "validation-reconstr-{:d}.png".format(epoch_idx)))
                _plot_reconstr(reconstr_test, img_fname=os.path.join(paths.run.save,
                    "test-reconstr-{:d}.png".format(epoch_idx)))
            ## Save model parameters.
            if paths is not None:
                if (epoch_idx % params.train.i_save == 0) or (epoch_idx + 1 == params.train.n_epochs):
                    torch.save(model.state_dict(), f=os.path.join(paths.run.model, "checkpoint-{:d}.pth".format(epoch_idx)))
        # Finish training process of current specified experiment.
        msg = (
            "Finish the training process of experiment {}."
        ).format(load_params_i.name); log_info(msg)
    # Finish current training process.
    if paths is not None:
        writer = paths.run.logger.tensorboard; writer.close()
    # Log the end of current training process.
    msg = "Training finished with dataset {}.".format(params.train.dataset); log_info(msg)

# def _forward func
def _forward(inputs):
    """
    Forward the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including [X,].

    Returns:
        X_reconstr: (batch_size, seq_len, n_channels) - The reconstructed signals.
        loss: DotDict - The loss dictionary.
    """
    global model; model.eval()
    with torch.no_grad(): return model(inputs)

# def _train func
def _train(inputs):
    """
    Train the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including [X,].

    Returns:
        X_reconstr: (batch_size, seq_len, n_channels) - The reconstructed signals.
        loss: DotDict - The loss dictionary.
    """
    global model, optimizer; model.train()
    # Forward model to get the corresponding loss.
    X_reconstr, loss = model(inputs)
    # Use optimizer to update parameters.
    optimizer.zero_grad(); loss["total"].backward(); optimizer.step()
    # Return the final `X_reconstr` & `loss`.
    return X_reconstr, loss

"""
log funcs
"""
# def log_info func
def log_info(msg):
    """
    Log the specified infomation message.

    Args:
        msg: str - The information message to log.

    Returns:
        None
    """
    global paths
    # Log the specified infomation message.
    if paths is not None:
        print(msg); paths.run.logger.summaries.info(msg)

"""
tool funcs
"""
# def _plot_reconstr func
def _plot_reconstr(Xs, img_fname=None):
    """
    Plot the reconstruction 1D-signal along with the original 1D-signal.

    Args:
        Xs: (n_samples[list],) - The list of reconstruction tuples, each of which contains [X,X_reconstr].
        img_fname: str - The file name to save image.

    Returns:
        None
    """
    # Initialize `n_samples` & `n_types` from `Xs`.
    n_samples = len(Xs); assert n_samples > 0
    n_types = len(Xs[0]); assert n_types == 2
    # Initialize figure to prepare plot.
    unit_width = 10.; unit_height = 3.
    fig, axes = plt.subplots(n_samples, n_types, figsize=(n_types * unit_width, n_samples * unit_height)); fig.tight_layout()
    axes[0,0].set_title("Ground-Truth"); axes[0,1].set_title("Reconstruction")
    # Loop over all samples to plot reconstruction figures.
    for sample_idx, (ax_i, Xs_i) in enumerate(zip(axes, Xs)):
        ## Prepare data for plotting.
        # Initialize `X_i` & `X_reconstr_i` from `Xs_i`.
        # X_*_i - (seq_len,)
        X_i = Xs_i[0]; X_reconstr_i = Xs_i[1]
        # Calcualate correlation between `X_i` and `X_reconstr_i`.
        corr_i = np.corrcoef([X_i, X_reconstr_i])[0,1]
        ## Plot reconstruction figure.
        # Initialize `xs_i` for plotting lines.
        xs_i = np.arange(len(X_i))
        # Plot `Ground-Truth` figure.
        ax_i[0].plot(xs_i, X_i)
        # Plot `Reconstruction` figure.
        ax_i[1].plot(xs_i, X_reconstr_i)
        ax_i[1].set_ylabel("cor: {:.4f}".format(corr_i), weight="bold"); ax_i[1].yaxis.set_label_position("right")
    # Save or show reconstruction figure.
    if img_fname is None:
        plt.show()
    else:
        fig.savefig(img_fname)
    # Close plot.
    plt.close(fig)

"""
arg funcs
"""

# def get_args_parser func
def get_args_parser():
    """
    Parse arguments from command line.

    Args:
        None

    Returns:
        parser: object - The initialized argument parser.
    """
    # Initialize parser.
    parser = argparse.ArgumentParser("Brant MAE for brain signals", add_help=False)
    # Add training parmaeters.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjs", type=str, nargs="+", default=["010",])
    # Add distributed training parameters.
    parser.add_argument("--local_rank", type=int)
    # Return the final `parser`.
    return parser

if __name__ == "__main__":
    import os
    # local dep
    from params.brant_params import brant_mae_params as brant_params

    # macro
    dataset = "seeg_he2023xuanwu"

    # Initialize arguments parser.
    args_parser = get_args_parser(); args = args_parser.parse_args()
    # Initialize random seed according to arguments.
    seed = args.seed; utils.model.torch.set_seeds(seed)

    ## Initialize brant.
    # Initialize base.
    base = os.path.join(os.getcwd(), os.pardir, os.pardir)
    # Initialize arguments parser.
    args_parser = get_args_parser(); args = args_parser.parse_args()
    # Initialize brant_params.
    brant_params_inst = brant_params(dataset=dataset)
    brant_params_inst.train.base = base; brant_params_inst.train.subjs = args.subjs
    # Train brant.
    train(brant_params_inst)

