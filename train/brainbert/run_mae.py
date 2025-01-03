#!/usr/bin/env python3
"""
Created on 18:02, Mar. 26th, 2024

@author: Norbert Zheng
"""
import torch
import os, time
import argparse
import copy as cp
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
    from mask import *
    from wav2spec import *
else:
    from train.brainbert.mask import *
    from train.brainbert.wav2spec import *
import utils; import utils.model.torch
from utils.data import save_pickle, load_pickle
from models.brainbert import brainbert_mae as brainbert_model

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
    Initialize `brainbert_mae` training variables.

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
            "ERROR: Unknown dataset type {} in train.brainbert.run_mae."
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
    # Initialize the configuration of data items.
    data_items = []
    for subj_idx, subj_cfg_i in enumerate(subjs_cfg):
        # Initialize `subj_i` & `path_subj_i` & `ch_names_i` from `subj_cfg_i`.
        subj_i = subj_cfg_i.name; path_subj_i = subj_cfg_i.path; ch_names_i = subj_cfg_i.ch_names
        # Construct the path of pretrain dataset.
        path_dataset_i = os.path.join(path_subj_i, "pretrain", "dataset.default.1000hz")
        # Load dataset.info to get the specified channel indices.
        info_i = load_pickle(os.path.join(path_dataset_i, "info"))
        ch_idxs_i = np.array([info_i.ch_names.index(ch_name_i) for ch_name_i in ch_names_i], dtype=np.int64)\
            if ch_names_i is not None else np.arange(len(info_i.ch_names), dtype=np.int64)
        # Loop over all task-runs to add data items.
        for root_i, dirs_i, files_i in os.walk(path_dataset_i):
            for file_i in [file_i for file_i in files_i if file_i.endswith(".npy")]:
                data_items.append(utils.DotDict({
                    "subj_idx": subj_idx, "ch_idxs": ch_idxs_i, "sfreq": 1000,
                    "freq_len": params.model.freq_len, "path": os.path.join(root_i, file_i),
                }))
    # Split dataset according to split ratio.
    train_ratio = params.train.train_ratio; validation_ratio = test_ratio = (1. - train_ratio) / 2.
    split_ratio = (train_ratio, validation_ratio, test_ratio)
    data_splits = split_dataset(data_items, split_ratio=split_ratio)
    train_splits, validation_splits, test_splits = data_splits
    # Construct train & validation & test dataset.
    dataset_train = sEEGPretrainDataset(train_splits); dataset_test = sEEGPretrainDataset(test_splits)
    dataset_validation = sEEGPretrainDataset(validation_splits)
    assert dataset_train.n_channels == dataset_validation.n_channels == dataset_test.n_channels
    assert dataset_train.freq_len == dataset_validation.freq_len == dataset_test.freq_len
    assert dataset_train.n_freqs == dataset_validation.n_freqs == dataset_test.n_freqs
    n_channels = dataset_train.n_channels; freq_len = dataset_train.freq_len; n_freqs = dataset_train.n_freqs
    # Check whether train-set & validation-set & test-set exists.
    msg = (
        "INFO: Data preparation complete, with train-set ({:d} items) & validation-set ({:d} items) & test-set ({:d} items)."
    ).format(len(dataset_train), len(dataset_validation), len(dataset_test)); log_info(msg)
    # Update related hyper-parameters in `params`.
    params.model.n_channels = n_channels
    params.model.encoder.emb_len = params.model.tokenizer.token_len = params.model.freq_len = freq_len
    params.model.rgs.n_freqs = params.model.tokenizer.n_freqs = params.model.n_freqs = n_freqs
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

# def sEEGPretrainDataset class
class sEEGPretrainDataset(torch.utils.data.Dataset):
    """
    Brain signal pretrain dataset.
    """

    def __init__(self, data_items, **kwargs):
        """
        Initialize `sEEGPretrainDataset` object.

        Args:
            path_subjs: list - The path of datasets corresponding to each subject.
            kwargs: dict - The arguments related to initialize `torch.utils.data.Dataset`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `torch.utils.data.Dataset`
        # style model and inherit it's functionality.
        super(sEEGPretrainDataset, self).__init__(**kwargs)

        # Initialize parameters.
        self.data_items = data_items

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
        # Initialize `n_subjects` & `n_channels` & `freq_len`.
        subj_idxs = np.array([data_item_i.subj_idx for data_item_i in self.data_items], dtype=np.int64)
        self.n_subjects = len(set(subj_idxs)); self.n_channels = None; self.freq_len = None
        for subj_idx in range(self.n_subjects):
            # Load one sample data item to check `n_channels` & `freq_len`.
            # X_*_i - (n_channels_subj, seq_len)
            data_item_i = self.data_items[np.where(subj_idxs==subj_idx)[0][0]]
            X_s_i, X_n_i = np.load(data_item_i.path); assert X_s_i.shape == X_n_i.shape; X_i = X_s_i
            # Select the specific channels from `X_*_i` according to its `ch_idxs`.
            X_i = X_i[data_item_i.ch_idxs,:]
            # Update `n_channels` & `freq_len` according to the shape of `X_i`.
            self.n_channels = X_i.shape[0] if self.n_channels is None else min(X_i.shape[0], self.n_channels)
            self.freq_len = data_item_i.freq_len if self.freq_len is None else self.freq_len
            assert self.freq_len == data_item_i.freq_len; self.n_freqs = 40

    """
    preprocess funcs
    """
    # def stft func
    def stft(self, X, sfreq):
        """
        The whole pipeline to get the corresponding STFT spectrum.

        Args:
            X: (n_channels, seq_len) - The raw time series.
            sfreq: int - The sampling rate.

        Returns:
            S: (n_channels, freq_len, n_freqs) - The spectrum series.
        """
        # Execute stft filter to get the spectrum series.
        # S - (batch_size, freq_len, n_freqs)
        S = np.transpose(stft(X=X, sfreq=sfreq, norm_type="zscore", n_sfreqs=40,
            nperseg=200, noverlap=175, return_onesided=True), axes=[0,2,1])
        # Return the final `S`.
        return S

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
        X_s, X_n = np.load(data_item.path); assert X_s.shape == X_n.shape; X = X_s
        # Select the specific channels from `X` according to its `ch_idxs`.
        X = X[data_item.ch_idxs,:]
        # Convert the raw time series to the spectrum series.
        # S - (n_channels_subj, freq_len, n_freqs)
        S = self.stft(X=X, sfreq=data_item.sfreq)
        # Randomly select the start index from `S.shape[-2] - freq_len`.
        shift = np.random.randint((S.shape[-2] - self.freq_len)); S = S[:,shift:(shift+self.freq_len)]
        # Randomly select the channel indices to have `n_channels` channels.
        # S - (n_channels, freq_len, n_freqs)
        ch_idxs = np.random.permutation(S.shape[0])[:self.n_channels]; S = S[ch_idxs,...]
        # Mask data according using random mask strategy.
        # S_masked - (n_channels, freq_len, n_freqs); mask - (n_channels, freq_len, n_freqs)
        S_masked = []; mask = []
        for ch_idx, S_i in enumerate(S):
            S_masked_i, mask_i = mask_data(data=S_i, mask_params=default_mask_params.random)
            S_masked.append(S_masked_i); mask.append(mask_i)
        S_masked = np.stack(S_masked, axis=0); mask = np.stack(mask, axis=0)
        # Construct the final data dict.
        data = {
            "S": torch.from_numpy(S).to(dtype=torch.float32),
            "mask": torch.from_numpy(mask).to(dtype=torch.bool),
            "S_masked": torch.from_numpy(S_masked).to(dtype=torch.float32),
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
    msg = "INFO: The parameters is initialized as {}.".format(params); log_info(msg)
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
        raise ValueError("ERROR: Unknown dataset {} in train.brainbert.run_mae.".format(params.train.dataset))
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

        # Reset the iteration information of params.
        params.iteration(iteration=0)
        if paths is not None: save_pickle(os.path.join(paths.run.save, "params"), utils.DotDict(params))
        # Initialize model of current time segment.
        model = brainbert_model(params.model).to(device=params.model.device)
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
                    train_batch["S"].to(device=params.model.device),
                    train_batch["mask"].to(device=params.model.device),
                    train_batch["S_masked"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Train model for current batch.
                S_reconstr_i, loss_i = _train(batch_i)
                # Numpy the outputs of current batch.
                S_reconstr_i = S_reconstr_i.detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_train, key_i):
                        loss_train[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_train[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
                reconstr_train.append(np.stack([
                    batch_i[0].detach().cpu().numpy(), batch_i[1].detach().cpu().numpy(), S_reconstr_i,
                ], axis=0))
            # Record information related to train process.
            for key_i, item_i in loss_train.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_train[key_i] = item_i
            reconstr_train = np.concatenate(reconstr_train, axis=1)
            reconstr_train = [(
                reconstr_train[0,sample_idx,channel_idx,...],
                reconstr_train[1,sample_idx,channel_idx,...],
                reconstr_train[2,sample_idx,channel_idx,...],
            ) for sample_idx, channel_idx in zip(
                np.random.randint(low=0, high=reconstr_train.shape[1], size=(params.train.n_samples,)),
                np.random.randint(low=0, high=reconstr_train.shape[2], size=(params.train.n_samples,)),
            )]
            # Prepare for model validation process.
            loss_validation = utils.DotDict(); reconstr_validation = []
            # Execute validation process.
            for validation_batch in dataset_validation:
                # Initialize `batch_i` from `validation_batch`.
                batch_i = [
                    validation_batch["S"].to(device=params.model.device),
                    validation_batch["mask"].to(device=params.model.device),
                    validation_batch["S_masked"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Validate model for current batch.
                S_reconstr_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                S_reconstr_i = S_reconstr_i.detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_validation, key_i):
                        loss_validation[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_validation[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
                reconstr_validation.append(np.stack([
                    batch_i[0].detach().cpu().numpy(), batch_i[1].detach().cpu().numpy(), S_reconstr_i
                ], axis=0))
            # Record information related to validation process.
            for key_i, item_i in loss_validation.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_validation[key_i] = item_i
            reconstr_validation = np.concatenate(reconstr_validation, axis=1)
            reconstr_validation = [(
                reconstr_validation[0,sample_idx,channel_idx,...],
                reconstr_validation[1,sample_idx,channel_idx,...],
                reconstr_validation[2,sample_idx,channel_idx,...],
            ) for sample_idx, channel_idx in zip(
                np.random.randint(low=0, high=reconstr_validation.shape[1], size=(params.train.n_samples,)),
                np.random.randint(low=0, high=reconstr_validation.shape[2], size=(params.train.n_samples,)),
            )]
            # Prepare for model test process.
            loss_test = utils.DotDict(); reconstr_test = []
            # Execute test process.
            for test_batch in dataset_test:
                # Initialize `batch_i` from `test_batch`.
                batch_i = [
                    test_batch["S"].to(device=params.model.device),
                    test_batch["mask"].to(device=params.model.device),
                    test_batch["S_masked"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Test model for current batch.
                S_reconstr_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                S_reconstr_i = S_reconstr_i.detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_test, key_i):
                        loss_test[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_test[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
                reconstr_test.append(np.stack([
                    batch_i[0].detach().cpu().numpy(), batch_i[1].detach().cpu().numpy(), S_reconstr_i
                ], axis=0))
            # Record information related to test process.
            for key_i, item_i in loss_test.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_test[key_i] = item_i
            reconstr_test = np.concatenate(reconstr_test, axis=1)
            reconstr_test = [(
                reconstr_test[0,sample_idx,channel_idx,...],
                reconstr_test[1,sample_idx,channel_idx,...],
                reconstr_test[2,sample_idx,channel_idx,...],
            ) for sample_idx, channel_idx in zip(
                np.random.randint(low=0, high=reconstr_test.shape[1], size=(params.train.n_samples,)),
                np.random.randint(low=0, high=reconstr_test.shape[2], size=(params.train.n_samples,)),
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
        inputs: tuple - The input data, including [S,mask,S_masked].

    Returns:
        S_reconstr: (batch_size, n_channels, freq_len, n_freqs) - The reconstructed spectrum series.
        loss: DotDict - The loss dictionary.
    """
    global model; model.eval()
    with torch.no_grad(): return model(inputs)

# def _train func
def _train(inputs):
    """
    Train the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including [S,mask,S_masked].

    Returns:
        S_reconstr: (batch_size, n_channels, freq_len, n_freqs) - The reconstructed spectrum series.
        loss: DotDict - The loss dictionary.
    """
    global model, optimizer; model.train()
    # Forward model to get the corresponding loss.
    S_reconstr, loss = model(inputs)
    # Use optimizer to update parameters.
    optimizer.zero_grad(); loss["total"].backward(); optimizer.step()
    # Return the final `S_reconstr` & `loss`.
    return S_reconstr, loss

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
def _plot_reconstr(Ss, img_fname=None):
    """
    Plot the reconstruction ND-data along with the original ND-data.

    Args:
        Ss: (n_samples[list],) - The list of reconstruction tuples, each of which contains [S,mask,S_reconstr].
        img_fname: str - The file name to save image.

    Returns:
        None
    """
    # Initialize `n_samples` & `n_types` from `Ss`.
    n_samples = len(Ss); assert n_samples > 0
    n_types = len(Ss[0]); assert n_types == 3
    # Initialize figure to prepare plot.
    unit_width = 10.; unit_height = 3.
    fig, axes = plt.subplots(n_samples, n_types, figsize=(n_types * unit_width, n_samples * unit_height)); fig.tight_layout()
    axes[0,0].set_title("Raw Spectrum Series"); axes[0,1].set_title("Masked Spectrum Series")
    axes[0,2].set_title("Reconstructed Spectrum Series")
    # Loop over all samples to plot reconstruction figures.
    for sample_idx, (ax_i, Ss_i) in enumerate(zip(axes, Ss)):
        ## Prepare data for plotting.
        # Initialize `S_i` & `mask_i` & `S_reconstr_i` from `Ss_i`.
        # S_*_i - (freq_len, n_freqs)
        S_i = Ss_i[0]; mask_i = Ss_i[1]; S_reconstr_i = Ss_i[2]
        ## Plot reconstruction figure.
        # Plot `Ground-Truth` figure.
        ax_i[0].imshow(S_i, cmap="viridis", origin="lower", aspect="auto")
        # Plot `Mask` figure.
        ax_i[1].imshow(S_i, cmap="viridis", origin="lower", aspect="auto")
        row_idxs, col_idxs = np.where(mask_i > 0.)
        for row_idx, col_idx in zip(row_idxs, col_idxs):
            ax_i[1].add_patch(plt.Rectangle(xy=((col_idx-0.5), (row_idx-0.5)),
                width=1., height=1., color="red"))
        # Plot `Reconstruction` figure.
        ax_i[2].imshow(S_reconstr_i, cmap="viridis", origin="lower", aspect="auto")
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
    parser = argparse.ArgumentParser("BrainBERT MAE for brain signals", add_help=False)
    # Add training parmaeters.
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subjs", type=str, nargs="+", default=["023",])
    # Add distributed training parameters.
    parser.add_argument("--local_rank", type=int)
    # Return the nal `parser`.
    return parser

if __name__ == "__main__":
    import os
    # local dep
    from params.brainbert_params import brainbert_mae_params as brainbert_params

    # Initialize macros.
    dataset = "seeg_he2023xuanwu"

    # Initialize arguments parser.
    args_parser = get_args_parser(); args = args_parser.parse_args()
    # Initialize random seed according to arguments.
    seed = args.seed; utils.model.torch.set_seeds(seed)

    # Initialize base path.
    base = os.path.join(os.getcwd(), os.pardir, os.pardir)
    # Initialize brainbert_params.
    brainbert_params_inst = brainbert_params(dataset=dataset)
    brainbert_params_inst.train.base = base; brainbert_params_inst.train.subjs = args.subjs
    # Train brainbert.
    train(brainbert_params_inst)
    
