#!/usr/bin/env python3
"""
Created on 21:12, Mar. 28th, 2024

@author: Norbert Zheng
"""
import torch
import os, time
import argparse
import copy as cp
import numpy as np
import scipy as sp
from collections import Counter
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
import utils; import utils.model.torch; import utils.data.seeg
from utils.data import load_pickle
from models.dewave import dewave as dewave_model

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
    Initialize `dewave` training variables.

    Args:
        params_: DotDict - The parameters of current training process.

    Returns:
        None
    """
    global params, paths
    # Initialize params.
    params = cp.deepcopy(params_)
    paths = utils.Paths(base=params.train.base, params=params)
    paths.run.logger.tensorboard = SummaryWriter(paths.run.train)
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
        dataset_train: torch.utils.data.DataLoader - The train dataset, including (X_train, y_train).
        dataset_validation: torch.utils.data.DataLoader - The validation dataset, including (X_validation, y_validation).
        dataset_test: torch.utils.data.DataLoader - The test dataset, including (X_test, y_test).
    """
    global params
    # Load data from specified dataset.
    try:
        func = getattr(sys.modules[__name__], "_".join(["_load_data", params.train.dataset]))
        dataset_train, dataset_validation, dataset_test = func(load_params)
    except Exception:
        raise ValueError((
            "ERROR: Unknown dataset type {} in train.dewave.run_cls."
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
        dataset_train: torch.utils.data.DataLoader - The train dataset, including (X_train, y_train).
        dataset_validation: torch.utils.data.DataLoader - The validation dataset, including (X_validation, y_validation).
        dataset_test: torch.utils.data.DataLoader - The test dataset, including (X_test, y_test).
    """
    global params, paths
    # Initialize subj_cfg.
    subj_cfg = load_params.subj_cfg
    # Load data from specified subject run.
    func = getattr(getattr(utils.data.seeg.he2023xuanwu, load_params.task), "load_subj_{}".format(load_params.type))
    dataset = func(subj_cfg.path, ch_names=subj_cfg.ch_names, use_align=load_params.use_align)
    X = dataset.X_s.astype(np.float32); y = dataset.y.astype(np.int64)
    # If the type of dataset is `bipolar`.
    if load_params.type.startswith("bipolar"):
        # Truncate `X` to let them have the same length.
        # TODO: Here, we only keep the [0.0~0.8]s-part of [audio,image] that after onset. And we should
        # note that the [0.0~0.8]s-part of image is the whole onset time of image, the [0.0~0.8]s-part
        # of audio is the sum of the whole onset time of audio and the following 0.3s padding.
        # X - (n_samples, seq_len, n_channels)
        X = X
        # Resample the original data to the specified `resample_rate`.
        sample_rate = 1000; X = sp.signal.resample(X, int(np.round(X.shape[1] /\
            (sample_rate / load_params.resample_rate))), axis=1)
        # Truncate data according to epoch range (-0.2,1.0), the original epoch range is (-0.5,2.0).
        X = X[:,int(np.round((-0.5 - (-0.5)) * load_params.resample_rate)):\
            int(np.round((2.5 - (-0.5)) * load_params.resample_rate)),:]
        # Do Z-score for each channel.
        # TODO: As we do z-score for each channel, we do not have to scale the reconstruction
        # loss by the variance of each channel. We can check `np.var(X, axis=(0,1))` is near 1.
        X = (X - np.mean(X, axis=(0,1), keepdims=True)) / np.std(X, axis=(0,1), keepdims=True)
    # Get unknown type of dataset.
    else:
        raise ValueError("ERROR: Unknown type {} of dataset.".format(load_params.type))
    # Initialize trainset & testset.
    # X - (n_samples, seq_len, n_channels); y - (n_samples,)
    train_ratio = params.train.train_ratio; train_idxs = []; test_idxs = []
    for label_i in sorted(set(y)):
        label_idxs = np.where(y == label_i)[0].tolist()
        train_idxs.extend(label_idxs[:int(train_ratio * len(label_idxs))])
        test_idxs.extend(label_idxs[int(train_ratio * len(label_idxs)):])
    for train_idx in train_idxs: assert train_idx not in test_idxs
    train_idxs = np.array(train_idxs, dtype=np.int64); test_idxs = np.array(test_idxs, dtype=np.int64)
    X_train = X[train_idxs,:,:]; y_train = y[train_idxs]; X_test = X[test_idxs,:,:]; y_test = y[test_idxs]
    # Check whether trainset & testset both have data items.
    if len(X_train) == 0 or len(X_test) == 0: return ([], []), ([], [])
    # Make sure there is no overlap between X_train & X_test.
    samples_same = None; n_samples = 10; assert X_train.shape[1] == X_test.shape[1]
    for _ in range(n_samples):
        sample_idx = np.random.randint(X_train.shape[1])
        sample_same_i = np.intersect1d(X_train[:,sample_idx,0], X_test[:,sample_idx,0], return_indices=True)[-1].tolist()
        samples_same = set(sample_same_i) if samples_same is None else set(sample_same_i) & samples_same
    assert len(samples_same) == 0
    # Check whether labels are enough, then transform y to sorted order.
    assert len(set(y_train)) == len(set(y_test)); labels = sorted(set(y_train))
    # y - (n_samples, n_labels)
    y_train = np.array([labels.index(y_i) for y_i in y_train], dtype=np.int64); y_train = np.eye(len(labels))[y_train]
    y_test = np.array([labels.index(y_i) for y_i in y_test], dtype=np.int64); y_test = np.eye(len(labels))[y_test]
    # Execute sample permutation. We only shuffle along the axis.
    if load_params.permutation: np.random.shuffle(y_train)
    # Further split test-set into validation-set & test-set.
    validation_idxs = np.random.choice(np.arange(X_test.shape[0]), size=int(X_test.shape[0]/2), replace=False)
    validation_mask = np.zeros((X_test.shape[0],), dtype=np.bool_); validation_mask[validation_idxs] = True
    X_validation = X_test[validation_mask,:,:]; y_validation = y_test[validation_mask,:]
    X_test = X_test[~validation_mask,:,:]; y_test = y_test[~validation_mask,:]
    # Log information of data loading.
    msg = (
        "INFO: Data preparation for subject ({}) complete, with train-set ({}) & validation-set ({}) & test-set ({})."
    ).format(subj_cfg.name, X_train.shape, X_validation.shape, X_test.shape)
    print(msg); paths.run.logger.summaries.info(msg)
    # Construct dataset from data items.
    dataset_train = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i,
    }) for X_i, y_i in zip(X_train, y_train)], use_aug=True)
    dataset_validation = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i,
    }) for X_i, y_i in zip(X_validation, y_validation)], use_aug=False)
    dataset_test = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i,
    }) for X_i, y_i in zip(X_test, y_test)], use_aug=False)
    # Shuffle and then batch the dataset.
    dataset_train = torch.utils.data.DataLoader(dataset_train,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    dataset_validation = torch.utils.data.DataLoader(dataset_validation,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    dataset_test = torch.utils.data.DataLoader(dataset_test,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    # Update `n_channels` & `seq_len` & `n_labels`.
    _, seq_len, n_channels = X.shape; n_labels = len(labels)
    # Update related hyper-parameters in `params`.
    params.model.tokenizer.n_channels = params.model.n_channels = n_channels
    params.model.seq_len = seq_len; token_len = seq_len
    for n_strides_i in params.model.tokenizer.n_strides: token_len //= n_strides_i
    params.model.encoder.emb_len = params.model.tokenizer.token_len = token_len
    params.model.cls.d_feature = (
        params.model.encoder.emb_len * params.model.encoder.d_model
    )
    params.model.cls.n_labels = params.model.n_labels = n_labels
    # Return the final `dataset_train` & `dataset_validation` & `dataset_test`.
    return dataset_train, dataset_validation, dataset_test

# def FinetuneDataset class
class FinetuneDataset(torch.utils.data.Dataset):
    """
    Brain signal finetune dataset.
    """

    def __init__(self, data_items, use_aug=False, **kwargs):
        """
        Initialize `FinetuneDataset` object.

        Args:
            data_items: list - The list of data items, including [X,y].
            use_aug: bool - The flag that indicates whether enable augmentations.
            kwargs: dict - The arguments related to initialize `torch.utils.data.Dataset`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `torch.utils.data.Dataset`
        # style model and inherit it's functionality.
        super(FinetuneDataset, self).__init__(**kwargs)

        # Initialize parameters.
        self.data_items = data_items; self.use_aug = use_aug

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
        # Initialize the maximum shift steps.
        self.max_steps = self.data_items[0].X.shape[1] // 10

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
        ## Load data item.
        # Initialize `data_item` according to `index`.
        data_item = self.data_items[index]
        # Load data item from `data_item`.
        # X - (n_channels, seq_len); y - (n_labels,)
        X = data_item.X; y = data_item.y
        ## Execute data augmentations.
        if self.use_aug:
            # Randomly shift `X` according to `max_steps`.
            X_shifted = np.zeros(X.shape, dtype=X.dtype)
            n_steps = np.random.choice((np.arange(2 * self.max_steps + 1, dtype=np.int64) - self.max_steps))
            if n_steps > 0:
                X_shifted[:,n_steps:] = X[:,:-n_steps]
            elif n_steps < 0:
                X_shifted[:,:n_steps] = X[:,-n_steps:]
            else:
                pass
            X = X_shifted
        ## Construct the data dict.
        # Construct the final data dict.
        data = {
            "X": torch.from_numpy(X.T).to(dtype=torch.float32),
            "y": torch.from_numpy(y).to(dtype=torch.float32),
        }
        # Return the final `data`.
        return data

"""
train funcs
"""
# def train func
def train():
    """
    Train the model.

    Args:
        None

    Returns:
        None
    """
    global _forward, _train
    global params, paths, model, optimizer
    # Log the start of current training process.
    paths.run.logger.summaries.info("Training started with dataset {}.".format(params.train.dataset))
    # Initialize model device.
    params.model.device = torch.device("cuda:{:d}".format(0)) if torch.cuda.is_available() else torch.device("cpu")
    print(params.model.device); paths.run.logger.summaries.info(params.model.device)
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
        }); load_type = "bipolar_default"; load_task = "word_recitation"; use_align = False
        # Initialize the specified available_runs according to subjs_cfg.
        subj_cfg = subjs_cfg[params.train.subj]
        # Set `resample_rate` according to `load_type`.
        if load_type.startswith("bipolar"):
            resample_rate = 500
        # `load_params` contains all the experiments that we want to execute for every run.
        load_params = [
            # train-task-all-speak-test-task-all-speak
            utils.DotDict({
                "name": "train-task-all-speak-test-task-all-speak", "type": load_type,
                "permutation": False, "resample_rate": resample_rate, "task": load_task, "use_align": use_align,
            }),
        ]
    else:
        raise ValueError("ERROR: Unknown dataset {} in train.dewave.run_cls.".format(params.train.dataset))
    # Loop over all the experiments.
    for load_params_idx in range(len(load_params)):
        # Add `subj_cfg` to `load_params_i`.
        load_params_i = cp.deepcopy(load_params[load_params_idx]); load_params_i.subj_cfg = subj_cfg
        # Log the start of current training iteration.
        msg = (
            "Training started with experiment {} with subject {}."
        ).format(load_params_i.name, load_params_i.subj_cfg.name)
        print(msg); paths.run.logger.summaries.info(msg)
        # Load data from specified experiment.
        dataset_train, dataset_validation, dataset_test = load_data(load_params_i)

        # Train the model for each time segment.
        accuracies_validation = []; accuracies_test = []

        # Reset the iteration information of params.
        params.iteration(iteration=0)
        # Initialize model of current time segment.
        model = dewave_model(params.model)
        model = model.to(device=params.model.device)
        if params.train.use_graph_mode: model = torch.compile(model)
        # Make an ADAM optimizer for model.
        optim_cfg = utils.DotDict({"name":"adamw","lr":params.train.lr_i,"weight_decay":0.05,})
        optimizer = utils.model.torch.create_optimizer(cfg=optim_cfg, model=model)
        for epoch_idx in range(params.train.n_epochs):
            # Update params according to `epoch_idx`, then update optimizer.lr.
            params.iteration(iteration=epoch_idx)
            for param_group_i in optimizer.param_groups: param_group_i["lr"] = params.train.lr_i
            # Record the start time of preparing data.
            time_start = time.time()
            # Prepare for model train process.
            accuracy_train = []; loss_train = utils.DotDict()
            # Execute train process.
            for train_batch in dataset_train:
                # Initialize `batch_i` from `train_batch`.
                batch_i = [
                    train_batch["X"].to(device=params.model.device),
                    train_batch["y"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Train model for current batch.
                y_pred_i, loss_i = _train(batch_i)
                # Numpy the outputs of current batch.
                y_pred_i = y_pred_i.detach().cpu().numpy(); y_true_i = batch_i[1].detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                accuracy_i = np.array([
                    np.mean(np.argmax(y_pred_i, axis=-1) == np.argmax(batch_i[1].detach().cpu().numpy(), axis=-1)), batch_size_i,
                ], dtype=np.float32); accuracy_train.append(accuracy_i)
                cross_entropy_i = cal_cross_entropy(y_pred_i, y_true_i); loss_i.total = loss_i.cls = cross_entropy_i
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_train, key_i):
                        loss_train[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_train[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            # Record information related to train process.
            accuracy_train = np.stack(accuracy_train, axis=0)
            accuracy_train = np.sum(accuracy_train[:,0] * accuracy_train[:,1]) / np.sum(accuracy_train[:,1])
            for key_i, item_i in loss_train.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_train[key_i] = item_i
            # Prepare for model validation process.
            accuracy_validation = []; loss_validation = utils.DotDict()
            # Execute validation process.
            for validation_batch in dataset_validation:
                # Initialize `batch_i` from `validation_batch`.
                batch_i = [
                    validation_batch["X"].to(device=params.model.device),
                    validation_batch["y"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Validate model for current batch.
                y_pred_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                y_pred_i = y_pred_i.detach().cpu().numpy(); y_true_i = batch_i[1].detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                accuracy_i = np.array([
                    np.mean(np.argmax(y_pred_i, axis=-1) == np.argmax(batch_i[1].detach().cpu().numpy(), axis=-1)), batch_size_i,
                ], dtype=np.float32); accuracy_validation.append(accuracy_i)
                cross_entropy_i = cal_cross_entropy(y_pred_i, y_true_i); loss_i.total = loss_i.cls = cross_entropy_i
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_validation, key_i):
                        loss_validation[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_validation[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            # Record information related to validation process.
            accuracy_validation = np.stack(accuracy_validation, axis=0)
            accuracy_validation = np.sum(accuracy_validation[:,0] * accuracy_validation[:,1]) / np.sum(accuracy_validation[:,1])
            for key_i, item_i in loss_validation.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_validation[key_i] = item_i
            accuracies_validation.append(accuracy_validation)
            # Prepare for model test process.
            accuracy_test = []; loss_test = utils.DotDict()
            # Execute test process.
            for test_batch in dataset_test:
                # Initialize `batch_i` from `test_batch`.
                batch_i = [
                    test_batch["X"].to(device=params.model.device),
                    test_batch["y"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Test model for current batch.
                y_pred_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                y_pred_i = y_pred_i.detach().cpu().numpy(); y_true_i = batch_i[1].detach().cpu().numpy()
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                accuracy_i = np.array([
                    np.mean(np.argmax(y_pred_i, axis=-1) == np.argmax(batch_i[1].detach().cpu().numpy(), axis=-1)), batch_size_i,
                ], dtype=np.float32); accuracy_test.append(accuracy_i)
                cross_entropy_i = cal_cross_entropy(y_pred_i, y_true_i); loss_i.total = loss_i.cls = cross_entropy_i
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_test, key_i):
                        loss_test[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_test[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            # Record information related to test process.
            accuracy_test = np.stack(accuracy_test, axis=0)
            accuracy_test = np.sum(accuracy_test[:,0] * accuracy_test[:,1]) / np.sum(accuracy_test[:,1])
            for key_i, item_i in loss_test.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_test[key_i] = item_i
            accuracies_test.append(accuracy_test)
            ## Write progress to summaries.
            # Log information related to current training epoch.
            time_stop = time.time()
            msg = (
                "Finish train epoch {:d} in {:.2f} seconds."
            ).format(epoch_idx, time_stop-time_start)
            print(msg); paths.run.logger.summaries.info(msg)
            # Log information related to train process.
            msg = "Accuracy(train): [{:.2f}%].\n".format(accuracy_train * 100.); loss_keys = list(loss_train.keys())
            msg += "Loss(train): {:.5f} ({})".format(loss_train[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_train[loss_keys[loss_idx]], loss_keys[loss_idx])
            print(msg); paths.run.logger.summaries.info(msg)
            # Log information related to validation process.
            msg = "Accuracy(validation): [{:.2f}%].\n".format(accuracy_validation * 100.); loss_keys = list(loss_validation.keys())
            msg += "Loss(validation): {:.5f} ({})".format(loss_validation[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_validation[loss_keys[loss_idx]], loss_keys[loss_idx])
            print(msg); paths.run.logger.summaries.info(msg)
            # Log information related to test process.
            msg = "Accuracy(test): [{:.2f}%].\n".format(accuracy_test * 100.); loss_keys = list(loss_test.keys())
            msg += "Loss(test): {:.5f} ({})".format(loss_test[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_test[loss_keys[loss_idx]], loss_keys[loss_idx])
            print(msg); paths.run.logger.summaries.info(msg)
            ## Write progress to tensorboard.
            # Get the pointer of writer.
            writer = paths.run.logger.tensorboard
            # Log information related to train process.
            for key_i, loss_i in loss_train.items():
                writer.add_scalar(os.path.join("losses", "train", key_i), loss_i, global_step=epoch_idx)
            writer.add_scalar(os.path.join("accuracies", "train"), accuracy_train, global_step=epoch_idx)
            # Log information related to validation process.
            for key_i, loss_i in loss_validation.items():
                writer.add_scalar(os.path.join("losses", "validation", key_i), loss_i, global_step=epoch_idx)
            writer.add_scalar(os.path.join("accuracies", "validation"), accuracy_validation, global_step=epoch_idx)
            # Log information related to test process.
            for key_i, loss_i in loss_test.items():
                writer.add_scalar(os.path.join("losses", "test", key_i), loss_i, global_step=epoch_idx)
            writer.add_scalar(os.path.join("accuracies", "test"), accuracy_test, global_step=epoch_idx)
            # Summarize model information.
            if epoch_idx == 0:
                msg = summary(model, col_names=("num_params", "params_percent", "trainable",))
                print(msg); paths.run.logger.summaries.info(msg)
        # Convert `accuracies_validation` & `accuracies_test` to `np.array`.
        # accuracies_* - (n_epochs,)
        accuracies_validation = np.round(np.array(accuracies_validation, dtype=np.float32), decimals=4)
        accuracies_test = np.round(np.array(accuracies_test, dtype=np.float32), decimals=4)
        epoch_maxacc_idx = np.where(accuracies_validation == np.max(accuracies_validation))[0]
        epoch_maxacc_idx = epoch_maxacc_idx[np.argmax(accuracies_test[epoch_maxacc_idx])]
        # Finish training process of current specified experiment.
        msg = (
            "Finish the training process of experiment {}."
        ).format(load_params_i.name)
        print(msg); paths.run.logger.summaries.info(msg)
        # Initialize log information for current subject.
        msg = (
            "For subject {}, we get test-accuracy ({:.2f}%) according to max validation-accuracy ({:.2f}%) at epoch {:d}."
        ).format(load_params_i.subj_cfg.name, accuracies_test[epoch_maxacc_idx] * 100.,
            accuracies_validation[epoch_maxacc_idx] * 100., epoch_maxacc_idx)
        print(msg); paths.run.logger.summaries.info(msg)
    # Finish current training process.
    writer = paths.run.logger.tensorboard; writer.close()
    # Log the end of current training process.
    msg = "Training finished with dataset {}.".format(params.train.dataset)
    print(msg); paths.run.logger.summaries.info(msg)

# def _forward func
def _forward(inputs):
    """
    Forward the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including [X,y_true].

    Returns:
        y_pred: (batch_size, n_labels) - The predicted labels.
        loss: DotDict - The loss dictionary.
    """
    global model; model.eval()
    with torch.no_grad(): return model(inputs)

# def _train func
def _train(inputs):
    """
    Train the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including [X,y_true].

    Returns:
        y_pred: (batch_size, n_labels) - The predicted labels.
        loss: DotDict - The loss dictionary.
    """
    global model, optimizer; model.train()
    # Forward model to get the corresponding loss.
    y_pred, loss = model(inputs)
    # Use optimizer to update parameters.
    optimizer.zero_grad(); loss["total"].backward(); optimizer.step()
    # Return the final `y_pred` & `loss`.
    return y_pred, loss

"""
vis funcs
"""
# def log_distr func
def log_distr(data, n_bins=10, n_hashes=100):
    """
    Log information related to data distribution.

    Args:
        data: (n_samples,) - The samples from data distribution.
        n_bins: int - The number of data range, each of which is a base unit to calculate probability.
        n_hashes: int - The total number of hashes (i.e., #) to identify the distribution probability.

    Returns:
        msg: str - The message related to data distribution.
    """
    # Create histogram bins.
    # bins - (n_bins+1,)
    bins = np.linspace(np.min(data), np.max(data), num=(n_bins + 1))
    # Calculate histogram counts.
    # counts - (n_bins,); probs - (n_bins,)
    counts, _ = np.histogram(data, bins=bins); probs = counts / np.sum(counts)
    # Print the histogram.
    msg = "\n"
    for bin_idx in range(len(probs)):
        range_i = "{:.5f} - {:.5f}".format(bins[bin_idx], bins[bin_idx+1]).ljust(20)
        distr_i = "#" * int(np.ceil(probs[bin_idx] * n_hashes))
        msg += "{} | {}\n".format(range_i, distr_i)
    # Return the final `msg`.
    return msg

"""
tool funcs
"""
# def cal_cross_entropy func
def cal_cross_entropy(y_pred, y_true):
    """
    Calcualate the cross-entropy according to `y_pred` & `y_true`.

    Args:
        y_pred: (*, n_labels) - The predicted probability distribution.
        y_true: (*, n_labels) - The original probability distribution.

    Returns:
        cross_entropy: np.float32 - The calculated cross entropy.
    """
    # Normalize `y_*` to get well-defined probability distribution.
    # y_* - (*, n_labels)
    y_pred = y_pred / np.sum(y_pred, axis=-1, keepdims=True) + 1e-12
    y_true = y_true / np.sum(y_true, axis=-1, keepdims=True) + 1e-12
    assert (y_pred >= 0.).all() and (y_true >= 0.).all()
    # Calculate the cross-entropy according to `y_pred` & `y_true`.
    # cross_entropy - (*,)
    cross_entropy = -np.sum(y_true * np.log(y_pred), axis=-1)
    # Average to get the final  `cross_entropy`.
    cross_entropy = np.mean(cross_entropy)
    # Return the final `cross_entropy`.
    return cross_entropy

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
    parser = argparse.ArgumentParser("DeWave CLS for brain signals", add_help=False)
    # Add training parmaeters.
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,])
    parser.add_argument("--subj", type=str, default="011")
    # Return the final `parser`.
    return parser

if __name__ == "__main__":
    import os
    # local dep
    from params.dewave_params import dewave_params as dewave_params

    # macro
    dataset = "seeg_he2023xuanwu"

    # Initialize base path.
    base = os.path.join(os.getcwd(), os.pardir, os.pardir)
    # Initialize arguments parser.
    args_parser = get_args_parser(); args = args_parser.parse_args()
    # Initialize dewave_params.
    dewave_params_inst = dewave_params(dataset=dataset)
    dewave_params_inst.train.base = base; dewave_params_inst.train.subj = args.subj
    # Initialize the training process.
    init(dewave_params_inst)
    # Loop the training process over random seeds.
    for seed_i in args.seeds:
        # Initialize random seed, then train dewave.
        utils.model.torch.set_seeds(seed_i); train()

