#!/usr/bin/env python3
"""
Created on 13:10, Dec. 13th, 2022

@author: Norbert Zheng
"""
import os, shutil
import fnmatch, datetime
import logging, pickle
import copy as cp
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
from utils import DotDict

__all__ = [
    "Paths",
]

class Paths:
    """
    `Paths` class of the whole data-base & current training process.
    """

    def __init__(self, base, params):
        """
        Create [train,validation,test]-set of specified data-base, and
        create directories for storing data during a model training run.
        :param base: The base path of current project.
        :param params: The parameters of current iteration.
        """
        ## Initialize parameters.
        self.base, self.params = base, cp.deepcopy(params)

        ## Initialize variables.
        # Initialize data-related variables.
        self.data = DotDict({"base":os.path.join(self.base, "data"),})
        self.data.train = os.path.join(self.data.base, "train")
        self.data.validation = os.path.join(self.data.base, "validation")
        self.data.test = os.path.join(self.data.base, "test")
        # Check whether all data-related paths exists.
        if not os.path.exists(self.data.train) or\
           not os.path.exists(self.data.validation) or\
           not os.path.exists(self.data.test):
            shutil.rmtree(self.data.train, ignore_errors=True)
            shutil.rmtree(self.data.validation, ignore_errors=True)
            shutil.rmtree(self.data.test, ignore_errors=True)
        os.makedirs(self.data.train, exist_ok=True)
        os.makedirs(self.data.validation, exist_ok=True)
        os.makedirs(self.data.test, exist_ok=True)
        # Initialize run-related variables.
        self.run = DotDict({"base":None,"train":None,"model":None,"save":None,"script":None,})
        # Get current `date` for saving folder, and initialize current
        # `run` to create a new run folder within the current date.
        date, run = datetime.datetime.today().strftime("%Y-%m-%d"), 0
        # Find the current `run`: the first run that doesn't exist yet.
        while True:
            # Construct new paths.
            self.run.base = os.path.join(self.base, "summaries", date, str(run))
            self.run.train = os.path.join(self.run.base, "train")
            self.run.model = os.path.join(self.run.base, "model")
            self.run.save = os.path.join(self.run.base, "save")
            self.run.script = os.path.join(self.run.base, "script")
            # Update current `run`.
            run += 1
            # Once paths doesn't exist yet, create new folders.
            if not os.path.exists(self.run.train) and\
               not os.path.exists(self.run.model) and\
               not os.path.exists(self.run.save):
                os.makedirs(self.run.train); os.makedirs(self.run.model)
                os.makedirs(self.run.save); os.makedirs(self.run.script); break

        ### Initialize other variables and configuration files.
        ## Initialize other run-related variables.
        # Initialize the name of model.
        self._init_run_name()
        # Initialize logger of current run.
        self.run.logger = DotDict({
            "summaries": self._init_run_logger(self.run.base, "summaries"),
        })
        # Copy scripts.
        self._init_run_script()
        # Save params.
        self._init_run_params()

    """
    init run funcs
    """
    # def _init_run_name func
    def _init_run_name(self):
        """
        Initialize the name of current model training run, which is saved in `run`.
        """
        # Initialize name of current `Paths` object.
        self.name = "_".join(self.params.__class__.__name__.split("_")[:-1])
        with open(os.path.join(self.run.base, "name"), "w") as f:
            f.write(self.name)

    # def _init_run_logger func
    def _init_run_logger(self, path, name):
        """
        Create logger, output during training can be stored to file in
        a consistent way, which is saved in `run`.
        :param path: The directory path of logger.
        :param name: The file name of logger.
        :return logger: Created logger object.
        """
        # Create new logger.
        logger = logging.getLogger(name); logger.setLevel(logging.INFO)
        # Remove any existing handlers so you don't output to old files, or to
        # new files twice - important when resuming training exsiting model.
        logger.handlers = []
        # Create a file handler, and create a logging format.
        handler = logging.FileHandler(os.path.join(path, name+".log")); handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s: %(message)s"); handler.setFormatter(formatter)
        logger.addHandler(handler)
        # Return the logger object.
        return logger

    # def _init_run_script func
    def _init_run_script(self):
        """
        Copy scripts of current run to script path, which is saved in `script`.
        """
        # Initialize `ignore_dirs` & `ignore_patterns`.
        ignore_dirs = ["data", "docs", "pretrains", "slurm", "summaries", "summaries-hpc",]
        ignore_patterns = [
            # dirs
            "__pycache__",
            # files
            "*.ipynb",
        ]
        # Define the rule of ignore.
        def ignore_func(dir_, files):
            # Get the current sub-directory level relative to the directory being copied.
            sub_level = dir_[len(self.base):].count(os.path.sep)
            # Filter out directories to ignore.
            ignore_files = [file_i for file_i in files if os.path.isdir(os.path.join(dir_, file_i)) and\
                (file_i in ignore_dirs) and (sub_level == 0)]
            # Filter out file patterns to ignore.
            ignore_files.extend([file_i for file_i in files if\
                any([fnmatch.fnmatch(file_i, pattern_i) for pattern_i in ignore_patterns])])
            # Return the final `ignore_files`.
            return ignore_files
        # Copy scripts while ignoring the specified patterns.
        shutil.copytree(self.base, self.run.script, dirs_exist_ok=True, ignore=ignore_func)

    # def _init_run_params func
    def _init_run_params(self):
        """
        Save the parameters of current iteration, which is saved in `save`.
        """
        Paths.save_pickle(os.path.join(self.run.save, "params"), DotDict(self.params))

    """
    static funcs
    """
    ## def pickle funcs
    # def save_pickle func
    @staticmethod
    def save_pickle(fname, obj):
        """
        Save object to pickle file.
        :param fname: The file name to save object.
        :param obj: The object to be saved.
        """
        with open(fname, "wb") as f:
            pickle.dump(obj, f)

    # def load_pickle func
    @staticmethod
    def load_pickle(fname):
        """
        Load object from pickle file.
        :param fname: The file name to load object.
        :return obj: The object loaded from file.
        """
        with open(fname, "rb") as f:
            obj = pickle.load(f)
        return obj

if __name__ == "__main__":
    import mne
    # local dep
    from params import lasso_regression_params

    # macro
    base = os.path.join(os.getcwd(), os.pardir, os.pardir)

    ## Check `Paths` class.
    # Get current training parameters.
    params = lasso_regression_params()
    # Instantiate `Paths` object.
    paths_inst = Paths(base=base, params=params)

