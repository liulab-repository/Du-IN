#!/usr/bin/env python3
"""
Created on 21:02, Feb. 14th, 2022

@author: Norbert Zheng
"""
import copy as cp
import numpy as np

__all__ = [
    "DotDict",
]

class DotDict(dict):

    # init class
    def __init__(self, *args, **kwargs):
        # We trust the dict to init itself better than we can.
        dict.__init__(self, *args, **kwargs)
        # Because of that, we do duplicate work, but it's worth it.
        for k, v in self.items():
            self.__setitem__(k, v)

    ## def object funcs
    # def get func
    def __getattr__(self, k):
        try:
            return dict.__getitem__(self, k)
        except KeyError:
            # Maintain consistent syntactical behaviour.
            raise AttributeError(
                "ERROR: 'DotDict' object has no attribute '" + str(k) + "'"
            )

    # def set func
    def __setitem__(self, k, v):
        dict.__setitem__(self, k, DotDict.dict2dotdict(v))
    __setattr__ = __setitem__

    # def del func
    def __delattr__(self, k):
        try:
            dict.__delitem__(self, k)
        except KeyError:
            raise AttributeError(
                "ERROR: 'DotDict' object has no attribute '" + str(k) + "'"
            )

    """
    static funcs
    """ 
    # def dict2dotdict func
    @staticmethod
    def dict2dotdict(obj):
        """
        Recursively convert `dict` objects in `dict`, `list`, `set`, and `tuple` objects
        to `dotdict` objects. Not change the original dotdict, generate new object.
        :param obj: The `dict`-style object to be transformed into `DotDict`-style object.
        """
        if isinstance(obj, dict):
            obj = DotDict(obj)
        elif isinstance(obj, list):
            obj = list(DotDict.dict2dotdict(item) for item in obj)
        elif isinstance(obj, set):
            obj = set(DotDict.dict2dotdict(item) for item in obj)
        elif isinstance(obj, tuple):
            obj = tuple(DotDict.dict2dotdict(item) for item in obj)

        return obj

    # def to_dict func
    @staticmethod
    def dotdict2dict(obj):
        """
        Recursively transforms `dotdict` object in `dict`, `list`, `set`, and `tuple`
        objects into `dotdict` objects. Not change the original dotdict, generate new object.
        :param obj: The `DotDict`-style object to be transformed into `dict`-style object.
        """
        # init data
        data = obj
        # set data
        if isinstance(obj, dict):
            data = {}
            for k, v in obj.items():
                data[k] = DotDict.dotdict2dict(v)
        elif isinstance(obj, list):
            data = list(DotDict.dotdict2dict(item) for item in obj)
        elif isinstance(obj, set):
            data = list(DotDict.dotdict2dict(item) for item in obj)
        elif isinstance(obj, tuple):
            data = list(DotDict.dotdict2dict(item) for item in obj)

        return data

    # def iter_keys func
    @staticmethod
    def iter_keys(obj):
        """
        Get all keys iteratively.
        :param obj: The `dict` object to get its keys iteratively.
        :return keys: The keys list of object.
        """
        # Initialize keys.
        keys = []
        # Check whether `obj` is dict.
        if isinstance(obj, dict):
            for key, val in obj.items():
                # If `val` is dict, then iteratively get its keys.
                if isinstance(val, dict):
                    keys_i = DotDict.iter_keys(val)
                    if not keys_i: continue
                    # Insert the current key at the head of list.
                    for key_i in keys_i:
                        key_i.insert(0, key)
                # If `val` is not dict, then directly get its key.
                else:
                    keys_i = [[key,],]
                keys.extend(keys_i)
        # Return the final keys.
        return keys

    # def iter_getattr func
    @staticmethod
    def iter_getattr(obj, key):
        """
        Get the corresponding value of key.
        :param obj: The `dict` object to get its val iteratively.
        :param key: The query key, list or str.
        :return val: The value of query key.
        """
        key = cp.deepcopy(key)
        if isinstance(key, str): key = [key,]
        assert type(key) is list; val = obj
        try:
            for _ in range(len(key)):
                val = val.get(key.pop(0))
        except Exception:
            raise ValueError("ERROR: Cannot get the value of {} from {}".format(
                ".".join(key), val))
        return val

    # def iter_setattr func
    @staticmethod
    def iter_setattr(obj, key, val):
        """
        Set the corresponding value of key.
        :param obj: The `dict` object to get its val iteratively.
        :param key: The query key, list or str.
        :param val: The value of query key.
        """
        key = cp.deepcopy(key)
        if isinstance(key, str): key = [key,]
        assert type(key) is list
        for _ in range(len(key)-1):
            key_i = key.pop(0)
            if not hasattr(obj, key_i): setattr(obj, key_i, DotDict())
            obj = getattr(obj, key_i)
        key_i = key.pop(0); setattr(obj, key_i, val)

    # def dotdictlst2dotdict func
    @staticmethod
    def dotdictlst2dotdict(dotdictlst):
        """
        Convert the list of `DotDict` to the `DotDict` of list.
        Not change the original dotdictlst, generate new object.
        :param dotdictlst: The list of `DotDict`, each of which has the same keys.
        :return dotdict: The `DotDict` of list, each of which has the same shape.
        """
        # Check whether every item in dotdictlst has the same iter_keys.
        assert len(dotdictlst) > 0; assert isinstance(dotdictlst[0], DotDict)
        keys = DotDict.iter_keys(dotdictlst[0])
        for item in dotdictlst:
            assert keys == DotDict.iter_keys(item)
        # Initialize dotdict.
        dotdict = DotDict()
        # Fill the dotdict using each item of dotdictlst.
        for item in dotdictlst:
            # For the first item, simply copy the components.
            if not dotdict.keys():
                for key in keys:
                    DotDict.iter_setattr(dotdict, key, [DotDict.iter_getattr(item, key),])
            # For all next items, add the components to the existing list.
            else:
                for key in keys:
                    DotDict.iter_getattr(dotdict, key).append(
                        DotDict.iter_getattr(item, key))
        # Return the final dotdict.
        return dotdict

    # def dotdict2dotdictlst func
    @staticmethod
    def dotdict2dotdictlst(dotdict):
        """
        Convert the `DotDict` of list to the list of `DotDict`.
        Not change the original dotdict, generate new object.
        :param dotdict: The `DotDict` of list, each of which has the same shape.
        :return dotdictlst: The list of `DotDict`, each of which has the same keys.
        """
        # Check whether every item dotdict has the same shape.
        assert isinstance(dotdict, DotDict)
        n_iters = None; keys = DotDict.iter_keys(dotdict)
        for key in keys:
            if not n_iters:
                n_iters = len(DotDict.iter_getattr(dotdict, key))
            else:
                assert n_iters == len(DotDict.iter_getattr(dotdict, key))
        # Initialize dotdictlst.
        dotdictlst = []
        # Fill the dotdictlst using each item of dotdict.
        for iter_idx in range(n_iters):
            # Initialize dotdict_i.
            dotdict_i = DotDict()
            # Fill the dotdict_i using each item of dotdict.
            for key in keys:
                DotDict.iter_setattr(dotdict_i, key, DotDict.iter_getattr(dotdict, key)[iter_idx])
            # Append dotdict_i to dotdictlst.
            dotdictlst.append(dotdict_i)
        # Return the final dotdictlst.
        return dotdictlst

    # def dotdict2numpydotdict func
    @staticmethod
    def dotdict2numpydict(dotdict):
        """
        Convert `DotDict` object to `np.array`.
        Not change the original dotdict, generate new object.
        :param dotdict: The `DotDict` of list, each of which has the same shape.
        :return numpydict: The `DotDict` of list, each of which is `np.array`.
        """
        numpydict = DotDict()
        keys = DotDict.iter_keys(dotdict)
        for key in keys:
            try:
                DotDict.iter_setattr(numpydict, key, np.array(DotDict.iter_getattr(dotdict, key)))
            except Exception:
                DotDict.iter_setattr(numpydict, key, DotDict.iter_getattr(dotdict, key))
        # Return the final numpydict.
        return numpydict

if __name__ == "__main__":
    # Instantiate DotDict.
    dotdict_inst = DotDict({
        "a": 1, "b": 2, "c": DotDict({
            "d": 3, "e": 4, "f": DotDict({
                "g": 5, "h": 6
    })})})
    # dotdict2dict.
    dotdict_inst = DotDict.dotdict2dict(dotdict_inst)
    # dict2dotdict.
    dotdict_inst = DotDict.dict2dotdict(dotdict_inst)
    # Get iter_keys.
    iter_keys = DotDict.iter_keys(dotdict_inst)
    # Get iter_getattr.
    iter_getattr = DotDict.iter_getattr(dotdict_inst, iter_keys[-1])

    # Initialize DotDict list.
    dotdictlst = [DotDict({"a":1,"b":2,"c":{"d":5,"e":6}}),
        DotDict({"a":3,"b":4,"c":{"d":7,"e":8}})]
    # Convert dotdictlst to dotdict.
    dotdict = DotDict.dotdictlst2dotdict(dotdictlst)
    # Convert dotdict to dotdictlst.
    dotdictlst = DotDict.dotdict2dotdictlst(dotdict)
    # Convert dotdict to numpydict.
    numpydict = DotDict.dotdict2numpydict(dotdict)

