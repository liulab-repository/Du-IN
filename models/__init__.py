#!/usr/bin/env python3
"""
Created on 15:53, Dec. 19th, 2022

@author: Norbert Zheng
"""
import os
## Initialize macros.
# Initialize path_module macro.
path_module = os.path.dirname(os.path.abspath(__file__))
# Initialize models macro.
models = [os.path.splitext(fname_i)[0] for fname_i in os.listdir(path_module)\
    if (not (fname_i.startswith("_") or fname_i.startswith("."))) and (fname_i != "Makefile")]

