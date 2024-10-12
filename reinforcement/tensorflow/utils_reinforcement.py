#/usr/bin/env python3
from __future__ import annotations
import os, sys, logging, math, time, h5py
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict

import numpy as np

_DATASETS_ENDPOINT = Path(__file__).resolve().parent.parent / 'datasets'
_ASSETS_ENDPOINT = Path(__file__).resolve().parent.parent / 'assets'
_MODELS_ENDPOINT = Path(__file__).resolve().parent.parent / 'models'
_TESTS_ENDPOINT = Path(__file__).resolve().parent.parent / 'tests'
_UTILIZERS = Path(__file__).resolve().parent.parent / 'utilizers'


print(_DATASETS_ENDPOINT)

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes