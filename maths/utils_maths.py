#/usr/bin/env python3
from __future__ import annotations
import os, sys, logging, math, time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict


_DATASETS_ENDPOINT = Path(__file__).resolve().parent.parent.parent / 'datasets'
_ASSETS_ENDPOINT = Path(__file__).resolve().parent.parent.parent / 'assets'
_MODELS_ENDPOINT = Path(__file__).resolve().parent.parent.parent / 'models'
_TESTS_ENDPOINT = Path(__file__).resolve().parent.parent.parent / 'tests'
_UTILIZERS = Path(__file__).resolve().parent.parent.parent / 'utilizers'