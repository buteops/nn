#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import os, sys
sys.path.append(Path.cwd().as_posix())

if __name__ == '__main__':
    print(os.getcwd())
