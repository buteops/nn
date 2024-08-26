#!/usr/bin/env python3
from __future__ import annotations
import os, pathlib, urllib.request, zipfile

def rps_download():
  DPATH = pathlib.Path("datasets").as_posix()
  if not os.path.isdir(DPATH): os.makedirs(DPATH)
  data_url = 'https://github.com/dicodingacademy/assets/releases/download/release-rps/rps.zip'
  urllib.request.urlretrieve(data_url, f'{DPATH}/rps.zip')
  zip_ref = zipfile.ZipFile(f'{DPATH}/rps.zip', 'r')
  zip_ref.extractall(DPATH)
  zip_ref.close()
  os.remove(f'{DPATH}/rps.zip')