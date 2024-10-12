#!/usr/bin/env python3

from pathlib import Path
from setuptools import find_packages, setup
from typing import List

directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding = 'utf-8') as readme:
    long_description = readme.read()

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
  requirements=[]
  with open(file_path) as file_obj:
    requirements=file_obj.readlines()
    requirements=[req.replace("\n","") for req in requirements]
    if HYPEN_E_DOT in requirements:
      requirements.remove(HYPEN_E_DOT)
  return requirements

setup(
  name="mlopsency",
  version="1.0.0",
  description = "you interest to mlops? let's train together 👊",
  package_dir = {"":"mlopsency"},
  author="pandohansamuel19",
  license = 'MIT',
  long_description = long_description,
  long_description_content_type='text/markdown',
  python_requires = '>=3.10',
  classifiers=[
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Operating System :: OS Independent",
  ],
  packages = find_packages(),
  install_requires = get_requirements('linux-requirements.txt'),
  extras_require = {
    "dev": ["pytest>=7.0", "twine>=4.0.2"],
  }
)