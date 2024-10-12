"""Image and label map utility functions."""

import logging
import numpy as np


def create_category_index(categories):
  """Creates dictionary of COCO compatible categories keyed by category id.

  Args:
    categories: a list of dicts, each of which has the following keys:
      'id': (required) an integer id uniquely identifying this category.
      'name': (required) string representing category name
        e.g., 'cat', 'dog'.

  Returns:
    category_index: a dict containing the same entries as categories, but keyed
      by the 'id' field of each category.
  """
  category_index = {}
  for index, cat in enumerate(categories):
    category_index[index] = cat
  return category_index


def load_labelmap(path):

  with open(path, 'r') as fid:
    label_map = list(map(str.strip, fid.readlines()))
  return label_map


def get_label_map_dict(label_map_path):
  """Reads a label map and returns a dictionary of label names to id.

  Args:
    label_map_path: path to label_map.

  Returns:
    A dictionary mapping label names to id.
  """
  label_map = load_labelmap(label_map_path)
  label_map_dict = {}
  for item in label_map.item:
    label_map_dict[item.name] = item.id
  return label_map_dict


def load_image(frame, new_size=(300, 300)):
  # Get the dimensions
  height, width, _ = frame.shape # Image shape
  new_width, new_height = new_size # Target shape 

  # Calculate the target image coordinates
  left = (width - new_width) // 2
  top = (height - new_height) // 2
  right = (width + new_width) // 2
  bottom = (height + new_height) // 2

  image = frame[left: right, top: bottom, :]
  return image
