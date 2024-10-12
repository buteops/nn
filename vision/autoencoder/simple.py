#!/usr/bin/env python3

from __future__ import annotations
# import keras
import numpy as np
import matplotlib.pyplot as plt

"""First Autoencoder"""

def generate_data(m):
  '''plots m random points on a 3D plane'''

  angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
  data = np.empty((m, 3))
  data[:,0] = np.cos(angles) + np.sin(angles)/2 + 0.1 * np.random.randn(m)/2
  data[:,1] = np.sin(angles) * 0.7 + 0.1 * np.random.randn(m) / 2
  data[:,2] = data[:, 0] * 0.1 + data[:, 1] * 0.3 + 0.1 * np.random.randn(m)

  return data

if __name__ == '__main__':
  # use the function above to generate data points
  X_train = generate_data(100)
  X_train = X_train - X_train.mean(axis=0, keepdims=0)

  # preview the data
  ax = plt.axes(projection='3d')
  ax.scatter3D(X_train[:, 0], X_train[:, 1], X_train[:, 2], cmap='Reds')