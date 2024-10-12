#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys
sys.path.append(Path.cwd().as_posix())

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
  (training_images, training_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
  
  # You can put between 0 to 59999 here
  index = 0
  # Set number of characters per row when printing
  np.set_printoptions(linewidth=320)

  # Print the label and image
  print(f'LABEL: {training_labels[index]}')
  print(f'\nIMAGE PIXEL ARRAY:\n {training_images[index]}')


  training_images  = training_images / 255.0
  test_images = test_images / 255.0
  model = keras.models.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
  
  # Declare sample inputs and convert to a tensor
  inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
  inputs = tf.convert_to_tensor(inputs)
  print(f'input to softmax function: {inputs.numpy()}')

  # Feed the inputs to a softmax activation function
  outputs = keras.activations.softmax(inputs)
  print(f'output of softmax function: {outputs.numpy()}')

  # Get the sum of all values after the softmax
  sum = tf.reduce_sum(outputs)
  print(f'sum of outputs: {sum}')

  # Get the index with highest value
  prediction = np.argmax(outputs)
  print(f'class with highest probability: {prediction}')
  
  model.compile(optimizer = keras.optimizers.Adam(), loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(training_images, training_labels, epochs=5)
  
  classifications = model.predict(test_images)

  print(classifications[0])
  print(test_labels[0])