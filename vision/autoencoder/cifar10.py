from __future__ import annotations
# from typing import List, Any, Callable

import keras
import tensorflow as tf
# import tensorflow_datasets as tfds
from keras import backend as K

BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1024


def image_map(image, label): return tf.cast(image, dtype=tf.float32) / 255.0
def image_processing():
  train_dataset = tfds.load('cifar10', split='train', as_supervised=True)
  train_dataset = train_dataset.map(image_map)
  train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
  test_dataset = tfds.load('cifar10', split='test', as_supervised=True)
  test_dataset = test_dataset.map(image_map)
  test_dataset = test_dataset.batch(BATCH_SIZE)
  return train_dataset, test_dataset

def build_model():
  '''
    Create the autoencoder model. you will want to downsample the image in the encoder layers then upsample it in the decoder path. Note that the output layer should be the same dimensions as the original image. Your input images will have the shape `(32, 32, 3)`. If you deviate from this, your model may not be recognized by the grader and may fail.

    We included a few hints to use the Sequential API below but feel free to remove it and use the Functional API just like in the ungraded labs if you're more comfortable with it. Another reason to use the latter is if you want to visualize the encoder output. As shown in the ungraded labs, it will be easier to indicate multiple outputs with the Functional API. That is not required for this assignment though so you can just stack layers sequentially if you want a simpler solution.
  '''
  K.clear_session()
  
  train, test = image_processing()
  model = keras.models.Sequential([
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu", input_shape=(32, 32, 3)),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
    keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
    keras.layers.UpSampling2D(size=(2, 2)),
    keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation="relu"),
    keras.layers.UpSampling2D(size=(2, 2)),
    keras.layers.Conv2D(filters=3, kernel_size=(3, 3), padding="same", activation="sigmoid")
  ])
  
  model.compile(optimizer='adam', metrics=['accuracy'], loss='mean_squared_error')
  train_steps = len(train) // BATCH_SIZE
  val_steps = len(test) // BATCH_SIZE

  ### START CODE HERE ###
  model.fit(train, steps_per_epoch=train_steps, validation_data=test, validation_steps=val_steps, epochs=50)
  model.summary()
  return model.evaluate(test, steps=10)