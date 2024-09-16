#!/usr/bin/env python3
from __future__ import annotations
import random
"""
Key Points:
- Neural Networks are inspired by human brain neuron that translated to the computer
- Dense Layers, consist interconnected neurons that connecting to every next layers. each connections has a `weight` associated with it
- Biases, that included are to offset the output, either positive or negative.
- Weight and Biases are tunable parameters that will impacting the neurons
- ***Output = activation(sum(weight·inputs) + bias)***
- **Generalizations**: Learning process that trying to fit the data, instead memorizing
"""

random.seed(10)
INPUTS = [random.uniform(-1, 5) for _ in range(4)]
WEIGHTS = [[random.uniform(-5, 5) for _ in range(4)] for _ in range(3)]
BIAS = [random.uniform(-1, 5) for _ in range(3)]

def first_neuron():
  """TODO
    - Here simple implementations for sum(weight·inputs) + bias; This is a single Neurons
    - The Layer of Neurons; The scenario for 2 neurons in layers with 4 inputs
  """

  outputs = []
  for neuron_weights, neuron_bias in zip(WEIGHTS, BIAS):
      neuron_output = 0
      for n_input, weight in zip(INPUTS, neuron_weights):
          neuron_output += n_input*weight
      neuron_output += neuron_bias
      outputs.append(round(neuron_output, 2))
  return outputs

if __name__ == '__main__':
  print(first_neuron())
