import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple neural network
class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.fc1 = nn.Linear(10, 5)  # Input size: 10, Output size: 5
    self.fc2 = nn.Linear(5, 2)   # Input size: 5, Output size: 2

  def forward(self, x):
    x = torch.relu(self.fc1(x))  # Apply ReLU activation function
    x = self.fc2(x)
    return x

# Create an instance of the network
model = SimpleNN()

# Define some input data
input_data = torch.randn(3, 10)  # Batch size: 3, Input size: 10

# Perform a forward pass
output = model(input_data)

print("Output shape:", output.shape)
print("Output tensor:")
print(output)
