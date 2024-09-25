import torch
import torch.nn as nn
import numpy as np

np.random.seed(123)
torch.manual_seed(0)

# forward pass : input data is passed forward or propagated through a network
# backward pass : backpropagation is used to update weights and biases during training


# binary classification
print(f"---------------------Binary Classification-------------------------")

input_data = torch.randn(5,6)
print(f"Input data : {input_data}")
model = nn.Sequential(nn.Linear(6,4),
                      nn.Linear(4,1),
                      nn.Sigmoid())
output_data = model(input_data)
print(f"Output Data : {output_data}")

print(f"---------------------Multi-Class Classification-------------------------")
n_classes = 3

model = nn.Sequential(nn.Linear(6,4),
                      nn.Linear(4,1),
                      nn.Softmax(dim=-1))

output = model(input_data)
print(f"Output : {output}")

print(f"---------------------Regression-------------------------")
model = nn.Sequential(nn.Linear(6,4),
                      nn.Linear(4,n_classes))

output = model(input_data)
print(f"Output : {output}")



