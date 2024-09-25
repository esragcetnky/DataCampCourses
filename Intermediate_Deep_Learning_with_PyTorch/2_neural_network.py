import torch
import torch.nn as nn
import numpy as np
np.random.seed(123)
torch.manual_seed(0)
# Create input_tensor with three features
input_tensor = torch.tensor([[0.3456,
                              0.89513,
                              -0.6513]])

# Define our first linear layer
linear_layer = nn.Linear(in_features=3,
                         out_features=2)

# Pass input through linear layer
# output = ( weights * input ) + bias
output = linear_layer(input_tensor)
print(f"Output : {output}\nLayer weight : {linear_layer.weight} \nLayer bias : {linear_layer.bias} ")

tensor_1 =  torch.randn(1, 10)

print(f"-------------- Input ---------------\n{tensor_1}")

# Stacked layers with nn.Sequential()
model = nn.Sequential(nn.Linear(10,18),
                      nn.Linear(18,20),
                      nn.Linear(20,5))

print(f"---------------- final Result -----------\n{model(tensor_1)}")