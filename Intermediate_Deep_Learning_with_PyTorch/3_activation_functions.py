import torch
import torch.nn as nn
import numpy as np
np.random.seed(123)
torch.manual_seed(0)

### Sigmoid
# 0 - 1
# binary classification
input_tensor = torch.tensor([[6.0,2.0,10.3,-1.5]])
sigmoid = nn.Sigmoid()
output = sigmoid(input_tensor)
print(f"Sigmoid output :{output}")

model = nn.Sequential(nn.Linear(6,4),
                      nn.Linear(4,1),
                      nn.Sigmoid())


### Softmax
# 0 - 1
# multi-class classification
input_tensor = torch.tensor([[4.3,6.1, 2.3]])

probability = nn.Softmax(dim=-1)
output_tensor = probability(input_tensor)

print(f"Softmax output :{output_tensor}")