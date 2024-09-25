import torch
import torch.nn as nn
import numpy as np

import torch.nn.functional as F

np.random.seed(123)
torch.manual_seed(0)

# loss function : gives feedback to model during training
# loss = F (y, y_hat)

# Cross Entrophy
from torch.nn import CrossEntropyLoss

scores = torch.randn(1,2)
one_hot_target = torch.tensor([[1,0]])

criterion = CrossEntropyLoss()
criterion(scores.double(),one_hot_target.double())


