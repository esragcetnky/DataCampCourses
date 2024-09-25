import numpy as np
# importing torch
import torch
# Pytorch supports
    # image data with torchvision
    # audio data with torchaudio
    # text data with torchtext

np.random.seed(123)
torch.manual_seed(0)

list_1 = [[1,2,3],[4,5,6]]
tensor_1 = torch.tensor(list_1)

print("----------- Tensor 1 Type--------------")
print(tensor_1)
print(type(tensor_1))


np_array_2 = np.array([1,10.5,-15])
tensor_2 = torch.from_numpy(np_array_2)

print("----------- Tensor 2 Shape, Type--------------")
print(tensor_2)
print(tensor_2.shape)
print(tensor_2.dtype)


print("----------- Tensor 2 Device--------------")
print(tensor_2.device)

print("----------- Tensor 1 x Tensor 2--------------")
print(tensor_2*tensor_1)

print("----------- Tensor 1 - Tensor 2--------------")
print(tensor_1-tensor_2)