import torch
import numpy as np

if __name__ == '__main__':
    locations = torch.rand((2, 10+1, 2))
    dynamic_shape = (2, 1, 10 + 1)
    loads = torch.full(dynamic_shape, 1.)
    demands = torch.randint(1, 9 + 1, dynamic_shape)
    # demands = demands / float(20)
    demands[:, 0, 0] = 0  # depot starts with a demand of 0
    dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))
    print(dynamic)