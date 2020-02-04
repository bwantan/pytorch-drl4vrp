import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def train(actor, critic, c, a, b, **kwargs):
    print(actor)
    print(critic)
    print(c)
    print(a)

if __name__ == '__main__':
    num_samples = 2
    input_size = 10
    max_demand = 9
    dynamic_shape = (num_samples, 1, input_size + 1)
    demands = torch.randint(1, max_demand + 1, dynamic_shape)
    demands[:,0,0] = 0
    loads = torch.full(dynamic_shape, 1.)
    load2 = torch.ones(dynamic_shape)
    dynamic = torch.tensor(np.concatenate((loads, demands), axis=1))
    print(loads)
    print(dynamic)