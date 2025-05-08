import numpy
import torch

from sinkhorn_loss_wasserstein import *
from utils import *;

martix=numpy.random.randint(1,100,size=(3,3))

dev=[[0,0],[1,1],[2,2]]
dev=torch.tensor(dev)
martix=torch.tensor(martix)
get_hits_sinkhorn(test_pair=dev, S=martix)
print()