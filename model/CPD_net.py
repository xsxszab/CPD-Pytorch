
import numpy as np
import torch
import torch.nn as nn
import torchvision
import PIL

from branch_vgg16 import Branch_vgg16


class HAM(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
