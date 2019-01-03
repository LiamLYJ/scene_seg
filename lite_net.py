import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as nn_F
import numpy as np
# from nets import *


class lite_net(nn.Module):
    def __init__(self,):
        super(lite_net, self).__init__()

        self.conv1 = nn.Conv2d(6, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.relu(x)

        x = self.conv4(x)
        y = self.sigmoid(x)

        return y
