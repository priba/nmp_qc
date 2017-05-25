#!/usr/bin/python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class NNet(nn.Module):

    def __init__(self, n_in, n_out):
        super(NNet, self).__init__()

        self.fc1 = nn.Linear(n_in, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_out)

    def forward(self, x):

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NNetM(nn.Module):

    def __init__(self, n_in, n_out):
        super(NNetM, self).__init__()

        self.fc1 = nn.Linear(n_in, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_out[0]*n_out[1])

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def main():
    net = NNet(n_in=100, n_out=20)
    print(net)

if __name__=='__main__':
    main()
