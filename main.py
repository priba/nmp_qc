#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodologi defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]

"""

# Own Modules
import datasets
from MessageFunction import MessageFunction
from UpdateFunction import UpdateFunction
from ReadoutFunction import ReadoutFunction

# Torch
import torch
import torch.optim as optim
import torch.nn as nn

import argparse
import os
import numpy as np

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


print('Argument Parser')
# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--dataset', default='qm9', help='QM9')
parser.add_argument('--datasetPath', default=['./data/qm9/dsgdb9nsd/'], help='dataset path')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='Input batch size for training (default: 64)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='Number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='Learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
# i/o
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='How many batches to wait before logging training status')

args = parser.parse_args()

# Check if CUDA is enabled
args.cuda = not args.no_cuda and torch.cuda.is_available()

# Load data
root = args.datasetPath[0]

print('Prepare files')
dtype = torch.FloatTensor

files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

idx = np.random.permutation(len(files))
idx = idx.tolist()

valid_ids = [files[i] for i in idx[0:10000]]
test_ids = [files[i] for i in idx[10000:20000]]
train_ids = [files[i] for i in idx[20000:]]

data_train = datasets.Qm9(root, train_ids)
data_valid = datasets.Qm9(root, valid_ids)
data_test = datasets.Qm9(root, test_ids)


# Define model and optimizer
class Nmp(nn.Module):
    def __init__(self, d, in_n, out, l_target):
        super(Nmp, self).__init__()

        # Define message 1 & 2
        self.m = nn.ModuleList([
                MessageFunction('duvenaud'),
                MessageFunction('duvenaud')
            ])

        # Define Update 1 & 2
        self.u = nn.ModuleList([
                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[0].get_out_size(in_n[0], in_n[1]), 'out': out[0]}),
                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[0].get_out_size(out[0], in_n[1]), 'out': out[1]})
            ])

        # Define Readout
        self.r = ReadoutFunction('duvenaud',
                                 args={'layers': len(self.m) + 1, 'in': [in_n[0], out[0], out[1]], 'out': out[2],
                                       'target': l_target})


    def forward(self, g_tuple):

        # Separate
        g, h_in, e = g_tuple

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):
            h.append({})
            for v in g.nodes_iter():
                neigh = g.neighbors(v)
                m_neigh = dtype()
                for w in neigh:
                    if (v, w) in e:
                        e_vw = e[(v, w)]
                    else:
                        e_vw = e[(w, v)]
                    m_v = self.m[t].forward(h[t][v], h[t][w], e_vw)
                    if len(m_neigh):
                        m_neigh += m_v
                    else:
                        m_neigh = m_v

                # Duvenaud
                opt = {'deg': len(neigh)}
                h[t+1][v] = self.u[t].forward(h[t][v], m_neigh, opt)

        # Readout
        return self.r.forward(h)

print('Define model')
# Select one graph
g_tuple, l = data_train[0]
g, h_t, e = g_tuple

print('\tStatistics')
#statDict = datasets.utils.get_graph_stats(data_valid, ['degrees', 'mean', 'std'])
d = [1, 2, 3, 4]

print('\tCreate model')
model = Nmp(d, [len(h_t.values()[0]), len(e.values()[0])], [25, 30, 35], len(l))

print('Check cuda')
if args.cuda:
    model.cuda()

print('Optimizer')
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# TODO Train function
def train(epoch):
	# TODO
    pass

# TODO Test function
def test(epoch):
	# TODO
    pass

# TODO Epoch for loop
for epoch in range(1, args.epochs + 1):
    pass