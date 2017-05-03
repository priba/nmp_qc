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
from models.model import Nmp

# Torch
import torch
import torch.optim as optim

import argparse
import os
import numpy as np

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


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


def main():
    global args
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    root = args.datasetPath[0]

    print('Prepare files')
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

    # TODO Epoch for loop
    for epoch in range(1, args.epochs + 1):
        train(epoch)


# TODO Train function
def train(train_loader, model, criterion, optimixer, epoch):
    # switch to train mode
    model.train()


# TODO
def validate(val_loader, model, criterion):
    # switch to evaluate mode
    model.eval()

if __name__ == '__main__':
    main()