#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodologi defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]

"""

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

import argparse
import torch

## Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--dataset', default='MUTAG', help='MUTAG')
parser.add_argument('--datasetPath', default='./db/mutag', help='dataset path')
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

## TODO Load data

## TODO Define model and optimizer

## TODO Train function
def train(epoch):
	# TODO
    pass

## TODO Test function
def test(epoch):
	# TODO
    pass

## TODO Epoch for loop
for epoch in range(1, args.epochs + 1):
 pass