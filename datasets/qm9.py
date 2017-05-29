#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
qm9.py:

Usage:

"""

# Networkx should be imported before torch
import networkx as nx

import torch.utils.data as data
import numpy as np
import argparse

import datasets.utils as utils
import time
import os,sys

import torch

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)

from GraphReader.graph_reader import xyz_graph_reader

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

class Qm9(data.Dataset):

    # Constructor
    def __init__(self, root_path, ids, vertex_transform=utils.qm9_nodes, edge_transform=utils.qm9_edges,
                 target_transform=None, e_representation='raw_distance'):
        self.root = root_path
        self.ids = ids
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.e_representation = e_representation

    def __getitem__(self, index):
        g, target = xyz_graph_reader(os.path.join(self.root, self.ids[index]))
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        if self.edge_transform is not None:
            g, e = self.edge_transform(g, self.e_representation)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (g, h, e), target

    def __len__(self):
        return len(self.ids)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform

if __name__ == '__main__':

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='QM9 Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['../data/qm9/dsgdb9nsd'])

    args = parser.parse_args()
    root = args.root[0]

    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    idx = np.random.permutation(len(files))
    idx = idx.tolist()

    valid_ids = [files[i] for i in idx[0:10000]]
    test_ids  = [files[i] for i in idx[10000:20000]]
    train_ids = [files[i] for i in idx[20000:]]

    data_train = Qm9(root, train_ids, vertex_transform=utils.qm9_nodes, edge_transform=lambda g: utils.qm9_edges(g, e_representation='raw_distance'))
    data_valid = Qm9(root, valid_ids)
    data_test = Qm9(root, test_ids)

    print(len(data_train))
    print(len(data_valid))
    print(len(data_test))
    
    print(data_train[1])
    print(data_valid[1])
    print(data_test[1])

    start = time.time()
    print(utils.get_graph_stats(data_valid, 'degrees'))
    end = time.time()
    print('Time Statistics Par')
    print(end - start)
