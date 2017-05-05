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
import pickle
import h5py

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)

from GraphReader.graph_reader import xyz_graph_reader

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"
        
def _load_pickle_file(pickle_file):
    f = h5py.File(pickle_file, "r")
    data = []
    for key in f.keys():
        data.append(f[key])
    return tuple(data)

class Qm9(data.Dataset):

    # Constructor
    def __init__(self, root_path, ids, type_, vertex_transform=utils.qm9_nodes, edge_transform=utils.qm9_edges,
                 target_transform=None):
        self.root = root_path
        self.ids = ids
        self.type_ = type_
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        self.create_hdf5()
#        self.create_pickle() # create pickle

    def __getitem__(self, index):
        g, target = xyz_graph_reader(os.path.join(self.root, self.ids[index]))
#        For pickle
#        with open(self.pickle_file, 'rb') as f:
#            d = pickle.load(f)
#        g, target = d[index]
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        if self.edge_transform is not None:
            g, e = self.edge_transform(g)
            g = g.adjacency_list()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (g, h, e), target

    def __len__(self):
        return len(self.ids)

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform
        
    def create_pickle(self):
        self.pickle_dir = os.path.abspath(os.path.join(self.root, os.pardir, 'pickles' ))
        self.pickle_file = os.path.join(self.pickle_dir, self.type_ + '.pickle')
        if os.path.exists(self.pickle_file):
            print('Pickle already exists.')
        else:    
            if not os.path.isdir(self.pickle_dir):
                os.makedirs(self.pickle_dir)            
            d = []
            for i, file in enumerate(self.ids):
                print('Pickling {0} Set: {1}/{2}'.format(self.type_, i+1, len(self.ids)))
                g, target = xyz_graph_reader(os.path.join(self.root, file))
                d += [[g, target]]
            with open(os.path.join(self.pickle_file), 'wb') as f:        
                pickle.dump(d, f)
                
    def create_hdf5(self):
        
        self.hdf5_dir = os.path.abspath(os.path.join(self.root, os.pardir, 'hdf5s' ))
        self.hdf5_file = os.path.join(self.hdf5_dir, self.type_ + '.hdf5')
        if os.path.exists(self.hdf5_file):
            print(self.hdf5_file +' already exists.')
        else:    
            if not os.path.isdir(self.hdf5_dir):
                os.makedirs(self.hdf5_dir)               
            
            d = []
            for i, file in enumerate(self.ids):
                print('Zipping {0} Set: {1}/{2}'.format(self.type_, i+1, len(self.ids)))
                g, target = xyz_graph_reader(os.path.join(self.root, file))
                d += [[g.adjacency_list(), target]]
                
            f = h5py.File(self.hdf5_file, 'w')
            dset = f.create_dataset('qm9_h5py', (len(d),2))
            dset[...] = d
            f.close()
        

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
