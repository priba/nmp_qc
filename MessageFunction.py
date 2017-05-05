#!/usr/bin/python                                                                                                                                                                                               
# -*- coding: utf-8 -*-

"""
    MessageFunction.py: Propagates a message depending on two nodes and their common edge.

    Usage:

"""

from __future__ import print_function

# Own modules
import datasets

import numpy as np
import os
import argparse
import time
import torch

import torch.nn as nn

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat" 


class MessageFunction(nn.Module):

    # Constructor
    def __init__(self, message_def='duvenaud', args={}):
        super(MessageFunction, self).__init__()
        self.m_definition = ''
        self.m_function = None
        self.args = {}
        self.__set_message(message_def, args)

    # Message from h_v to h_w through e_vw
    def forward(self, h_v, h_w, e_vw, args=None):
        return self.m_function(h_v, h_w, e_vw, args)

    # Set a message function
    def __set_message(self, message_def, args={}):
        self.m_definition = message_def.lower()

        self.m_function = {
                    'duvenaud':     self.m_duvenaud,
                    'ggnn':         self.m_ggnn,
                    'intnet':       self.m_intnet,
                    'mgc':          self.m_mgc,
                    'bruna':        self.m_bruna,
                    'defferrard':   self.m_deff,
                    'kipf':         self.m_kipf
                }.get(self.m_definition, None)

        if self.m_function is None:
            print('WARNING!: Message Function has not been set correctly\n\tIncorrect definition ' + message_def)
            quit()

        self.learn_args, self.learn_modules, self.args = {
                'duvenaud': self.init_duvenaud(args)
            }.get(self.m_definition, (nn.ParameterList([]),nn.ModuleList([]),{}))

        self.m_size = {
                'duvenaud': self.out_duvenaud
            }.get(self.m_definition, None)

    # Get the name of the used message function
    def get_definition(self):
        return self.m_definition

    # Get Output size
    def get_out_size(self, size_h, size_e, args=None):
        return self.m_size(size_h, size_e, args)

    # Definition of various state of the art message functions
    
    # Duvenaud et al. (2015), Convolutional Networks for Learning Molecular Fingerprints
    def m_duvenaud(self, h_v, h_w, e_vw, args):
        m = torch.cat([h_w, e_vw], 0)
        return m

    def out_duvenaud(self, size_h, size_e, args):
        return size_h + size_e

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}
        return learn_args, learn_modules, args

    # Li et al. (2016), Gated Graph Neural Networks (GG-NN)
    def m_ggnn(self, h_v, h_w, e_vw, args):
        m = torch.mm(args.edge_mat(e_vw), torch.t(h_w))
        return m

    # Battaglia et al. (2016), Interaction Networks
    def m_intnet(self, h_v, h_w, e_vw, args):
        m = torch.cat([h_v, h_w, e_vw], 0)
        # TODO NN taking m
        return m

    # Kearnes et al. (2016), Molecular Graph Convolutions
    def m_mgc(self, h_v, h_w, e_vw, args):
        m = e_vw
        return m
    
    # Laplacian based methods
    # Bruna et al. (2013)
    def m_bruna(self, h_v, h_w, e_vw, args):
        # TODO
        m = [] 
        return m

    # Defferrard et al. (2016)
    def m_deff(self, h_v, h_w, e_vw, args):
        # TODO
        m = []
        return m

    # Kipf & Welling (2016)
    def m_kipf(self, h_v, h_w, e_vw, args):
        # TODO
        m = []
        return m


if __name__ == '__main__':
    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='QM9 Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['./data/qm9/dsgdb9nsd/'])

    args = parser.parse_args()
    root = args.root[0]

    files = [f for f in os.listdir(root) if os.path.isfile(os.path.join(root, f))]

    idx = np.random.permutation(len(files))
    idx = idx.tolist()

    valid_ids = [files[i] for i in idx[0:10000]]
    test_ids  = [files[i] for i in idx[10000:20000]]
    train_ids = [files[i] for i in idx[20000:]]

    data_train = datasets.Qm9(root, train_ids)
    data_valid = datasets.Qm9(root, valid_ids)
    data_test = datasets.Qm9(root, test_ids)

    # Define message
    m = MessageFunction('duvenaud')

    print(m.get_definition())

    start = time.time()

    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    m_t = {}
    for v in g.nodes_iter():
        neigh = g.neighbors(v)
        m_neigh = type(h_t)
        for w in neigh:
            if (v,w) in e:
                e_vw = e[(v, w)]
            else:
                e_vw = e[(w, v)]
            m_v = m.forward(h_t[v], h_t[w], e_vw)
            if len(m_neigh):
                m_neigh += m_v
            else:
                m_neigh = m_v

        m_t[v] = m_neigh

    end = time.time()

    print('Input nodes')
    print(h_t)
    print('Message')
    print(m_t)
    print('Time')
    print(end - start)