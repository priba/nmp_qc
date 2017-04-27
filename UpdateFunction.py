#!/usr/bin/python                                                                                                                                                                                               
# -*- coding: utf-8 -*-

"""
    UpdateFunction.py: Updates the nodes using the previous state and the message.
    
    Usage:

"""

from __future__ import print_function

# Own modules
import datasets
from MessageFunction import MessageFunction

import numpy as np
import time
import os
import argparse
import torch

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat" 


class UpdateFunction:

    # Constructor
    def __init__(self, update_def='nn'):
        self.set_update(update_def)

    # Update node hv given message mv
    def U(self, h_v, m_v):
        return self.u_function(h_v, m_v)

    # Set update function
    def set_update(self, update_def):
        self.u_definition = update_def.lower()

        self.u_function = {
                    'duvenaud':   self.u_duvenaud
                }.get(self.u_definition, None)

        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    ## Definition of various state of the art update functions

    # Dummy
    def u_duvenaud(self, h_v, m_v, args):
        sigma(torch.mm(self.args[args.degree],m_v))
        return []

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
    u = UpdateFunction('duvenaud')

    print(m.get_definition())

    start = time.time()

    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    h_t1 = {}
    for v in g.nodes_iter():
        neigh = g.neighbors(v)
        m_neigh = torch.FloatTensor()
        for w in neigh:
            if (v, w) in e:
                e_vw = e[(v, w)]
            else:
                e_vw = e[(w, v)]
            if len(m_neigh):
                m_neigh += m.M(h_t[v], h_t[w], e_vw)
            else:
                m_neigh = m.M(h_t[v], h_t[w], e_vw)

        h_t1[v] = u.U(h_t[v], m_neigh)

    end = time.time()

    print('Input nodes')
    print(h_t)
    print('Message')
    print(h_t1)
    print('Time')
    print(end - start)