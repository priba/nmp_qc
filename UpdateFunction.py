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
    def __init__(self, update_def='nn', args=None):
        self.set_update(update_def, args)

    # Update node hv given message mv
    def U(self, h_v, m_v, opt):
        return self.u_function(h_v, m_v, opt)

    # Set update function
    def set_update(self, update_def, args):
        self.u_definition = update_def.lower()

        self.u_function = {
                    'duvenaud':   self.u_duvenaud
                }.get(self.u_definition, None)
        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

        self.args = {
                'duvenaud': self.init_duvenaud(args)
            }.get(self.u_definition, None)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    ## Definition of various state of the art update functions

    # Duvenaud
    def u_duvenaud(self, h_v, m_v, opt):
        return torch.nn.Sigmoid(torch.mm(self.args[opt['deg']], m_v))

    def init_duvenaud(self, params):
        args={}
        # Define a parameter matrix H for each degree.
        for d in params['deg']:
            args[d] = torch.nn.Parameter(torch.FloatTensor(params['in'], params['out']))
        return args

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

    # d = datasets.utils.get_graph_stats(data_test, 'degrees')
    d = [1,2,3,4]

    ## Define message
    m = MessageFunction('duvenaud')

    ## Parameters for the update function
    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    m_v = m.M(h_t[0], h_t[1], e[e.keys()[0]])
    in_n = len(m_v)

    ## Define Update
    u = UpdateFunction('duvenaud', args={'deg': d, 'in': in_n , 'out': 30})

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
            m_v = m.M(h_t[v], h_t[w], e_vw)
            if len(m_neigh):
                m_neigh += m_v
            else:
                m_neigh = m_v

        # Duvenaud
        opt = {'deg': len(neigh)}
        h_t1[v] = u.U(h_t[v], m_neigh, opt)

    end = time.time()

    print('Input nodes')
    print(h_t)
    print('Message')
    print(h_t1)
    print('Time')
    print(end - start)