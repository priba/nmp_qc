#!/usr/bin/python                                                                                                                                                                                               
# -*- coding: utf-8 -*-

"""
    UpdateFunction.py: Updates the nodes using the previous state and the message.
    
    Usage:

"""

from __future__ import print_function

# Own modules
import datasets
from datasets import utils
from MessageFunction import MessageFunction

import numpy as np
import time
import os
import argparse
import torch

import torch.nn as nn
import torch.nn.functional as F

#dtype = torch.cuda.FloatTensor
dtype = torch.FloatTensor

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat" 


class UpdateFunction(nn.Module):

    # Constructor
    def __init__(self, update_def='nn', args={}):
        super(UpdateFunction, self).__init__()
        self.u_definition = ''
        self.u_function = None
        self.args = {}
        self.__set_update(update_def, args)

    # Update node hv given message mv
    def forward(self, h_v, m_v, opt={}):
        return self.u_function(h_v, m_v, opt)

    # Set update function
    def __set_update(self, update_def, args):
        self.u_definition = update_def.lower()

        self.u_function = {
                    'duvenaud':   self.u_duvenaud,
                    'ggnn':       self.u_ggnn
                }.get(self.u_definition, None)

        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

        init_parameters = {
            'duvenaud':     self.init_duvenaud,
            'ggnn':         self.init_ggnn
        }.get(self.u_definition, lambda x: (nn.ParameterList([]), nn.ModuleList([]), {}))

        self.learn_args, self.learn_modules, self.args = init_parameters(args)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    # Get the name of the used update function
    def get_args(self):
        return self.args

    ## Definition of various state of the art update functions

    # Duvenaud
    def u_duvenaud(self, h_v, m_v, opt):

        param_sz = self.learn_args[0][opt['deg']].size()
        parameter_mat = torch.t(self.learn_args[0][opt['deg']])[None, ...].expand(m_v.size(0), param_sz[1], param_sz[0])

        aux = torch.bmm(parameter_mat, torch.transpose(m_v, 1, 2))

        return torch.transpose(torch.nn.Sigmoid()(aux), 1, 2)

    def init_duvenaud(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['deg'] = params['deg']
        args['out'] = params['out']

        # Define a parameter matrix H for each degree.
        learn_args.append(torch.nn.Parameter(torch.randn(len(params['deg']), params['in'], params['out'])))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

        # GG-NN, Li et al.
    def u_ggnn(self, h_v, m_v, opt):
        return self.learn_modules[0](m_v, F.pad(h_v, pad=(0, 0, self.args['padding'])))[0]

    def init_ggnn(self, params):
        learn_args = []
        learn_modules = []
        args = {}

        args['padding'] = params['padding']

        # GRU
        learn_modules.append(nn.GRU(params['input_sz'], params['hidden_sz'], batch_first=True))

        return nn.ParameterList(learn_args), nn.ModuleList(learn_modules), args

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

    print('STATS')
    # d = datasets.utils.get_graph_stats(data_test, 'degrees')
    d = [1, 2, 3, 4]

    print('Message')
    ## Define message
    m = MessageFunction('duvenaud')

    ## Parameters for the update function
    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    m_v = m.forward(h_t[0], h_t[1], e[list(e.keys())[0]])
    in_n = len(m_v)
    out_n = 30

    print('Update')
    ## Define Update
    u = UpdateFunction('duvenaud', args={'deg': d, 'in': in_n , 'out': out_n})

    print(m.get_definition())
    print(u.get_definition())

    start = time.time()

    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    h_t1 = {}
    for v in g.nodes_iter():
        neigh = g.neighbors(v)
        m_neigh = dtype()
        for w in neigh:
            if (v, w) in e:
                e_vw = e[(v, w)]
            else:
                e_vw = e[(w, v)]
            m_v = m.forward(h_t[v], h_t[w], e_vw)
            if len(m_neigh):
                m_neigh += m_v
            else:
                m_neigh = m_v

        # Duvenaud
        opt = {'deg': len(neigh)}
        h_t1[v] = u.forward(h_t[v], m_neigh, opt)

    end = time.time()

    print('Input nodes')
    print(h_t)
    print('Message')
    print(h_t1)
    print('Time')
    print(end - start)