#!/usr/bin/python
# -*- coding: utf-8 -*-

from MessageFunction import MessageFunction
from UpdateFunction import UpdateFunction
from ReadoutFunction import ReadoutFunction

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

dtype = torch.FloatTensor


class NMP_Duvenaud(nn.Module):
    """
    in_n (size_v, size_e)
    """
    def __init__(self, d, in_n, out_update, hidden_state_readout, l_target, type='regression'):
        super(NMP_Duvenaud, self).__init__()

        n_layers = len(out_update)

        # Define message 1 & 2
        self.m = nn.ModuleList([MessageFunction('duvenaud') for _ in range(n_layers)])

        # Define Update 1 & 2
        self.u = nn.ModuleList([UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[i].get_out_size(in_n[0], in_n[1]), 'out': out_update[0]}) if i == 0 else
                                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[i].get_out_size(out_update[i-1], in_n[1]), 'out': out_update[i]}) for i in range(n_layers)])

        # Define Readout
        self.r = ReadoutFunction('duvenaud',
                                 args={'layers': len(self.m) + 1,
                                       'in': [in_n[0] if i == 0 else out_update[i-1] for i in range(n_layers+1)],
                                       'out': hidden_state_readout,
                                       'target': l_target})

        self.type = type

    def forward(self, g, h_in, e):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):

            u_args = self.u[t].get_args()

            h_t = Variable(torch.Tensor(np.zeros((h_in.size(0), h_in.size(1), u_args['out']))).type_as(h[t].data))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[t].forward(h[t][:, v, :], h[t], e[:, v, :])

                # Nodes without edge set message to 0
                m = g[:, v, :, None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Duvenaud
                deg = torch.sum(g[:, v, :].data, 1)

                for i in range(len(u_args['deg'])):
                    ind = deg == u_args['deg'][i]
                    ind = Variable(torch.squeeze(torch.nonzero(torch.squeeze(ind))))

                    opt = {'deg': i}

                    # Separate degrees
                    # Update
                    if len(ind) != 0:
                        aux = self.u[t].forward(torch.index_select(h[t].clone(), 0, ind)[:, v, :], torch.index_select(m.clone(), 0, ind), opt)

                        ind = ind.data.cpu().numpy()
                        for j in range(len(ind)):
                           h_t[ind[j], v, :] = aux[j, :]

            h.append(h_t.clone())
        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.Softmax()(res)
        return res


class NMP_GGNN(nn.Module):
    """
        in_n (size_v, size_e)
    """

    def __init__(self, d, in_n, hidden_state_size, message_size, n_layers, l_target, type='regression'):
        super(NMP_GGNN, self).__init__()

        # Define message
        self.m = nn.ModuleList([MessageFunction('ggnn', args={'e_label': d, 'in': hidden_state_size, 'out': message_size})])

        # Define Update
        self.u = nn.ModuleList([UpdateFunction('ggnn',
                                                args={'in_m': message_size,
                                                'out': hidden_state_size})])

        # Define Readout
        self.r = ReadoutFunction('ggnn',
                                 args={'in': hidden_state_size,
                                       'target': l_target})

        self.type = type

        self.args = {}
        self.args['out'] = hidden_state_size

        self.n_layers = n_layers

    def forward(self, g, h_in, e):

        h = []

        # Padding to some larger dimension d
        h_t = torch.cat([h_in, Variable(torch.Tensor(h_in.size(0), h_in.size(1), self.args['out'] - h_in.size(2)).type_as(h_in.data).zero_())], 2)

        h.append(h_t.clone())

        # Layer
        for t in range(0, self.n_layers):

            h_t = Variable(torch.Tensor(h[0].size(0), h[0].size(1), h[0].size(2)).type_as(h_in.data).zero_())

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[0].forward(h[t][:, v, :], h[t], e[:, v, :])

                # Nodes without edge set message to 0
                m = g[:, v, :, None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Update
                h_t[:, v, :] = self.u[0].forward(h[t][:, v, :], m)

            # Delete virtual nodes
            h_t = (torch.sum(h_in, 2).expand_as(h_t) > 0).type_as(h_t)*h_t
            h.append(h_t.clone())

        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.Softmax()(res)
        return res


class NMP_IntNet(nn.Module):
    """
    in_n (size_v, size_e)
    """
    def __init__(self, in_n, out_message, out_update, l_target, type='regression'):
        super(NMP_IntNet, self).__init__()

        n_layers = len(out_update)

        # Define message 1 & 2
        self.m = nn.ModuleList([MessageFunction('intnet', args={'in': 2*in_n[0] + in_n[1], 'out': out_message[i]})
                                if i == 0 else
                                MessageFunction('intnet', args={'in': out_update[i-1], 'out': out_message[i]})
                                for i in range(n_layers)])

        # Define Update 1 & 2
        self.u = nn.ModuleList([UpdateFunction('intnet', args={'in': out_message[i], 'out': out_update[i]})
                                for i in range(n_layers)])

        # Define Readout
        self.r = ReadoutFunction('intnet', args={'in': out_update[-1], 'target': l_target})

        self.type = type

    def forward(self, g, h_in, e):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):

            u_args = self.u[t].get_args()
            h_t = Variable(torch.Tensor(np.zeros((h_in.size(0), h_in.size(1), u_args['out']))).type_as(h[t].data))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                for w in range(h_in.size(1)):

                    m = self.m[t].forward(h[t][:, v, :], h[t][:, w, :], e[:, v, w, :]).cuda()

                    # Nodes without edge set message to 0
                    m = g[:, v, :, None].expand_as(m) * m

                    m = torch.sum(m, 1)

                    # Interaction Net
                    opt = {}
                    opt['x_v'] = []

                    h_t[:, v, :] = self.u[t].forward(h[t][:, v, :], m, opt)

            h.append(h_t.clone())

        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.Softmax()(res)
        return res
