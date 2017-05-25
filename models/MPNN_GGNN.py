#!/usr/bin/python
# -*- coding: utf-8 -*-

from MessageFunction import MessageFunction
from UpdateFunction import UpdateFunction
from ReadoutFunction import ReadoutFunction

import torch
import torch.nn as nn
from torch.autograd import Variable

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class MpnnGGNN(nn.Module):
    """
        in_n (size_v, size_e)
    """

    def __init__(self, d, in_n, hidden_state_size, message_size, n_layers, l_target, type='regression'):
        super(MpnnGGNN, self).__init__()

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

            h_t = Variable(torch.zeros(h[0].size(0), h[0].size(1), h[0].size(2)).type_as(h_in.data))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[0].forward(h[t][:, v, :], h[t], e[:, v, :])

                # Nodes without edge set message to 0
                m = g[:, v, :, None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Update
                h_t[:, v, :] = self.u[0].forward(h[t][:, v, :], m)

            # Delete virtual nodes
            h_t = (torch.sum(torch.abs(h_in), 2).expand_as(h_t) > 0).type_as(h_t)*h_t
            h.append(h_t.clone())

        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res