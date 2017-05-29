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


class MpnnIntNet(nn.Module):
    """
        MPNN as proposed by Battaglia et al..

        This class implements the whole Battaglia et al. model following the functions proposed by Gilmer et al. as
        Message, Update and Readout.

        Parameters
        ----------
        in_n : int list
            Sizes for the node and edge features.
        out_message : int list
            Output sizes for the different Message functions.
        out_update : int list
            Output sizes for the different Update functions.
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, in_n, out_message, out_update, l_target, type='regression'):
        super(MpnnIntNet, self).__init__()

        n_layers = len(out_update)

        # Define message 1 & 2
        self.m = nn.ModuleList([MessageFunction('intnet', args={'in': 2*in_n[0] + in_n[1], 'out': out_message[i]})
                                if i == 0 else
                                MessageFunction('intnet', args={'in': 2*out_update[i-1] + in_n[1], 'out': out_message[i]})
                                for i in range(n_layers)])

        # Define Update 1 & 2
        self.u = nn.ModuleList([UpdateFunction('intnet', args={'in': in_n[0]+out_message[i], 'out': out_update[i]})
                                if i == 0 else
                                UpdateFunction('intnet', args={'in': out_update[i-1]+out_message[i], 'out': out_update[i]})
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
            h_t = Variable(torch.zeros(h_in.size(0), h_in.size(1), u_args['out']).type_as(h[t].data))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[t].forward(h[t][:, v, :], h[t], e[:, v, :, :])

                # Nodes without edge set message to 0
                m = g[:, v, :,None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Interaction Net
                opt = {}
                opt['x_v'] = Variable(torch.Tensor([]).type_as(m.data))

                h_t[:, v, :] = self.u[t].forward(h[t][:, v, :], m, opt)

            h.append(h_t.clone())

        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res
