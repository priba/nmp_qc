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


class MpnnDuvenaud(nn.Module):
    """
        MPNN as proposed by Duvenaud et al..

        This class implements the whole Duvenaud et al. model following the functions proposed by Gilmer et al. as 
        Message, Update and Readout.

        Parameters
        ----------
        d : int list.
            Possible degrees for the input graph.
        in_n : int list
            Sizes for the node and edge features.
        out_update : int list
            Output sizes for the different Update functions.
        hidden_state_readout : int
            Input size for the neural net used inside the readout function.
        l_target : int
            Size of the output.
        type : str (Optional)
            Classification | [Regression (default)]. If classification, LogSoftmax layer is applied to the output vector.
    """

    def __init__(self, d, in_n, out_update, hidden_state_readout, l_target, type='regression'):
        super(MpnnDuvenaud, self).__init__()

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

    def forward(self, g, h_in, e, plotter=None):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):

            u_args = self.u[t].get_args()

            h_t = Variable(torch.zeros(h_in.size(0), h_in.size(1), u_args['out']).type_as(h[t].data))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[t].forward(h[t][:, v, :], h[t], e[:, v, :])

                # Nodes without edge set message to 0
                m = g[:, v, :, None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Duvenaud
                deg = torch.sum(g[:, v, :].data, 1)

                # Separate degrees
                for i in range(len(u_args['deg'])):
                    ind = deg == u_args['deg'][i]
                    ind = Variable(torch.squeeze(torch.nonzero(torch.squeeze(ind))), volatile=True)

                    opt = {'deg': i}

                    # Update
                    if len(ind) != 0:
                        aux = self.u[t].forward(torch.index_select(h[t], 0, ind)[:, v, :], torch.index_select(m, 0, ind), opt)

                        ind = ind.data.cpu().numpy()
                        for j in range(len(ind)):
                            h_t[ind[j], v, :] = aux[j, :]

            if plotter is not None:
                num_feat = h_t.size(2)
                color = h_t[0,:,:].data.cpu().numpy()
                for i in range(num_feat):
                    plotter(color[:, i], 'layer_' + str(t) + '_element_' + str(i) + '.png')

            h.append(h_t.clone())
        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.LogSoftmax()(res)
        return res
