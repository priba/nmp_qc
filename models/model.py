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
    def __init__(self, d, in_n, out, l_target, type='regression'):
        super(NMP_Duvenaud, self).__init__()

        n_layers = len(out)

        # Define message 1 & 2
        self.m = nn.ModuleList([MessageFunction('duvenaud') for _ in range(n_layers)])

        # Define Update 1 & 2
        self.u = nn.ModuleList([UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[i].get_out_size(in_n[0], in_n[1]), 'out': out[0]}) if i == 0 else
                                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[i].get_out_size(out[i-1], in_n[1]), 'out': out[i]}) for i in range(n_layers)])

        # Define Readout
        self.r = ReadoutFunction('duvenaud',
                                 args={'layers': len(self.m) + 1,
                                       'in': [in_n[0] if i == 0 else out[i-1] for i in range(n_layers+1)],
                                       'out': out[n_layers-1],
                                       'target': l_target})

        self.type = type

    def forward(self, g, h_in, e):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):

            u_args = self.u[t].get_args()

            h_t = Variable(torch.Tensor(np.zeros((h_in.size(0), h_in.size(1), u_args['out']))).type(h[t].data.type()))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[t].forward(h[t][:, v], h[t], e[:, v, :])

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

                        torch.index_select(h_t, 0, ind)[:, v, :] = torch.squeeze(aux)
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

    def __init__(self, d, in_n, out, l_target, type='regression'):
        super(NMP_GGNN, self).__init__()

        n_layers = len(out)

        # Define message 1 & 2
        self.m = nn.ModuleList([MessageFunction('ggnn', args={'e_labels': d, 'in': in_n, 'out': in_n})
                                for _ in range(n_layers)])

        # Define Update 1 & 2
        self.u = nn.ModuleList([UpdateFunction('ggnn',
                                               args={'e_labels': d, 'in': self.m[i].get_out_size(in_n[0], in_n[1]),
                                                     'out': out[0]}) if i == 0 else
                                UpdateFunction('ggnn',
                                               args={'e_labels': d, 'in': self.m[i].get_out_size(out[i - 1], in_n[1]),
                                                     'out': out[i]}) for i in range(n_layers)])

        # Define Readout
        self.r = ReadoutFunction('ggnn',
                                 args={'layers': len(self.m) + 1,
                                       'in': [in_n[0] if i == 0 else out[i - 1] for i in range(n_layers)],
                                       'out': out[n_layers - 1],
                                       'target': l_target})

        self.type = type

    def forward(self, g, h_in, e):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):

            u_args = self.u[t].get_args()

            h_t = Variable(torch.Tensor(np.zeros((h_in.size(0), h_in.size(1), u_args['out']))).type(h[t].data.type()))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[t].forward(h[t][:, v], h[t], e[:, v, :])

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
                        aux = self.u[t].forward(torch.index_select(h[t].clone(), 0, ind)[:, v, :],
                                                torch.index_select(m.clone(), 0, ind), opt)

                        torch.index_select(h_t, 0, ind)[:, v, :] = torch.squeeze(aux)
                        ind = ind.data.cpu().numpy()
                        for j in range(len(ind)):
                            h_t[ind[j], v, :] = aux[j, :]

            h.append(h_t.clone())
        # Readout
        res = self.r.forward(h)
        if self.type == 'classification':
            res = nn.Softmax()(res)
        return res


class Nmp1(nn.Module):
    def __init__(self, d, in_n, out, l_target):
        super(Nmp1, self).__init__()

        # Define message 1 & 2
        self.m = nn.ModuleList([
                MessageFunction('duvenaud'),
                MessageFunction('duvenaud'),
                MessageFunction('duvenaud')
            ])

        # Define Update 1 & 2
        self.u = nn.ModuleList([
                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[0].get_out_size(in_n[0], in_n[1]), 'out': out[0]}),
                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[1].get_out_size(out[0], in_n[1]), 'out': out[1]}),
                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[2].get_out_size(out[1], in_n[1]), 'out': out[2]})
            ])

        # Define Readout
        self.r = ReadoutFunction('duvenaud',
                                 args={'layers': len(self.m) + 1, 'in': [in_n[0], out[0], out[1], out[2]], 'out': out[3],
                                       'target': l_target})

    def forward(self, g, h_in, e):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):

            u_args = self.u[t].get_args()

            h_t = Variable(torch.Tensor(np.zeros((h_in.size(0), h_in.size(1), u_args['out']))).type(h[t].data.type()))

            # Apply one layer pass (Message + Update)
            for v in range(0, h_in.size(1)):

                m = self.m[t].forward(h[t][:, v], h[t], e[:,v,:])

                # Nodes without edge set message to 0
                m = g[:, v, :, None].expand_as(m) * m

                m = torch.sum(m, 1)

                # Duvenaud
                deg = torch.sum(g[:, v, :].data, 1)

                for i in range(len(u_args['deg'])):
                    ind = deg == u_args['deg'][i]
                    # ind = torch.squeeze(torch.nonzero(torch.squeeze(ind_binary)))
                    ind = Variable(torch.squeeze(torch.nonzero(torch.squeeze(ind))))
                    # ind = torch.squeeze(torch.nonzero(torch.squeeze(ind)))

                    opt = {'deg': i}

                    # Separate degrees
                    # Update
                    if len(ind) != 0:
                        aux = self.u[t].forward(torch.index_select(h[t].clone(), 0, ind)[:, v, :], torch.index_select(m.clone(), 0, ind), opt)
                        # aux = self.u[t].forward(h[t][:, v, :].clone(), m.clone(), opt)
                        # aux = torch.mul(ind_binary[..., None].expand_as(aux).float(), aux.data)
                        torch.index_select(h_t, 0, ind)[:, v, :] = torch.squeeze(aux)
                        ind = ind.data.cpu().numpy()
                        for j in range(len(ind)):
                           h_t[ind[j], v, :] = aux[j, :]

            h.append(h_t.clone())

        # Readout
        return nn.Softmax()(self.r.forward(h))
