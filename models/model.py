#!/usr/bin/python
# -*- coding: utf-8 -*-

from MessageFunction import MessageFunction
from UpdateFunction import UpdateFunction
from ReadoutFunction import ReadoutFunction

import torch
import torch.nn as nn

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

dtype = torch.FloatTensor

class Nmp(nn.Module):
    def __init__(self, d, in_n, out, l_target):
        super(Nmp, self).__init__()

        # Define message 1 & 2
        self.m = nn.ModuleList([
                MessageFunction('duvenaud'),
                MessageFunction('duvenaud')
            ])

        # Define Update 1 & 2
        self.u = nn.ModuleList([
                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[0].get_out_size(in_n[0], in_n[1]), 'out': out[0]}),
                UpdateFunction('duvenaud', args={'deg': d, 'in': self.m[0].get_out_size(out[0], in_n[1]), 'out': out[1]})
            ])

        # Define Readout
        self.r = ReadoutFunction('duvenaud',
                                 args={'layers': len(self.m) + 1, 'in': [in_n[0], out[0], out[1]], 'out': out[2],
                                       'target': l_target})

    def forward(self, g, h_in, e):

        h = []
        h.append(h_in)

        # Layer
        for t in range(0, len(self.m)):
            h_t = []
            for v in range(0, len(g)):
                neigh = g[v]
                m_neigh = []
                for w in neigh:
                    if (v, w) in e:
                        e_vw = e[(v, w)]
                    else:
                        e_vw = e[(w, v)]
                    m_neigh.append(self.m[t].forward(h[t][v], h[t][w], e_vw))
                m_neigh = torch.squeeze(torch.sum(torch.stack(m_neigh, 0),0))
                # Duvenaud
                opt = {'deg': len(neigh)}
                h_t.append(self.u[t].forward(h[t][v], m_neigh, opt))
            h.append(torch.stack(h_t, 0))
        # Readout
        return self.r.forward(h)