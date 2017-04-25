#!/usr/bin/python                                                                                                                                                                                               
# -*- coding: utf-8 -*-

"""
    MessageFunction.py: Propagates a message depending on two nodes and their common edge.

    Usage:

"""

from __future__ import print_function

import numpy as np
import torch

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat" 

class MessageFunction:

    # Constructor
    def __init__(self, message_def='duvenaud'):
        self.m_function = set_message(message_def)
        self.m_definition = lower(message_def)
    
    # Message from h_v to h_w through e_vw
    def M(self, h_v, h_w, e_vw, args=None):
        return self.m_function(h_v, h_w, e_vw, args)

    # Set a message function
    def set_message(self, message_def):
        self.m_definition = lower(message_def)

        # TODO Check the correct funciton
        self.m_definition = m_duvenaud

    # Get the name of the used message function
    def get_definition(self):
        return self.m_definition

    ## Definition of various state of the art message functions
    
    # Duvenaud et al. (2015), Convolutional Networks for Learning Molecular Fingerprints
    def m_duvenaud(self, h_v, h_w, e_vw, args):
        m = torch.cat([h_w, e_vw] , 1)
        return m

    # Li et al. (2016), Gated Graph Neural Networks (GG-NN)
    def m_ggnn(self, h_v, h_w, e_vw, args)):
        m = torch.mm(args.edge_mat(e_vw), torch.t(h_w))
        return m


    # Battaglia et al. (2016), Interaction Networks
    

    # Kearnes et al. (2016), Molecular Graph Convolutions

    # Laplacian based methods
    # Bruna et al. (2013)

    # Defferrard et al. (2016)

    # Kipf & Welling (2016)

if __name__ == '__main__':

    # TODO
    # Read Graph
    # Apply message function

