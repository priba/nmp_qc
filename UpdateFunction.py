#!/usr/bin/python                                                                                                                                                                                               
# -*- coding: utf-8 -*-

"""
    UpdateFunction.py: Updates the nodes using the previous state and the message.
    
    Usage:

"""

from __future__ import print_function


__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat" 

class UpdateFunction:

    # Constructor
    def __init__(self, update_def='nn')
        self.set_update(update_def)

    # Update node hv given message mv
    def U(self, h_v, mv):
        return self.u_function(h_v, m_v)

    # Set update function
    def set_update(self):
        pass

if __name__ == '__main__':
    pass
    # TODO
