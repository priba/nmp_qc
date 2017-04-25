#!/usr/bin/python                                                                                                                                                                                               
# -*- coding: utf-8 -*-

"""
    MessageFunction.py: Propagates a message depending on two nodes and their common edge.

    Usage:

"""

from __future__ import print_function

import torch

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat" 


class ReadoutFunction:

    # Constructor
    def __init__(self, readout_def='nn'):
        self.set_readout(readout_def)

    # Readout graph given node values at las layer
    def R(self, h_v):
        return self.r_function(hv)

    # Set a readout function
    def set_readout(self, readout_def):
        self.r_definition = readout_def.lower()

        self.r_function = {
                    'nn': self.r_dummy
                }.get(self.r_definition, None)
        if self.r_definition is None:
            print('WARNING!: Readout Function has not been set correctly\n\tIncorrect definition ' + set_readout_def)
            quit()
    
    # Get the name of the used readout function
    def get_definition(self):
        return self.r_definition

    ## Definition of various state of the art update functions

    # Dummy
    def r_dummy(self, h):
        return []

if __name__ == '__main__'
    pass
    # TODO
