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
    def __init__(self, update_def='nn'):
        self.set_update(update_def)

    # Update node hv given message mv
    def U(self, h_v, m_v):
        return self.u_function(h_v, m_v)

    # Set update function
    def set_update(self, update_def):
        self.u_definition = update_def.lower()

        self.u_function = {
                    'dummy':   self.u_dummy
                }.get(self.u_definition, None)
        
        if self.u_function is None:
            print('WARNING!: Update Function has not been set correctly\n\tIncorrect definition ' + update_def)

    # Get the name of the used update function
    def get_definition(self):
        return self.u_definition

    ## Definition of various state of the art update functions

    # Dummy
    def u_dummy(self, h_v, m_v, args):
        return []

if __name__ == '__main__':
    pass
    # TODO
