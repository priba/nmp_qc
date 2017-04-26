#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    utils.py: Functions to process dataset graphs.

    Usage:

"""

from __future__ import print_function

import networkx as nx
import rdkit

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


def qm9_nodes(g, hydrogen=False):
    h = {}
    for n, d in g.nodes_iter(data=True):
        h_t = []
        # Atom type (One-hot H, C, N, O F)
        h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F']]
        # Atomic number
        h_t.append(d['a_num'])
        # Partial Charge
        h_t.append(d['pc'])
        # Acceptor
        h_t.append(d['acceptor'])
        # Donor
        h_t.append(d['donor'])
        # Aromatic
        h_t.append(int(d['aromatic']))
        # Hybradization
        h_t += [int(d['hybridization'] == x) for x in [rdkit.Chem.rdchem.HybridizationType.SP, rdkit.Chem.rdchem.HybridizationType.SP2, rdkit.Chem.rdchem.HybridizationType.SP3]]
        # If number hydrogen is used as a
        if hydrogen:
            h_t.append(d['num_h'])
        h[n] = h_t
    return h

def qm9_edges():
    return