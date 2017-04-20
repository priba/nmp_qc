#!/usr/bin/python                                                                                    
# -*- coding: utf-8 -*-

"""
xyzGraphReader.py: Reads a graph file in XYZ-like format.

Usage:
 
"""

import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

def xyzGraphReader(graph_file):
    with open(graph_file,'r') as f:
        na = int(f.readline())
        print na
	for i in range(na + 2):
            f.readline()
        smiles = f.readline()
        smiles = smiles.split()
	smiles = smiles[0]
    #for line in f:
    #	print line 
    m = Chem.MolFromSmiles(smiles)
    Draw.MolToFile(m, 'test.png')
    return nx.Graph()  
    
if __name__ == '__main__':
    
    graph_file = '../data/qm9/dsgdb9nsd/dsgdb9nsd_033462.xyz'
    
    g = xyzGraphReader(graph_file)
