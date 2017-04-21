#!/usr/bin/python                                                                                    
# -*- coding: utf-8 -*-

"""
xyzGraphReader.py: Reads a graph file in XYZ-like format.

Usage:
 
"""

import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import numpy as np
# from rdkit.Chem.inchi import MolFromInchi 

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

def initGraph(prop):
    prop = prop.split()
    g_tag = prop[0]
    g_index = int(prop[1])
    g_A = float(prop[2])
    g_B = float(prop[3]) 
    g_C = float(prop[4]) 
    g_mu = float(prop[5])
    g_alpha = float(prop[6]) 
    g_homo = float(prop[7])
    g_lumo = float(prop[8]) 
    g_gap = float(prop[9])
    g_r2 = float(prop[10])
    g_zpve = float(prop[11]) 
    g_U0 = float(prop[12]) 
    g_U = float(prop[13])
    g_H = float(prop[14])
    g_G = float(prop[15])
    g_Cv = float(prop[16])

    return nx.Graph(tag=g_tag, index=g_index, A=g_A, B=g_B, C=g_C, mu=g_mu, alpha=g_alpha, homo=g_homo,
                    lumo=g_lumo, gap=g_gap, r2=g_r2, zpve=g_zpve, U0=g_U0, U=g_U, H=g_H, G=g_G, Cv=g_Cv)

def addAtom(g, prop):
    prop = prop.split()
    a_type = prop[0]
    a_coord = np.array(prop[1:4], dtype='|S4')
    a_coord = a_coord.astype(np.float)
    a_pc = float(prop[4])
    g.add_node(len(g.nodes()), type=a_type, coord=a_coord, pc=a_pc)
    return g


def xyzGraphReader(graph_file, verbose=False):

    with open(graph_file,'r') as f:
        # Number of atoms
        na = int(f.readline())

        # Graph properties
        properties = f.readline()
        g = initGraph(properties)

        # Atoms properties
        for i in range(na):
            atom_properties = f.readline()
            g = addAtom(g, atom_properties) 

        # Frequencies
        f.readline()

        # SMILES
        smiles = f.readline()
        smiles = smiles.split()
        smiles = smiles[0]
        
        m = Chem.MolFromSmiles(smiles)
        print na
        print m.GetNumAtoms()
        
        for i in xrange(0,m.GetNumAtoms()):
            atom_i = m.GetAtomWithIdx(i)

            # Atom type
            atom_i.GetSymbol()
            # Atomic number
            atom_i.GetAtomicNum()

            # Aromatic
            atom_i.GetIsAromatic()
            # Hybridization
            atom_i.GetHybridization()

            for j in xrange(0, m.GetNumAtoms()):
                e_ij = m.GetBondBetweenAtoms(i, j)
                if e_ij != None:
                    e_ij.GetBondType()

                    # Acceptor?
                    e_ij.GetEndAtomIdx()
                    atom_j = m.GetAtomWithIdx(j)
                    if i not in g:
                        g[i] = []
                        g[i].append( (e_ij, j) )

        # InChI
        inchis = f.readline()
        inchis = inchis.split()
        inchis = inchis[0]
        print inchis
        #m2 = MolFromInchi(inchis)

        #print m2.GetNumAtoms()
    if verbose:
        Draw.MolToFile(m, 'test.png')
    return g  
    
if __name__ == '__main__':
    
    #    graph_file = '../data/qm9/dsgdb9nsd/dsgdb9nsd_033462.xyz'
    
    graph_file = '../data/qm9/dsC702H10nsd/dsC7O2H10nsd_0001.xyz'
    g = xyzGraphReader(graph_file)
