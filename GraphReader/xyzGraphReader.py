#!/usr/bin/python                                                                                    
# -*- coding: utf-8 -*-

"""
xyzGraphReader.py: Reads a graph file in XYZ-like format.

Usage:
 
"""

from __future__ import print_function

import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import numpy as np
import os

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


def init_graph(prop):
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


def xyz_graph_reader(graph_file, verbose=False):

    with open(graph_file,'r') as f:
        # Number of atoms
        na = int(f.readline())

        # Graph properties
        properties = f.readline()
        g = init_graph(properties)
        
        atom_properties = []
        # Atoms properties
        for i in range(na):
            a_properties = f.readline()
            a_properties = a_properties.split()
            atom_properties.append(a_properties)

        # Frequencies
        f.readline()

        # SMILES
        smiles = f.readline()
        smiles = smiles.split()
        smiles = smiles[0]
        
        m = Chem.MolFromSmiles(smiles)
        m = Chem.AddHs(m)

        fdef_name = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
        factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)
        feats = factory.GetFeaturesForMol(m)

        # Create nodes
        for i in xrange(0, m.GetNumAtoms()):
            atom_i = m.GetAtomWithIdx(i)

            g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                       aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(), coord=np.array(atom_properties[i][1:4], dtype='|S4').astype(np.float),
                       pc=float(atom_properties[i][4]))

        for i in xrange(0, len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1

        # Read Edges
        for i in xrange(0, m.GetNumAtoms()):
            for j in xrange(0, m.GetNumAtoms()):
                e_ij = m.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType(),
                               distance=np.linalg.norm(g.node[i]['coord']-g.node[j]['coord']))

    return g
    
if __name__ == '__main__':
    
    #    graph_file = '../data/qm9/dsgdb9nsd/dsgdb9nsd_033462.xyz'
    
    graph_file = '../data/qm9/dsC702H10nsd/dsC7O2H10nsd_0001.xyz'
    g = xyz_graph_reader(graph_file)
