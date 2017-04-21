# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:28:34 2017

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

"""

#import deepchem as dc
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import networkx as nx

import random

import os
from os import listdir
from os.path import isfile, join



random.seed(2)
torch.manual_seed(2)
np.random.seed(2)

T = 4
BATCH_SIZE = 64
MAXITER = 2000

def load_dataset(directory, dataset):
    
    file_path = join(directory, dataset)        
    files = [f for f in listdir(file_path) if isfile(join(file_path, f))]    
    
    classes = []
    graphs = []
    
    if dataset == 'enzymes':    
        for i in range(len(files)):
            g, c = create_graph_enzymes(join(directory, dataset, files[i]))
            graphs += [g]
            classes += [c]
    elif dataset == 'mutag':
        for i in range(len(files)):
            g, c = create_graph_mutag(join(directory, dataset, files[i]))
            graphs += [g]
            classes += [c]
        
    return graphs, classes

def create_graph_enzymes(file):
    
    f = open(file, 'r')
    lines = f.read().splitlines()
    
    # get the indices of the vertext, adj list and class
    idx_vertex = lines.index("#v - vertex labels")
    idx_adj_list = lines.index("#a - adjacency list")
    idx_clss = lines.index("#c - Class")
    
    # node label    
    vl = [int(ivl) for ivl in lines[idx_vertex+1:idx_adj_list]]
    
    adj_list = lines[idx_adj_list+1:idx_clss]
    sources = list(range(1,len(adj_list)+1))

    for i in range(len(adj_list)):
        if not adj_list[i]:
            adj_list[i] = str(sources[i])
        else:
            adj_list[i] = str(sources[i])+","+adj_list[i]

    g = nx.parse_adjlist(adj_list, nodetype=int, delimiter=",")
    
    for i in range(1, g.number_of_nodes()+1):
        g.node[i]['labels'] = np.array(vl[i-1])
    
    c = int(lines[idx_clss+1])
    
    return g, c
    
def create_graph_mutag(file):
    
    f = open(file, 'r')
    lines = f.read().splitlines()
    
    # get the indices of the vertext, adj list and class
    idx_vertex = lines.index("#v - vertex labels")
    idx_edge = lines.index("#e - edge labels")
    idx_clss = lines.index("#c - Class")
    
    # node label
    vl = [int(ivl) for ivl in lines[idx_vertex+1:idx_edge]]
    
    edge_list = lines[idx_edge+1:idx_clss]
    
    g = nx.parse_edgelist(edge_list, nodetype = int, data = (('weight',float),), delimiter=",")
    
    for i in range(1, g.number_of_nodes()+1):
        g.node[i]['labels'] = np.array(vl[i-1])
    
    c = int(lines[idx_clss+1])
    
    return g, c
    
def readout(h):
    
    reads = map(lambda x: F.relu(R(h[x])), h.keys())
    readout = Variable(torch.zeros(1, 128))
    for read in reads:
        readout = readout + read
    return readout

def message_pass(g, h, k):
  #flow_delta = Variable(torch.zeros(1, 1))
  #h_t = Variable(torch.zeros(1, 1, 75))
    for v in g.keys():
        neighbors = g[v]
        for neighbor in neighbors:
            e_vw = neighbor[0]
            w = neighbor[1]
      #bond_type = e_vw.GetBondType()
      #A_vw = A[str(e_vw.GetBondType())]

            m_v = h[w]
            catted = torch.cat([h[v], m_v], 1)
      #gru_act, h_t = GRU(catted.view(1, 1, 150), h_t)
      
      # measure convergence
      #pdist = nn.PairwiseDistance(2)
      #flow_delta = flow_delta + torch.sum(pdist(gru_act.view(1, 75), h[v]))
      
      #h[v] = gru_act.view(1, 75)
            h[v] = U(catted)
    

        
#    g = OrderedDict({})
#    h = OrderedDict({})    
    

#def construct_multigraph(smile):
#  g = OrderedDict({})
#  h = OrderedDict({})
#
#  molecule = Chem.MolFromSmiles(smile)
#  for i in xrange(0, molecule.GetNumAtoms()):
#    atom_i = molecule.GetAtomWithIdx(i)
#    h[i] = Variable(torch.FloatTensor(dc.feat.graph_features.atom_features(atom_i))).view(1, 75)
#    for j in xrange(0, molecule.GetNumAtoms()):
#      e_ij = molecule.GetBondBetweenAtoms(i, j)
#      if e_ij != None:
#        atom_j = molecule.GetAtomWithIdx(j)
#        if i not in g:
#          g[i] = []
#          g[i].append( (e_ij, j) )
#
#  return g, h