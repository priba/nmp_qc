#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
graph_reader.py: Reads graph datasets.

Usage:

"""
import numpy as np
import networkx as nx
import random

import argparse

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

import os

from os import listdir
from os.path import isfile, join

import xml.etree.ElementTree as ET

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

random.seed(2)
np.random.seed(2)


def load_dataset(directory, dataset, subdir = '01_Keypoint' ):    
    
    if dataset == 'enzymes':
        
        file_path = join(directory, dataset)        
        files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        
        classes = []
        graphs = []
        
        for i in range(len(files)):
            g, c = create_graph_enzymes(join(directory, dataset, files[i]))
            graphs += [g]
            classes += [c]
            
        train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = divide_datasets(graphs, classes)
            
    elif dataset == 'mutag':
        
        file_path = join(directory, dataset)        
        files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        
        classes = []
        graphs = []
        
        for i in range(len(files)):
            g, c = create_graph_mutag(join(directory, dataset, files[i]))
            graphs += [g]
            classes += [c]
            
        train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = divide_datasets(graphs, classes)
        
    elif dataset == 'MUTAG' or dataset == 'ENZYMES' or dataset == 'NCI1' or \
    dataset == 'NCI109' or dataset == 'DD':
        
        label_file = dataset + '.label'
        list_file = dataset + '.list'
        
        label_file_path = join(directory, dataset, label_file)
        list_file_path = join(directory, dataset, list_file)
        
        with open(label_file_path, 'r') as f:
            l = f.read()
            classes = [int(s) for s in l.split() if s.isdigit()]
            
        with open(list_file_path, 'r') as f:
            files = f.read().splitlines()
            
        graphs = load_graphml(join(directory, dataset), files)        
        train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = divide_datasets(graphs, classes)
            
    elif dataset == 'gwhist':
                    
        train_classes, train_files = read_2cols_set_files(join(directory,'Set/Train.txt'))
        test_classes, test_files = read_2cols_set_files(join(directory,'Set/Test.txt'))
        valid_classes, valid_files = read_2cols_set_files(join(directory,'Set/Valid.txt'))
        
        train_classes, valid_classes, test_classes = \
             create_numeric_classes(train_classes, valid_classes, test_classes)
        
        data_dir = join(directory, 'Data/Word_Graphs/01_Skew', subdir)
        
        train_graphs = load_gwhist(data_dir, train_files)
        valid_graphs = load_gwhist(data_dir, valid_files)
        test_graphs = load_gwhist(data_dir, test_files)
        
    elif dataset == 'qm9':
        
        file_path = join(directory, dataset, subdir)
        files = [f for f in listdir(file_path) if isfile(join(file_path, f))]
        
        data_dir = join(directory, dataset, subdir)
        
        graphs , labels = load_qm9(data_dir, files)
        
        # TODO: Split into train, valid and test sets and class information
        idx = np.random.permutation(len(labels))

        valid_graphs = [graphs[i] for i in idx[0:10000]]
        valid_classes = [labels[i] for i in idx[0:10000]]
        test_graphs = [graphs[i] for i in idx[10000:20000]]
        test_classes = [labels[i] for i in idx[10000:20000]]
        train_graphs = [graphs[i] for i in idx[20000:]]
        train_classes = [labels[i] for i in idx[20000:]]
        
    return train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes


def create_numeric_classes(train_classes, valid_classes, test_classes):
    
    classes = train_classes + valid_classes + test_classes
    uniq_classes = sorted(list(set(classes)))
    train_classes_ = [0] * len(train_classes)
    valid_classes_ = [0] * len(valid_classes)
    test_classes_ = [0] * len(test_classes)
    for ix in range(len(uniq_classes)):
        idx = [i for i, c in enumerate(train_classes) if c == uniq_classes[ix]]
        for i in idx:
            train_classes_[i] = ix
        idx = [i for i, c in enumerate(valid_classes) if c == uniq_classes[ix]]
        for i in idx:
            valid_classes_[i] = ix
        idx = [i for i, c in enumerate(test_classes) if c == uniq_classes[ix]]
        for i in idx:
            test_classes_[i] = ix

    return train_classes_, valid_classes_, test_classes_        


def load_gwhist(data_dir, files):
    
    graphs = []
    for i in range(len(files)):
        g = create_graph_gwhist(join(data_dir, files[i]))
        graphs += [g]
 
    return graphs


def load_graphml(data_dir, files):
    
    graphs = []    
    for i in range(len(files)):
        g = nx.read_graphml(join(data_dir,files[i]))
        graphs += [g]
        
    return graphs


def load_qm9(data_dir, files):
    
    graphs = []
    labels = []
    for i in range(len(files)):
        g , l = xyz_graph_reader(join(data_dir, files[i]))
        graphs += [g]
        labels.append(l)
        
    return graphs, labels


def read_2cols_set_files(file):
    
    f = open(file, 'r')
    lines = f.read().splitlines()
    f.close()
    
    classes = []
    files = []
    for line in lines:        
        c, f = line.split(' ')[:2]
        classes += [c]
        files += [f + '.gxl']

    return classes, files


def read_cxl(file):
    files = []
    classes = []
    tree_cxl = ET.parse(file)
    root_cxl = tree_cxl.getroot()
    for f in root_cxl.iter('print'):
        files += [f.get('file')]
        classes += [f.get('class')]
    return classes, files


def divide_datasets(graphs, classes):
    
    uc = list(set(classes))
    tr_idx = []
    va_idx = []
    te_idx = []
    
    for c in uc:
        idx = [i for i, x in enumerate(classes) if x == c]
        tr_idx += sorted(np.random.choice(idx, int(0.8*len(idx)), replace=False))
        va_idx += sorted(np.random.choice([x for x in idx if x not in tr_idx], int(0.1*len(idx)), replace=False))
        te_idx += sorted(np.random.choice([x for x in idx if x not in tr_idx and x not in va_idx], int(0.1*len(idx)), replace=False))
            
    train_graphs = [graphs[i] for i in tr_idx]
    valid_graphs = [graphs[i] for i in va_idx]
    test_graphs = [graphs[i] for i in te_idx]
    train_classes = [classes[i] for i in tr_idx]
    valid_classes = [classes[i] for i in va_idx]
    test_classes = [classes[i] for i in te_idx]
    
    return train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes


def create_graph_enzymes(file):
    
    f = open(file, 'r')
    lines = f.read().splitlines()
    f.close()
    
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
    f.close()
    
    # get the indices of the vertext, adj list and class
    idx_vertex = lines.index("#v - vertex labels")
    idx_edge = lines.index("#e - edge labels")
    idx_clss = lines.index("#c - Class")
    
    # node label
    vl = [int(ivl) for ivl in lines[idx_vertex+1:idx_edge]]
    
    edge_list = lines[idx_edge+1:idx_clss]
    
    g = nx.parse_edgelist(edge_list, nodetype=int, data=(('weight', float),), delimiter=",")
    
    for i in range(1, g.number_of_nodes()+1):
        g.node[i]['labels'] = np.array(vl[i-1])
    
    c = int(lines[idx_clss+1])
    
    return g, c


def create_graph_gwhist(file):
    
    tree_gxl = ET.parse(file)
    root_gxl = tree_gxl.getroot()
    
    vl = []    
    
    for node in root_gxl.iter('node'):
        for attr in node.iter('attr'):
            if(attr.get('name') == 'x'):
                x = attr.find('float').text
            elif(attr.get('name') == 'y'):
                y = attr.find('float').text
        vl += [[x, y]]

    g = nx.Graph()                        
    
    for edge in root_gxl.iter('edge'):
        s = edge.get('from')
        s = int(s.split('_')[1])
        t = edge.get('to')
        t = int(t.split('_')[1])
        g.add_edge(s, t)
        
    for i in range(g.number_of_nodes()):
        if i not in g.node:
            g.add_node(i)
        g.node[i]['labels'] = np.array(vl[i])
        
    return g


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def create_graph_grec(file):

    tree_gxl = ET.parse(file)
    root_gxl = tree_gxl.getroot()
    vl = []
    switch_node = {'circle': 0, 'corner': 1, 'endpoint': 2, 'intersection': 3}
    switch_edge = {'arc': 0, 'arcarc': 1, 'line': 2, 'linearc': 3}
    for node in root_gxl.iter('node'):
        for attr in node.iter('attr'):
            if (attr.get('name') == 'x'):
                x = int(attr.find('Integer').text)
            elif (attr.get('name') == 'y'):
                y = int(attr.find('Integer').text)
            elif (attr.get('name') == 'type'):
                t = switch_node.get(attr.find('String').text, 4)
        vl += [[x, y, t]]
    g = nx.Graph()
    for edge in root_gxl.iter('edge'):
        s = int(edge.get('from'))
        t = int(edge.get('to'))
        for attr in edge.iter('attr'):
            if(attr.get('name') == 'frequency'):
                f = attr.find('Integer').text
            elif(attr.get('name') == 'type0'):
                ta = switch_edge.get(attr.find('String').text)
            elif (attr.get('name') == 'angle0'):
                a = attr.find('String').text
                if isfloat(a):
                    a = float(a)
                else:
                    a = 0.0     # TODO: The erroneous string is replaced with 0.0
        g.add_edge(s, t, frequency=f, type=ta, angle=a)

    for i in range(len(vl)):
        if i not in g.node:
            g.add_node(i)
        g.node[i]['labels'] = np.array(vl[i][:3])

    return g


def create_graph_letter(file):

    tree_gxl = ET.parse(file)
    root_gxl = tree_gxl.getroot()
    vl = []
    for node in root_gxl.iter('node'):
        for attr in node.iter('attr'):
            if (attr.get('name') == 'x'):
                x = float(attr.find('float').text)
            elif (attr.get('name') == 'y'):
                y = float(attr.find('float').text)
        vl += [[x, y]]
    g = nx.Graph()
    for edge in root_gxl.iter('edge'):
        s = int(edge.get('from').split('_')[1])
        t = int(edge.get('to').split('_')[1])
        g.add_edge(s, t)

    for i in range(len(vl)):
        if i not in g.node:
            g.add_node(i)
        g.node[i]['labels'] = np.array(vl[i][:2])

    return g


# Initialization of graph for QM9
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

    labels = [g_mu, g_alpha, g_homo, g_lumo, g_gap, g_r2, g_zpve, g_U0, g_U, g_H, g_G, g_Cv]
    return nx.Graph(tag=g_tag, index=g_index, A=g_A, B=g_B, C=g_C, mu=g_mu, alpha=g_alpha, homo=g_homo,
                    lumo=g_lumo, gap=g_gap, r2=g_r2, zpve=g_zpve, U0=g_U0, U=g_U, H=g_H, G=g_G, Cv=g_Cv), labels


# XYZ file reader for QM9 dataset
def xyz_graph_reader(graph_file):

    with open(graph_file,'r') as f:
        # Number of atoms
        na = int(f.readline())

        # Graph properties
        properties = f.readline()
        g, l = init_graph(properties)
        
        atom_properties = []
        # Atoms properties
        for i in range(na):
            a_properties = f.readline()
            a_properties = a_properties.replace('.*^', 'e')
            a_properties = a_properties.replace('*^', 'e')
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
        for i in range(0, m.GetNumAtoms()):
            atom_i = m.GetAtomWithIdx(i)

            g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                       aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                       num_h=atom_i.GetTotalNumHs(), coord=np.array(atom_properties[i][1:4]).astype(np.float),
                       pc=float(atom_properties[i][4]))

        for i in range(0, len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1

        # Read Edges
        for i in range(0, m.GetNumAtoms()):
            for j in range(0, m.GetNumAtoms()):
                e_ij = m.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType(),
                               distance=np.linalg.norm(g.node[i]['coord']-g.node[j]['coord']))
                else:
                    # Unbonded
                    g.add_edge(i, j, b_type=None,
                               distance=np.linalg.norm(g.node[i]['coord'] - g.node[j]['coord']))
    return g , l
    
if __name__ == '__main__':

    g1 = create_graph_grec('/home/adutta/Workspace/Datasets/Graphs/GREC/data/image1_1.gxl')

    g2 = create_graph_letter('/home/adutta/Workspace/Datasets/STDGraphs/Letter/LOW/AP1_0000.gxl')

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='Read the specified directory, dataset and subdirectory.')
    # Positional arguments
    parser.add_argument('--dataset', default='GREC', nargs=1, help='Specify a dataset.')
    # Optional argument
    parser.add_argument('--dir', nargs=1, help='Specify the data directory.', default=['../data/'])
    parser.add_argument('--subdir', nargs=1, help='Specify a subdirectory.')
    
    args = parser.parse_args()    
   
    directory = args.dir[0]
    dataset = args.dataset[0]
    
    if dataset == 'gwhist' or dataset == 'qm9':
        if args.subdir is None:
            print('Error: No subdirectory mentioned for the dataset')
            quit()
        else:
            subdir = args.subdir[0]
    else:
        subdir = []
        
    print(dataset)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory,
                                                                                                       dataset, subdir)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
