# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 15:28:34 2017

"""

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

import numpy as np
import networkx as nx
import random
import getpass as gp

from os import listdir
from os.path import isfile, join

import xml.etree.ElementTree as ET

random.seed(2)
np.random.seed(2)

def load_dataset(directory, dataset, subdir_gwhist = '01_Keypoint' ):    
    
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
        
#        directory = '/home/adutta/Workspace/Datasets/GWHistoGraphs'                
        train_classes, train_files = read_2cols_set_files(join(directory,'Set/Train.txt'))
        test_classes, test_files = read_2cols_set_files(join(directory,'Set/Test.txt'))
        valid_classes, valid_files = read_2cols_set_files(join(directory,'Set/Valid.txt'))
        
        train_classes, valid_classes, test_classes = \
             create_numeric_classes(train_classes, valid_classes, test_classes)
        
        data_dir = join(directory, 'Data/Word_Graphs/01_Skew', subdir_gwhist)
        
        train_graphs = load_gwhist(data_dir, train_files)
        valid_graphs = load_gwhist(data_dir, valid_files)
        test_graphs = load_gwhist(data_dir, test_files)
        
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
            train_classes_[i] = ix + 1
        idx = [i for i, c in enumerate(valid_classes) if c == uniq_classes[ix]]
        for i in idx:
            valid_classes_[i] = ix + 1
        idx = [i for i, c in enumerate(test_classes) if c == uniq_classes[ix]]
        for i in idx:
            test_classes_[i] = ix + 1

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
    
    g = nx.parse_edgelist(edge_list, nodetype = int, data = (('weight',float),), delimiter=",")
    
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
        s = int(s.split('_')[1]) + 1
        t = edge.get('to')
        t = int(t.split('_')[1]) + 1
        g.add_edge(s,t)
        
    for i in range(1, g.number_of_nodes()+1):
        g.node[i]['labels'] = np.array(vl[i-1])
        
    return g
    
if __name__ == '__main__':
    
    user = gp.getuser()
    
    directory = '/home/' + user + '/Workspace/Datasets/Graphs'
    
    dataset = 'enzymes'
    print(dataset)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    dataset = 'mutag'
    print(dataset)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    dataset = 'MUTAG'
    print(dataset)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    dataset = 'ENZYMES'
    print(dataset)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    dataset = 'NCI1'
    print(dataset)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    dataset = 'NCI109'
    print(dataset)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    dataset = 'DD'
    print(dataset)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    directory = '/home/' + user + '/Workspace/Datasets/GWHistoGraphs'
    dataset = 'gwhist'
    
    subdir_gwhist = '01_Keypoint'
    print(subdir_gwhist)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset, subdir_gwhist)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    subdir_gwhist = '02_Grid-NNA'
    print(subdir_gwhist)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset, subdir_gwhist)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    subdir_gwhist = '03_Grid-MST'
    print(subdir_gwhist)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset, subdir_gwhist)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    subdir_gwhist = '04_Grid-DEL'
    print(subdir_gwhist)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset, subdir_gwhist)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    subdir_gwhist = '05_Projection'
    print(subdir_gwhist)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset, subdir_gwhist)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))
    
    subdir_gwhist = '06_Split'
    print(subdir_gwhist)
    train_graphs, train_classes, valid_graphs, valid_classes, test_graphs, test_classes = load_dataset(directory, dataset, subdir_gwhist)
    print(len(train_graphs), len(valid_graphs), len(test_graphs))