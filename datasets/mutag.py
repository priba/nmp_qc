"""
mutag.py:

Usage:

"""
import networkx as nx

import torch.utils.data as data
import os, sys
import argparse

import datasets.utils as utils

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)

from GraphReader.graph_reader import divide_datasets

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

class MUTAG(data.Dataset):
    
    def __init__(self, root_path, ids, classes):
        
        self.root = root_path
        self.classes = classes
        self.ids = ids
        
    def __getitem__(self, index):

        #TODO: Manually have to check the convert_node_labels_to_integers function
        g = nx.convert_node_labels_to_integers(nx.read_graphml(os.path.join(self.root, self.ids[index])))

        target = self.classes[index]

        h = self.vertex_transform(g)

        g, e = self.edge_transform(g)

        target = self.target_transform(target)

        return (g, h, e), target
        
    def __len__(self):
        return len(self.ids)

    def vertex_transform(self, g):
        h = []
        for n, d in g.nodes_iter(data=True):
            h_t = []
            h_t.append(d['label'])
            h.append(h_t)
        return h

    def edge_transform(self, g):
        e = {}
        for n1, n2, d in g.edges_iter(data=True):
            e_t = []
            e_t.append(d['label'])
            e[(n1, n2)] = e_t
        return nx.to_numpy_matrix(g), e

    def target_transform(self, target):
        return [target]
    
if __name__ == '__main__':

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='MUTAG Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['/home/adutta/Workspace/Datasets/Graphs/MUTAG'])

    args = parser.parse_args()
    root = args.root[0]
    
    label_file = 'MUTAG.label'
    list_file = 'MUTAG.list'
    with open(os.path.join(root, label_file), 'r') as f:
        l = f.read()
        classes = [int(s) for s in l.split() if s.isdigit()]            
    with open(os.path.join(root, list_file), 'r') as f:
        files = f.read().splitlines()
        
    train_ids, train_classes, valid_ids, valid_classes, test_ids, test_classes = divide_datasets(files, classes)

    data_train = MUTAG(root, train_ids, train_classes)
    data_valid = MUTAG(root, valid_ids, valid_classes)
    data_test = MUTAG(root, test_ids, test_classes)
    
    print(len(data_train))
    print(len(data_valid))
    print(len(data_test))
    
    print(data_train[1])
    print(data_valid[1])
    print(data_test[1])