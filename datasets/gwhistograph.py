"""
mutag.py:

Usage:

"""

import torch.utils.data as data
import os, sys
import argparse

import datasets.utils as utils

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)
    
from GraphReader.graph_reader import read_2cols_set_files, create_numeric_classes, create_graph_gwhist

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

class GWHISTOGRAPH(data.Dataset):
    
    def __init__(self, root_path, subset, ids, classes, vertex_transform=utils.gwhist_nodes, edge_transform=utils.gwhist_edges,
                 target_transform=None):
        
        self.root = root_path
        self.subdir = 'Data/Word_Graphs/01_Skew'
        self.subset = subset
        self.classes = classes
        self.ids = ids
        self.vertex_transform = vertex_transform
        self.edge_transform = edge_transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):        
                
        g = create_graph_gwhist(os.path.join(self.root, self.subdir, self.subset, self.ids[index]))
        target = self.classes[index]
        
        h = []
        if self.vertex_transform is not None:
            h = self.vertex_transform(g)

        e = []
        if self.edge_transform is not None:
            g, e = self.edge_transform(g)
            g = g.adjacency_list()

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (g, h, e), target
        
    def __len__(self):
        return len(self.ids)
    
if __name__ == '__main__':

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='GWHISTOGRAPH Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['/home/adutta/Workspace/Datasets/GWHistoGraphs'])
    parser.add_argument('--subset', nargs=1, help='Specify the sub dataset.', default=['01_Keypoint'])

    args = parser.parse_args()
    root = args.root[0]
    subset = args.subset[0]
    
    train_classes, train_ids = read_2cols_set_files(os.path.join(root,'Set/Train.txt'))
    test_classes, test_ids = read_2cols_set_files(os.path.join(root,'Set/Test.txt'))
    valid_classes, valid_ids = read_2cols_set_files(os.path.join(root,'Set/Valid.txt'))
    
    train_classes, valid_classes, test_classes = create_numeric_classes(train_classes, valid_classes, test_classes)
    
    data_train = GWHISTOGRAPH(root, subset, train_ids, train_classes)
    data_valid = GWHISTOGRAPH(root, subset, valid_ids, valid_classes)
    data_test = GWHISTOGRAPH(root, subset, test_ids, test_classes)
    
    print(len(data_train))
    print(len(data_valid))
    print(len(data_test))
    
    print(data_train[1])
    print(data_valid[1])
    print(data_test[1])