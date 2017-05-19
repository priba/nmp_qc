"""
mutag.py:

Usage:

"""

import torch.utils.data as data
import os, sys
import argparse
import networkx as nx

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)
    
from GraphReader.graph_reader import read_2cols_set_files, create_numeric_classes, create_graph_gwhist

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class GWHISTOGRAPH(data.Dataset):
    
    def __init__(self, root_path, subset, ids, classes, max_class_num):
        
        self.root = root_path
        self.subdir = 'Data/Word_Graphs/01_Skew'
        self.subset = subset
        self.classes = classes
        self.ids = ids
        self.max_class_num = max_class_num
        
    def __getitem__(self, index):        
                
        g = create_graph_gwhist(os.path.join(self.root, self.subdir, self.subset, self.ids[index]))

        target = self.classes[index]

        h = self.vertex_transform(g)

        g, e = self.edge_transform(g)

        target = self.target_transform(target)

        return (g, h, e), target
        
    def __len__(self):
        return len(self.ids)
        
    def target_transform(self, target):
        # [int(i == target-1) for i in range(self.max_class_num)]
        # return target_one_hot
        return [target]

    def vertex_transform(self, g):
        h = []
        for n, d in g.nodes_iter(data=True):
            h_t = []
            h_t += [float(x) for x in d['labels']]
            h.append(h_t)
        return h

    def edge_transform(self, g):
        e = {}
        for n1, n2, d in g.edges_iter(data=True):
            e_t = []
            e_t += [1]
            e[(n1, n2)] = e_t
        return nx.to_numpy_matrix(g), e
    
if __name__ == '__main__':

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='GWHISTOGRAPH Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['/home/adutta/Workspace/Datasets/GWHistoGraphs'])
    parser.add_argument('--subset', nargs=1, help='Specify the sub dataset.', default=['01_Keypoint'])

    args = parser.parse_args()
    root = args.root[0]
    subset = args.subset[0]
    
    train_classes, train_ids = read_2cols_set_files(os.path.join(root, 'Set/Train.txt'))
    test_classes, test_ids = read_2cols_set_files(os.path.join(root, 'Set/Test.txt'))
    valid_classes, valid_ids = read_2cols_set_files(os.path.join(root, 'Set/Valid.txt'))
    
    train_classes, valid_classes, test_classes = create_numeric_classes(train_classes, valid_classes, test_classes)

    num_classes = max(train_classes + valid_classes + test_classes)
    
    data_train = GWHISTOGRAPH(root, subset, train_ids, train_classes, num_classes)
    data_valid = GWHISTOGRAPH(root, subset, valid_ids, valid_classes, num_classes)
    data_test = GWHISTOGRAPH(root, subset, test_ids, test_classes, num_classes)
    
    print(len(data_train))
    print(len(data_valid))
    print(len(data_test))
    
    print(data_train[1])
    print(data_valid[1])
    print(data_test[1])
