"""
mutag.py:

Usage:

"""
import networkx as nx

import torch.utils.data as data
from os.path import join
import argparse

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

class GWHISTOGRAPH(data.Dataset):
    
    def __init__(self, root, transform=None, target_transform=None):
        
        self.root = root        
        label_file = 'MUTAG.label'
        list_file = 'MUTAG.list'        
        label_file_path = join(root, label_file)
        list_file_path = join(root, list_file)        
        with open(label_file_path, 'r') as f:
            l = f.read()
            self.classes = [int(s) for s in l.split() if s.isdigit()]            
        with open(list_file_path, 'r') as f:
            self.files = f.read().splitlines()            
        self.ids = list(range(len(self.classes)))
        
    def __getitem__(self, index):
        graph_id = self.ids[index]
        graph = nx.read_graphml(join(self.root, self.files[graph_id]))
        target = self.classes[graph_id]
        
        return graph, target
        
    def __len__(self):
        return len(self.ids)
    
if __name__ == '__main__':

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='MUTAG Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['../data/data_graphml'])

    args = parser.parse_args()
    root = args.root[0]

    mutag = MUTAG(root = root)
    
    print(mutag.files)