import torch.utils.data as data
import networkx as nx
from os.path import join
import numpy as np

class MUTAG(data.Dataset):
    
    def __init__(self, root, transform=None, target_transform=None):
        
        self.root = root        
        label_file = 'MUTAG.label'
        list_file = 'MUTAG.list'        
        label_file_path = join(root, 'MUTAG', label_file)
        list_file_path = join(root, 'MUTAG', list_file)        
        with open(label_file_path, 'r') as f:
            l = f.read()
            self.classes = [int(s) for s in l.split() if s.isdigit()]            
        with open(list_file_path, 'r') as f:
            self.files = f.read().splitlines()            
        self.ids = list(len(self.classes))
        
        # divide dataset into train, valid and test sets
        tr_idx = []
        va_idx = []
        te_idx = []        
        uc = list(set(self.classes))            
        for c in uc:
            idx = [i for i, x in enumerate(self.classes) if x == c]
            tr_idx += sorted(np.random.choice(idx, int(0.8*len(idx)), replace=False))
            va_idx += sorted(np.random.choice([x for x in idx if x not in tr_idx], int(0.1*len(idx)), replace=False))
            te_idx += sorted([x for x in idx if x not in tr_idx and x not in va_idx])
            
        self.train_ids = tr_idx
        self.valid_ids = va_idx
        self.test_ids = te_idx
        
    def __getitem__(self, index):
        graph_id = self.ids[index]
        graph = nx.read_graphml(join(self.root, self.files[graph_id]))
        target = self.classes[graph_id]
        
        return graph, target
        
    def __len__(self):
        return len(self.ids)