import torch.utils.data as data
import networkx as nx
from os.path import join

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
        
    def __getitem__(self, index):
        graph_id = self.ids[index]
        graph = nx.read_graphml(join(self.root, self.files[graph_id]))
        target = self.classes[graph_id]
        
        return graph, target
        
    def __len__(self):
        return len(self.ids)