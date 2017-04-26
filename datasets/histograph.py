import torch.utils.data as data
import networkx as nx
from os.path import join
import numpy as np
import xml.etree.ElementTree as ET
import argparse

class HISTOGRAPH(data.Dataset):
    
    def __init__(self, root, flag='Train', transform=None, target_transform=None):
        
        self.root = root
        if flag == 'Train':
            set_file = join(root, 'Set/Train.txt')
        elif flag == 'Valid':
            set_file = join(root, 'Set/Valid.txt')
        else:
            set_file = join(root, 'Set/Test.txt')
        self.classes, self.files = self.read_2cols_set_files(set_file)
        self.ids = list(range(len(self.classes)))
        
    def __getitem__(self, index):
        graph_id = self.ids[index]
        graph = self.create_graph_gwhist(join(self.root, self.files[graph_id]))        
        target = self.classes[graph_id]
        
        return graph, target
        
    def __len__(self):
        return len(self.ids)
        
    def read_2cols_set_files(self, file):
    
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

    # Parse optios for downloading
    parser = argparse.ArgumentParser(description='HISTOGRAPH Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.', default=['/home/adutta/Workspace/Datasets/GWHistoGraphs'])

    args = parser.parse_args()
    root = args.root[0]

    histograph_train = HISTOGRAPH(root = root, flag = 'Train')
    histograph_valid = HISTOGRAPH(root = root, flag = 'Valid')
    histograph_test = HISTOGRAPH(root = root, flag = 'Test')
    
    print(histograph_train.files)
    print(histograph_valid.files)
    print(histograph_test.files)