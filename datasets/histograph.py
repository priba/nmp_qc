import torch.utils.data as data
from os.path import join
import argparse

import os
import sys

reader_folder = os.path.realpath( os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)

from GraphReader.graph_reader import create_graph_gwhist, read_2cols_set_files

class HISTOGRAPH(data.Dataset):
    
    def __init__(self, root, flag='Train', transform=None, target_transform=None):
        
        self.root = root
        if flag == 'Train':
            set_file = join(root, 'Set/Train.txt')
        elif flag == 'Valid':
            set_file = join(root, 'Set/Valid.txt')
        else:
            set_file = join(root, 'Set/Test.txt')
        self.classes, self.files = read_2cols_set_files(set_file)
        self.ids = list(range(len(self.classes)))
        
    def __getitem__(self, index):
        graph_id = self.ids[index]
        graph = create_graph_gwhist(join(self.root, self.files[graph_id]))        
        target = self.classes[graph_id]
        
        return graph, target
        
    def __len__(self):
        return len(self.ids)
        
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