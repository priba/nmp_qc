import torch.utils.data as data
import os, sys
import argparse
import networkx as nx

reader_folder = os.path.realpath(os.path.abspath('../GraphReader'))
if reader_folder not in sys.path:
    sys.path.insert(1, reader_folder)

from GraphReader.graph_reader import read_cxl, create_graph_letter

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


class LETTER(data.Dataset):
    def __init__(self, root_path, subset, ids, classes, class_list):
        self.root = root_path
        self.subset = subset
        self.classes = classes
        self.ids = ids
        self.class_list = class_list

    def __getitem__(self, index):
        g = create_graph_letter(os.path.join(self.root, self.subset, self.ids[index]))
        target = self.classes[index]
        h = self.vertex_transform(g)
        g, e = self.edge_transform(g)
        target = self.target_transform(target)
        return (g, h, e), target

    def __len__(self):
        return len(self.ids)

    def target_transform(self, target):
        return [self.class_list.index(target)]

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
    parser = argparse.ArgumentParser(description='Letter Object.')
    # Optional argument
    parser.add_argument('--root', nargs=1, help='Specify the data directory.',
                        default=['/home/adutta/Workspace/Datasets/STDGraphs/Letter'])
    parser.add_argument('--subset', nargs=1, help='Specify the sub dataset.', default=['LOW'])

    args = parser.parse_args()
    root = args.root[0]
    subset = args.subset[0]

    train_classes, train_ids = read_cxl(os.path.join(root, subset, 'train.cxl'))
    test_classes, test_ids = read_cxl(os.path.join(root, subset, 'test.cxl'))
    valid_classes, valid_ids = read_cxl(os.path.join(root, subset, 'validation.cxl'))

    num_classes = len(list(set(train_classes + valid_classes + test_classes)))

    data_train = LETTER(root, subset, train_ids, train_classes, num_classes)
    data_valid = LETTER(root, subset, valid_ids, valid_classes, num_classes)
    data_test = LETTER(root, subset, test_ids, test_classes, num_classes)

    print(len(data_train))
    print(len(data_valid))
    print(len(data_test))

    for i in range(len(train_ids)):
        print(data_train[i])

    for i in range(len(valid_ids)):
        print(data_valid[i])

    for i in range(len(test_ids)):
        print(data_test[i])

    print(data_train[1])
    print(data_valid[1])
    print(data_test[1])
