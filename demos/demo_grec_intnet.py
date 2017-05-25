#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Trains a Neural Message Passing Model on various datasets. Methodology defined in:

    Gilmer, J., Schoenholz S.S., Riley, P.F., Vinyals, O., Dahl, G.E. (2017)
    Neural Message Passing for Quantum Chemistry.
    arXiv preprint arXiv:1704.01212 [cs.LG]
"""

# Torch
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import time
import argparse
import os
import sys

# Our Modules
reader_folder = os.path.realpath(os.path.abspath('..'))
if reader_folder not in sys.path:
    sys.path.append(reader_folder)
import datasets
from datasets import utils
from models.MPNN_IntNet import MpnnIntNet
from LogMetric import AverageMeter, Logger
from GraphReader.graph_reader import read_cxl

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

torch.multiprocessing.set_sharing_strategy('file_system')


# Parser check
def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]"%(x,))
    return x

# Argument parser
parser = argparse.ArgumentParser(description='Neural message passing')

parser.add_argument('--dataset', default='GREC', help='GREC')
parser.add_argument('--datasetPath', default='../data/GREC/', help='dataset path')
parser.add_argument('--logPath', default='../log/grec/intnet/', help='log path')
parser.add_argument('--plotLr', default=False, help='alolow plotting the data')
parser.add_argument('--plotPath', default='../plot/grec/intnet/', help='plot path')
parser.add_argument('--resume', default='../checkpoint/grec/intnet/',
                    help='path to latest checkpoint')
# Optimization Options
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='Input batch size for training (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--epochs', type=int, default=360, metavar='N',
                    help='Number of epochs to train (default: 360)')
parser.add_argument('--lr', type=lambda x: restricted_float(x, [1e-5, 0.5]), default=0.001, metavar='LR',
                    help='Initial learning rate [1e-5, 5e-4] (default: 1e-4)')
parser.add_argument('--lr-decay', type=lambda x: restricted_float(x, [.01, 1]), default=0.6, metavar='LR-DECAY',
                    help='Learning rate decay factor [.01, 1] (default: 0.6)')
parser.add_argument('--schedule', type=list, default=[0.1, 0.9], metavar='S',
                    help='Percentage of epochs to start the learning rate decay [0, 1] (default: [0.1, 0.9])')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
# i/o
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='How many batches to wait before logging training status')
# Accelerating
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')

best_acc1 = 0


def main():

    global args, best_acc1
    args = parser.parse_args()

    # Check if CUDA is enabled
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Load data
    root = args.datasetPath

    print('Prepare files')

    train_classes, train_ids = read_cxl(os.path.join(root, 'data/train.cxl'))
    test_classes, test_ids = read_cxl(os.path.join(root, 'data/test.cxl'))
    valid_classes, valid_ids = read_cxl(os.path.join(root, 'data/valid.cxl'))
    
    num_classes = len(list(set(train_classes + test_classes + valid_classes)))

    data_train = datasets.GREC(root, train_ids, train_classes)
    data_valid = datasets.GREC(root, valid_ids, valid_classes)
    data_test = datasets.GREC(root, test_ids, test_classes)
    
    # Define model and optimizer
    print('Define model')
    # Select one graph
    g_tuple, l = data_train[0]
    g, h_t, e = g_tuple

    #TODO: Need attention
    print('\tStatistics')
    stat_dict = {}
    stat_dict = datasets.utils.get_graph_stats(data_train, ['edge_labels'])

    # Data Loader
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=args.batch_size, shuffle=True, collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(data_valid,
                                               batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                               num_workers=args.prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=args.batch_size, collate_fn=datasets.utils.collate_g,
                                              num_workers=args.prefetch, pin_memory=True)

    print('\tCreate model')
    model = MpnnIntNet([len(h_t[0]), len(list(e.values())[0])], [15, 25, 20], [10, 20, 20], num_classes,
                       type='classification')

    print('Optimizer')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    criterion = nn.NLLLoss()

    evaluation = utils.accuracy

    print('Logger')
    logger = Logger(args.logPath)

    lr_step = (args.lr-args.lr*args.lr_decay)/(args.epochs*args.schedule[1] - args.epochs*args.schedule[0])

    # get the best checkpoint if available without training
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {}; accuracy {})".format(best_model_file, checkpoint['epoch'],
                                                                             best_acc1))
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    print('Check cuda')
    if args.cuda:
        print('\t* Cuda')
        model = model.cuda()
        criterion = criterion.cuda()

    # Epoch for loop
    for epoch in range(0, args.epochs):

        if epoch > args.epochs * args.schedule[0] and epoch < args.epochs * args.schedule[1]:
            args.lr -= lr_step
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, evaluation, logger)

        # evaluate on test set
        acc1 = validate(valid_loader, model, criterion, evaluation, logger)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        utils.save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_acc1': best_acc1,
                               'optimizer': optimizer.state_dict(), }, is_best=is_best, directory=args.resume)

        # Logger step
        logger.log_value('learning_rate', args.lr).step()

    # get the best checkpoint and test it with test set
    if args.resume:
        checkpoint_dir = args.resume
        best_model_file = os.path.join(checkpoint_dir, 'model_best.pth')
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        if os.path.isfile(best_model_file):
            print("=> loading best model '{}'".format(best_model_file))
            checkpoint = torch.load(best_model_file)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded best model '{}' (epoch {}; accuracy {})".format(best_model_file, checkpoint['epoch'],
                                                                             best_acc1))
        else:
            print("=> no best model found at '{}'".format(best_model_file))

    # For testing
    validate(test_loader, model, criterion, evaluation)


def train(train_loader, model, criterion, optimizer, epoch, evaluation, logger):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (g, h, e, target) in enumerate(train_loader):
        
        # Prepare input data
        target = torch.squeeze(target).type(torch.LongTensor)
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Measure data loading time
        data_time.update(time.time() - end)

        def closure():
            optimizer.zero_grad()

            # Compute output
            output = model(g, h, e)
            train_loss = criterion(output, target)

            acc = Variable(evaluation(output.data, target.data, topk=(1,))[0])

            # Logs
            losses.update(train_loss.data[0], g.size(0))
            accuracies.update(acc.data[0], g.size(0))
            # compute gradient and do SGD step
            train_loss.backward()
            return train_loss

        optimizer.step(closure)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0 and i > 0:
            
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.4f} ({acc.avg:.4f})'
                  .format(epoch, i, len(train_loader), batch_time=batch_time,
                          data_time=data_time, loss=losses, acc=accuracies))
                          
    logger.log_value('train_epoch_loss', losses.avg)
    logger.log_value('train_epoch_accuracy', accuracies.avg)

    print('Epoch: [{0}] Average Accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}; Avg Time x Batch {b_time.avg:.3f}'
          .format(epoch, acc=accuracies, loss=losses, b_time=batch_time))


def validate(val_loader, model, criterion, evaluation, logger=None):
    losses = AverageMeter()
    accuracies = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (g, h, e, target) in enumerate(val_loader):

        # Prepare input data
        target = torch.squeeze(target).type(torch.LongTensor)
        if args.cuda:
            g, h, e, target = g.cuda(), h.cuda(), e.cuda(), target.cuda()
        g, h, e, target = Variable(g), Variable(h), Variable(e), Variable(target)

        # Compute output
        output = model(g, h, e)

        # Logs
        test_loss = criterion(output, target)
        acc = Variable(evaluation(output.data, target.data, topk=(1,))[0])

        losses.update(test_loss.data[0], g.size(0))
        accuracies.update(acc.data[0], g.size(0))

    print(' * Average Accuracy {acc.avg:.3f}; Average Loss {loss.avg:.3f}'
          .format(acc=accuracies, loss=losses))

    if logger is not None:
        logger.log_value('test_epoch_loss', losses.avg)
        logger.log_value('test_epoch_accuracy', accuracies.avg)

    return accuracies.avg
    
if __name__ == '__main__':
    main()
