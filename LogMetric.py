#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"

def average_error_ratio(pred, target):
    if type(pred) is not np.ndarray:
        pred = np.array(pred)
    if type(target) is not np.ndarray:
        target = np.array(target)       
        
    return np.mean(np.divide(np.abs(pred - target), target))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count