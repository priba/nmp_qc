#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import os
from tensorboard_logger import configure, log_value

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


def error_ratio(pred, target):
    if type(pred) is not np.ndarray:
        pred = np.array(pred)
    if type(target) is not np.ndarray:
        target = np.array(target)       
        
    return np.mean(np.divide(np.abs(pred - target), np.abs(target)))


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


class Logger(object):
    def __init__(self, log_dir):
        if not os.path.isdir(log_dir):
            # if the directory does not exist we create the directory
            os.makedirs(log_dir)
        else:                      
            # clean previous logged data under the same directory name
            self._remove(log_dir)

        # configure the project
        configure(log_dir)

        self.global_step = 0

    def log_value(self, name, value):
        log_value(name, value, self.global_step)
        return self

    def step(self):
        self.global_step += 1

    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)  # remove dir and all contains
