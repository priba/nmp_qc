#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Plotter.py: Propagates a message depending on two nodes and their common edge.

    Usage:

"""

from __future__ import print_function

import networkx as nx
import matplotlib

import matplotlib.pyplot as plt
import os

matplotlib.use("Agg")

__author__ = "Pau Riba, Anjan Dutta"
__email__ = "priba@cvc.uab.cat, adutta@cvc.uab.cat"


"""
   Plots a Graph with the library networkx
"""

class Plotter():
    # Constructor
    def __init__(self, plot_dir = './'):
        self.plotdir = plot_dir

        if os.path.isdir(plot_dir):
            # clean previous logged data under the same directory name
            self._remove(plot_dir)

        os.makedirs(plot_dir)


    @staticmethod
    def _remove(path):
        """ param <path> could either be relative or absolute. """
        if os.path.isfile(path):
            os.remove(path)  # remove the file
        elif os.path.isdir(path):
            import shutil
            shutil.rmtree(path)  # remove dir and all contains

    def plot_graph(self, am, position=None, cls=None, fig_name='graph.png'):

        g = nx.from_numpy_matrix(am)

        if position is None:
            position=nx.drawing.circular_layout(g)

        if cls is None:
            cls='r'

        fig = plt.figure()
        nx.draw(g, pos=position, node_color=cls, ax=fig.add_subplot(111))

        fig.savefig(os.path.join(self.plotdir, fig_name))
