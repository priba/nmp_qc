#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
    Plotter.py: Propagates a message depending on two nodes and their common edge.

    Usage:

"""

from __future__ import print_function

import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import matplotlib.cm as cm
import os
import warnings


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

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            g = nx.from_numpy_matrix(am)

            if position is None:
                position=nx.drawing.circular_layout(g)

            fig = plt.figure()

            if cls is None:
                cls='r'
            else:
                # Make a user-defined colormap.
                cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", ["r", "b"])

                # Make a normalizer that will map the time values from
                # [start_time,end_time+1] -> [0,1].
                cnorm = mcol.Normalize(vmin=0, vmax=1)

                # Turn these into an object that can be used to map time values to colors and
                # can be passed to plt.colorbar().
                cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
                cpick.set_array([])
                cls = cpick.to_rgba(cls)
                plt.colorbar(cpick, ax=fig.add_subplot(111))


            nx.draw(g, pos=position, node_color=cls, ax=fig.add_subplot(111))

            fig.savefig(os.path.join(self.plotdir, fig_name))
