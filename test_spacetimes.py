#!/usr/bin/env python
'''
Created on 05 Oct 2020

@author: Christoph Minz
'''
from __future__ import annotations
import unittest
import numpy as np
from spacetimes import Spacetime, FlatSpacetime, deSitterSpacetime
from matplotlib import pyplot as plt


class TestSpacetime(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_LightconePlotter(self):
        spacetime: Spacetime = deSitterSpacetime(3, 1.0)
        dims: List[int] = [1, 2, 0]
        plt.figure(figsize=(8.0, 8.0))
        pr: float = 1.1
        spacetime.DefaultShape().plot(dims, alpha=0.02, color='yellow')
        plotting_params = {'color': 'orange', 'facecolor': 'orange',
                           'alpha': 0.1, 'linewidth': 2.5}
        timesign: float = 1.0
        timeslice: float = 2.8
        startpoint: np.ndarray = np.array([0.0, 0.4, 0.4, 0.1])
        plotconeat = spacetime.LightconePlotter(plt.gca(), dims, plotting_params,
                                                timesign, timeslice)
        plotconeat(startpoint)
        timesign: float = -1.0
        timeslice: float = -0.9
        plotconeat = spacetime.LightconePlotter(plt.gca(), dims, plotting_params,
                                                timesign, timeslice)
        plotconeat(startpoint)
        if len(dims) == 3:
            plt.plot([startpoint[dims[0]]], [startpoint[dims[1]]],
                     [startpoint[dims[2]]], 'ok')
            plt.gca().set(xlim=[-pr, pr], ylim=[-pr, pr], zlim=[-pr, pr])
        else:
            plt.plot([startpoint[dims[0]]], [startpoint[dims[1]]], 'ok')
            plt.gca().set(xlim=[-pr, pr], ylim=[-pr, pr])
        plt.show()


if __name__ == '__main__':
    unittest.main()
