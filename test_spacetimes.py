#!/usr/bin/env python
'''
Created on 05 Oct 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
import unittest
import numpy as np
from causets.spacetimes import Spacetime  # @UnresolvedImport @UnusedImport
from causets.spacetimes import deSitterSpacetime  # @UnresolvedImport
from causets.shapes import CoordinateShape  # @UnresolvedImport
from matplotlib import pyplot as plt  # @UnresolvedImport


class TestSpacetime(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_ConePlotter(self):
        # plot parameters:
        a: float = 1.00
        spacetime: Spacetime = deSitterSpacetime(3, a)
        dims: List[int] = [1, 2]
        pr: float = 1.1 * a
        plotting_params = {'facecolor': 'orange',
                           'alpha': 0.3, 'linewidth': 2.5}
        # plot shape and cone:
        plt.figure(figsize=(8.0, 8.0))
        CoordinateShape(3, 'cylinder', radius=a).plot(dims, alpha=0.05)
        startpoint: np.ndarray = np.array([0.0, 0.1 * a, -0.6 * a])
        timeslice: float = 1.5
        plotconeat = spacetime.ConePlotter(dims, plotting_params, 1.0)
        plotconeat(startpoint, timeslice)
        plt.gca().set(xlim=[-pr, pr], ylim=[-pr, pr])
        plt.show()


if __name__ == '__main__':
    unittest.main()
