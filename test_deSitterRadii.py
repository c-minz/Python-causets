#!/usr/bin/env python
'''
Created on 16 Aug 2021

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
import unittest
import numpy as np
from causets.spacetimes import Spacetime  # @UnresolvedImport @UnusedImport
from causets.spacetimes import deSitterSpacetime  # @UnresolvedImport
from causets.shapes import CircleEdge  # @UnresolvedImport
from matplotlib import pyplot as plt, patches, axes


class TestDeSitterSpacetime(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_RadiiFit(self):
        # plot parameters:
        a: float = 1.50
        spacetime: Spacetime = deSitterSpacetime(3, a)
        spacetime
        dims: List[int] = [1, 2]
        pr: float = 1.1 * a
        # plot cone origin and set axes limits:
        startpoint: np.ndarray = np.array([0.0, 0 * a, 0 * a])
        plt.figure(figsize=(8.0, 8.0))
        A: axes
        p: patches.Patch
        A = plt.gca()
        p = patches.Polygon(CircleEdge(np.array([0.0, 0.0, 0.0]), a),
                            color='gray', alpha=0.1)
        A.add_patch(p)
        plt.plot([0.0], [0.0], 'xk')
        plt.plot([startpoint[dims[0]]], [startpoint[dims[1]]], 'ok')
        A.set(xlim=[-pr, pr], ylim=[-pr, pr])
        XY = spacetime._XY_slice(0.7 * a, startpoint, dims)
        p = patches.Polygon(XY, color='green', alpha=0.3)
        A.add_patch(p)
        plt.show()


if __name__ == '__main__':
    unittest.main()
