#!/usr/bin/env python
'''
Created on 20 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
import unittest
from embeddedcauset import CoordinateShape, EmbeddedCauset
from test_causet import CausetTestCase
from matplotlib import pyplot as plt
import causet_plotting as cplt
from spacetimes import FlatSpacetime


class TestEmbeddedCauset(CausetTestCase):

    def setUp(self):
        self.C = EmbeddedCauset(
            shape=CoordinateShape(2, 'cube', center=[0.5, 5.3]))

    def tearDown(self):
        pass

    def test_create(self):
        self.C.create([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.C.relate()
        self.assertCauset(self.C, 4, True, False)

    def test_FromPermutation(self):
        C: EmbeddedCauset = EmbeddedCauset.FromPermutation(
            [3, 2, 5, 1, 8, 6, 7, 4])
        self.assertEqual(C.isPath(C.findAll(2, 3, 6, 7)), True)
        C: EmbeddedCauset = EmbeddedCauset.FromPermutation(
            [4, 3, 2, 1, 8, 7, 6, 5])
        self.assertCauset(C, 8, False, False)

    def test_plot(self):
        C: EmbeddedCauset = EmbeddedCauset(
            shape=CoordinateShape(3, 'cuboid', edges=[4.0, 2.0, 2.0]),
            coordinates=[[-1.5, 0.0, -0.3], [-0.5, 0.0, -0.1],
                         [0.5, 0.0, 0.1], [1.5, 0.0, 0.3]])
        plt.figure(figsize=(8.0, 8.0))
        cplt.setDefaultColors('UniYork')
        dims: List[int] = [1, 2, 0]
        P = C.TimeslicePlotter(dims=dims,
                               pastcones={'facecolor': 'none', 'alpha': 0.8},
                               futurecones={'facecolor': 'none', 'alpha': 0.8})
        P([0.7])
        ax = plt.gca()
        if len(dims) > 2:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        plt.show()


if __name__ == '__main__':
    unittest.main()
