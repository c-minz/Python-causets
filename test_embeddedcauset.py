#!/usr/bin/env python
'''
Created on 20 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
import unittest
import math
from causet import Causet
from event import CausetEvent
from embeddedcauset import EmbeddedCauset
from test_causet import CausetTestCase
from matplotlib import pyplot as plt


class TestEmbeddedCauset(CausetTestCase):

    def setUp(self):
        self.C = EmbeddedCauset(2, shape='cube')

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.C.Dim, 2)
        self.assertEqual(self.C.ShapeName, 'cube')
        a = self.C.ShapeCenter
        self.assertEqual(a.shape, (2,))
        self.assertEqual(a[0], 0.0)
        self.assertEqual(a[1], 0.0)
        self.C.create([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        self.C.relate()
        self.assertCauset(self.C, 4, True, False)

    def test_NewPermutation(self):
        # C: EmbeddedCauset = EmbeddedCauset.NewPermutation(
        #     [3, 2, 5, 1, 8, 6, 7, 4])
        # self.assertEqual(C.isPath(C.findAll(2, 3, 6, 7)), True)
        # C = EmbeddedCauset.NewPermutation([4, 3, 2, 1, 8, 7, 6, 5])
        # self.assertCauset(C, 8, False, False)
        # C: EmbeddedCauset = EmbeddedCauset.NewPermutation(
        #     [3, 12, 2, 5, 11, 10, 1, 8, 6, 9, 7, 4])
        C: Causet = Causet.NewSimplex(3)
        self.assertCauset(C, 15, False, False)
#         C.plot()
#         plt.show()
        eventSet = set(list(C._events)[:3])
        print(C.layered(eventSet))
        C.plotDiagram(eventSet)
        plt.show()
#         C.plot()
#         plt.show()

    def test_setShape_Errors(self):
        self.assertRaises(ValueError, self.C.setShape,
                          0, 'cube')  # dimension too low
        self.assertRaises(ValueError, self.C.setShape,
                          2, 'unknown')  # unknown shape
        self.assertRaises(ValueError, self.C.setShape,
                          2, 'diamond', center=[0.0, 0.0, 0.0])  # dimension mismatch
        self.assertRaises(ValueError, self.C.setShape,
                          2, 'cuboid', edges=[1.0, 1.0, 0.0])  # dimension mismatch
        self.assertRaises(ValueError, self.C.setShape,
                          2, 'cuboid', edges=[1.0, 0.0])  # non-positive edge length

    def test_ShapeLimits(self):
        self.C.setShape(2, 'cuboid', edges=[1.0, 1.6], center=[1.0, 3.0])
        self.assertRaises(ValueError, self.C.ShapeLimits,
                          2)  # no 3rd dimension
        lim = self.C.ShapeLimits(1)
        self.assertAlmostEqual(lim[0], 2.2, 5)
        self.assertAlmostEqual(lim[1], 3.8, 5)

    def test_Volume(self):
        self.C.setShape(2, 'cuboid', edges=[2.0, 3.0])
        self.assertEqual(self.C.Volume, 6.0)
        self.C.setShape(3, 'cube', edge=2.0)
        self.assertEqual(self.C.Volume, 8.0)
        self.C.setShape(3, 'ball', radius=2.1)
        self.assertAlmostEqual(self.C.Volume,
                               4.0 * math.pi * 2.1**3.0 / 3.0, 4)
        self.C.setShape(4, 'cylinder', radius=2.1, duration=2.4)
        self.assertAlmostEqual(self.C.Volume,
                               2.4 * 4.0 * math.pi * 2.1**3.0 / 3.0, 4)
        self.C.setShape(4, 'diamond', radius=2.1)
        self.assertAlmostEqual(self.C.Volume,
                               2.0 * 2.1 * 4.0 * math.pi * 2.1**3.0 / 3.0 / 4.0, 4)

#     def test_plot(self):
#         self.C.setShape(3, 'cuboid', edges=[4.0, 2.0, 2.0])
#         self.C.create([[-1.5, 0.0, -0.3], [-0.5, 0.0, -0.1],
#                        [0.5, 0.0, 0.1], [1.5, 0.0, 0.3]])
#         self.C.relate()
#         plt.figure(figsize=(8.0, 8.0))
#         P = self.C.plotter(dims=[1, 2, 0], pastcones={'alpha': 0.8},
#                            futurecones={'alpha': 0.8})
#         P([0.0])
#         ax = plt.gca()
#         ax.xaxis.pane.fill = False
#         ax.yaxis.pane.fill = False
#         ax.zaxis.pane.fill = False
#         ax.grid(color='green')
#         plt.show()


if __name__ == '__main__':
    unittest.main()
