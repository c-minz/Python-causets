#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
import unittest
import math
from causet import Causet
from sprinkledcauset import SprinkledCauset
from test_causet import CausetTestCase
from matplotlib import pyplot as plt
import numpy as np


class TestSprinkledCauset(CausetTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        C = SprinkledCauset(2, card=10, shape='cube')
        self.assertEqual(C.Dim, 2)
        self.assertEqual(C.ShapeName, 'cube')
        a = C.ShapeCenter
        self.assertEqual(a.shape, (2,))
        self.assertEqual(a[0], 0.0)
        self.assertEqual(a[1], 0.0)
        self.assertCauset(C, 10, False, False)

#     def test_sprinkle(self):
#         C = SprinkledCauset(4, card=0,
#                             shape={'name': 'cylinder', 'radius': 0.6})
#         C.sprinkle(10)
#         C.sprinkle(5)
#         C.sprinkle(5)
#         self.assertEqual(C.Card, 20)
#         self.assertEqual(C.Intensity, 20.0)
#         self.assertEqual(C.LengthScale,
#                          math.sqrt(math.sqrt(C.Volume / 20.0)))
#
#     def test_intensify(self):
#         C = SprinkledCauset(2, card=0,
#                             shape={'name': 'diamond'})
#         C.intensify(100.0)
#         self.assertAlmostEqual(C.Card, 110, delta=35)
#
    def test_plot(self):
        C = SprinkledCauset(3, card=500,
                            shape='cuboid', edges=[1.5, 4.0, 3.0],
                            center=[0.5, 1.3, 2.0])
        plt.figure(figsize=(8.0, 8.0))
        C.plot(dims=[1, 2, 0], links=False, labels=False)
        plt.show()


if __name__ == '__main__':
    unittest.main()
