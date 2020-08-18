#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
import unittest
import math
from sprinkledcausets import SprinkledCauset
from test_causets import CausetTestCase
from matplotlib import pyplot as plt


class TestSprinkledCauset(CausetTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        C = SprinkledCauset(2, card=10, shape={'name': 'cube'})
        self.assertEqual(C.Dim, 2)
        self.assertEqual(C.ShapeName, 'cube')
        a = C.ShapeCenter
        self.assertEqual(a.shape, (2,))
        self.assertEqual(a[0], 0.0)
        self.assertEqual(a[1], 0.0)
        self.assertCauset(C, 10, False, False)

    def test_sprinkle(self):
        C = SprinkledCauset(4, card=0,
                            shape={'name': 'cylinder', 'radius': 0.6})
        C.sprinkle(10)
        C.sprinkle(5)
        C.sprinkle(5)
        self.assertEqual(C.Card, 20)
        self.assertEqual(C.Intensity, 20.0)
        self.assertEqual(C.LengthScale,
                         math.sqrt(math.sqrt(C.Volume / 20.0)))

    def test_intensify(self):
        C = SprinkledCauset(2, card=0,
                            shape={'name': 'bicone'})
        C.intensify(100.0)
        self.assertAlmostEqual(C.Card, 110, delta=35)


if __name__ == '__main__':
    unittest.main()
