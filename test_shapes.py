#!/usr/bin/env python
'''
Created on 04 Oct 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
import unittest
import math
from causets.shapes import CoordinateShape  # @UnresolvedImport
from matplotlib import pyplot as plt


class TestCoordinateShape(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertRaises(ValueError, CoordinateShape,
                          0, '')  # dimension error
        self.assertRaises(ValueError, CoordinateShape,
                          3, 'banana')  # unknown shape
        self.assertRaises(ValueError, CoordinateShape,
                          2, 'diamond', hollow=1.1)  # too hollow
        cube = CoordinateShape(3, 'cube', edge=3.0, center=[0.0, 1.0, 2.0])
        self.assertEqual(cube.Dim, 3)
        self.assertEqual(cube.Name, 'cube')
        self.assertEqual(cube.Parameter('edge'), 3.0)
        self.assertEqual(cube.Volume, 3.0**3)

    def test_Dim(self):
        diamond = CoordinateShape(2, 'diamond')
        self.assertEqual(diamond.Dim, 2)
        self.assertRaises(ValueError, CoordinateShape, 0, 'diamond')

    def test_HollowVolumes(self):
        diamond = CoordinateShape(3, 'diamond', radius=2.5, hollow=0.1)
        self.assertAlmostEqual(diamond.Volume,
                               2.0 * math.pi * (2.5**3 - 0.25**3) / 3.0, 5)
        cylinder = CoordinateShape(2, 'cylinder', radius=2.5, hollow=0.9)
        self.assertAlmostEqual(cylinder.Volume, (5.0 - 4.5) * 5.0, 5)

    def test_Limits(self):
        cuboid = CoordinateShape(2, 'cuboid',
                                 edges=[1.0, 1.6], center=[1.0, 3.0])
        self.assertRaises(ValueError, cuboid.Limits, 2)  # no 3rd dimension
        lim = cuboid.Limits(1)
        self.assertAlmostEqual(lim[0], 3.0 - 1.6 / 2, 5)
        self.assertAlmostEqual(lim[1], 3.0 + 1.6 / 2, 5)

    def test_plot(self):
        S = CoordinateShape(3, 'bicone', radius=3.5, hollow=0.7,
                            center=[1.2, 1.0, 0.0])
        dims: List[int] = [1, 2, 0]
        plt.figure(figsize=(8.0, 8.0))
        S.plot(dims, alpha=0.1, edgecolor='gray')
        if len(dims) == 3:
            plt.gca().set(xlim=[-5.0, 5.0], ylim=[-5.0, 5.0], zlim=[-5.0, 5.0])
        else:
            plt.gca().set(xlim=[-5.0, 5.0], ylim=[-5.0, 5.0])
        plt.show()


if __name__ == '__main__':
    unittest.main()
