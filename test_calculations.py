#!/usr/bin/env python
'''
Created on 22 Aug 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List
import unittest
from fractions import Fraction
from causets.calculations import HarmonicNumber, HarmonicNumbers, \
    HarmonicNumberFraction, HarmonicNumberFractions


class TestCalculations(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_HamonicNumber(self):
        self.assertEqual(HarmonicNumber(0), 0.)
        self.assertEqual(HarmonicNumber(1), 1.)
        self.assertEqual(HarmonicNumber(2), 1.5)
        self.assertAlmostEqual(HarmonicNumber(10),
                               HarmonicNumber(9) + 1. / 10, 10)
        self.assertAlmostEqual(HarmonicNumber(100),
                               HarmonicNumber(98) + 1. / 99 + 1. / 100, 10)

    def test_HamonicNumberFraction(self):
        self.assertEqual(HarmonicNumberFraction(1), Fraction(1, 1))
        self.assertEqual(HarmonicNumberFraction(2), Fraction(3, 2))
        self.assertAlmostEqual(HarmonicNumberFraction(10),
                               HarmonicNumber(10), 10)
        self.assertAlmostEqual(HarmonicNumberFraction(100),
                               HarmonicNumber(100), 10)

    def test_HamonicNumbers(self):
        H: np.ndarray = HarmonicNumbers(200)
        for i in [0, 1, 10, 100, 101, 150, 200]:
            self.assertAlmostEqual(H[i], HarmonicNumber(i), 10)

    def test_HamonicNumberFractions(self):
        H: List[Fraction] = HarmonicNumberFractions(10)
        for i in range(len(H)):
            self.assertEqual(H[i], HarmonicNumberFraction(i), 10)


if __name__ == '__main__':
    unittest.main()
