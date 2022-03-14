#!/usr/bin/env python
'''
Created on 22 Aug 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List  # @UnusedImport
import unittest
from fractions import Fraction
import causets.calculations as calc   # @UnresolvedImport


class TestCalculations(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_HamonicNumber(self):
        self.assertEqual(calc.HarmonicNumber(0), 0.)
        self.assertEqual(calc.HarmonicNumber(1), 1.)
        self.assertEqual(calc.HarmonicNumber(2), 1.5)
        self.assertAlmostEqual(calc.HarmonicNumber(10),
                               calc.HarmonicNumber(9) + 1. / 10, 10)
        self.assertAlmostEqual(calc.HarmonicNumber(100),
                               calc.HarmonicNumber(98) + 1. / 99 + 1. / 100, 10)

    def test_HamonicNumberFraction(self):
        self.assertEqual(calc.HarmonicNumberFraction(1), Fraction(1, 1))
        self.assertEqual(calc.HarmonicNumberFraction(2), Fraction(3, 2))
        self.assertAlmostEqual(calc.HarmonicNumberFraction(10),
                               calc.HarmonicNumber(10), 10)
        self.assertAlmostEqual(calc.HarmonicNumberFraction(100),
                               calc.HarmonicNumber(100), 10)

    def test_HamonicNumbers(self):
        H: np.ndarray = calc.HarmonicNumbers(200)
        for i in [0, 1, 10, 100, 101, 150, 200]:
            self.assertAlmostEqual(H[i], calc.HarmonicNumber(i), 10)

    def test_HamonicNumberFractions(self):
        H: List[Fraction] = calc.HarmonicNumberFractions(10)
        for i in range(len(H)):
            self.assertEqual(H[i], calc.HarmonicNumberFraction(i), 10)


if __name__ == '__main__':
    unittest.main()
