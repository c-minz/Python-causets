#!/usr/bin/env python
'''
Created on 20 Jul 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
import unittest
from causets.embeddedcauset import CoordinateShape  # @UnresolvedImport
from causets.embeddedcauset import EmbeddedCauset  # @UnresolvedImport
from causets.test_causet import CausetTestCase  # @UnresolvedImport
import causets.causetplotting as cplt  # @UnresolvedImport


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
        cplt.plot(C, dims=[1, 0], spacetime=C.Spacetime, labels=True)
        cplt.show()
        self.assertEqual(C.isPath(C.findAll(2, 3, 6, 7)), False)
        self.assertEqual(C.isPath(C.findAll(2, 5, 6, 7)), True)
        C: EmbeddedCauset = EmbeddedCauset.FromPermutation(
            [4, 3, 2, 1, 8, 7, 6, 5])
        self.assertCauset(C, 8, False, False)

    def test_plot(self):
        C: EmbeddedCauset = EmbeddedCauset.FromPermutation(
            [1, 9, 21, 37, 5, 17, 33, 49, 13, 29, 45, 57, 25, 41, 53, 61,
             2, 10, 22, 38, 6, 18, 34, 50, 14, 30, 46, 58, 26, 42, 54, 62,
             3, 11, 23, 39, 7, 19, 35, 51, 15, 31, 47, 59, 27, 43, 55, 63,
             4, 12, 24, 40, 8, 20, 36, 52, 16, 32, 48, 60, 28, 44, 56, 64])
        cplt.plot(C, dims=[1, 0], spacetime=C.Spacetime, labels=True)
        cplt.show()


if __name__ == '__main__':
    unittest.main()
