#!/usr/bin/env python
'''
Created on 20 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
import unittest
from embeddedcauset import CoordinateShape, EmbeddedCauset
from test_causet import CausetTestCase


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


if __name__ == '__main__':
    unittest.main()
