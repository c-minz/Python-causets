#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
import unittest
from causets.sprinkledcauset import SprinkledCauset  # @UnresolvedImport
from causets.test_causet import CausetTestCase  # @UnresolvedImport


class TestSprinkledCauset(CausetTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        C = SprinkledCauset(card=10, dim=2, shape='cube')
        self.assertEqual(C.Dim, 2)
        self.assertEqual(C.Shape.Name, 'cube')
        a = C.Shape.Center
        self.assertEqual(a.shape, (2,))
        self.assertEqual(a[0], 0.0)
        self.assertEqual(a[1], 0.0)
        self.assertCauset(C, 10, False, False)


if __name__ == '__main__':
    unittest.main()
