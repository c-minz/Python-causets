#!/usr/bin/env python
'''
Created on 20 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
import unittest
from causets import Causet
from embeddedcausets import EmbeddedCauset
from matplotlib import pyplot as plt


class CausetTestCase(unittest.TestCase):

    def assertCauset(self, S: Causet, size: int,
                     is_chain: bool, is_antichain: bool):
        self.assertEqual(S.Card, size)
        self.assertEqual(S.isPath(), is_chain and (size > 0))
        self.assertEqual(S.isChain(), is_chain)
        self.assertEqual(S.isAntichain(), is_antichain)


class TestCauset(CausetTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_NewChain(self):
        self.assertCauset(Causet.NewChain(5), 5, True, False)
        self.assertCauset(Causet.NewChain(0), 0, True, True)
        self.assertCauset(Causet.NewChain(-5), 0, True, True)
        self.assertRaises(TypeError, Causet.NewChain, 1.2)
        self.assertRaises(TypeError, Causet.NewChain, 'chain')

    def test_NewAntichain(self):
        self.assertCauset(Causet.NewAntichain(5), 5, False, True)
        self.assertCauset(Causet.NewAntichain(0), 0, True, True)
        self.assertCauset(Causet.NewAntichain(-5), 0, True, True)
        self.assertRaises(TypeError, Causet.NewAntichain, 1.2)
        self.assertRaises(TypeError, Causet.NewAntichain, 'antichain')

    def test_NewSimplex(self):
        self.assertCauset(Causet.NewSimplex(0), 1, True, True)
        self.assertCauset(Causet.NewSimplex(1), 3, False, False)
        self.assertCauset(Causet.NewSimplex(2, False), 6, False, False)
        self.assertCauset(Causet.NewSimplex(3), 15, False, False)
        self.assertCauset(Causet.NewSimplex(4, False), 30, False, False)

    def test_NewFence(self):
        self.assertCauset(Causet.NewFence(-1), 0, True, True)
        self.assertCauset(Causet.NewFence(1), 2, True, False)
        self.assertCauset(Causet.NewFence(2), 4, False, False)
        self.assertCauset(Causet.NewFence(3), 6, False, False)
        self.assertCauset(Causet.NewFence(4), 8, False, False)

    def test_merge(self):
        S_antichain: Causet = Causet.NewAntichain(5)
        S_chain: Causet = Causet.NewChain(5)
        S: Causet = Causet.merge(S_antichain, S_chain)
        self.assertCauset(S, 10, False, False)
        self.assertEqual(len(S.PastInf), 5)
        self.assertEqual(len(S.FutureInf), 1)

    def test_Paths(self):
        S: Causet = Causet.NewAntichain(5)
        self.assertEqual(list(S.Paths(S.find(1), S.find(5))), [])
        S = Causet.NewPermutation([1, 5, 4, 3, 2, 6])
        i: int = 0
        bot: CausetEvent = S.find(1)
        central: Set[CausetEvent] = S.findAll(2, 3, 4, 5)
        top: CausetEvent = S.find(6)
        for p in S.Paths(bot, top, length='min'):
            i += 1
            self.assertEqual(S.isPath(p), True)
            self.assertEqual(len(p), 3)
            self.assertEqual(p[0], bot)
            self.assertIn(p[1], central)
            self.assertEqual(p[2], top)
        self.assertEqual(i, 4)
        S = Causet.NewSimplex(3)
        self.assertEqual(len(list(S.Paths(S.find('1'),
                                          S.find('1-2-3-4')))), 6)

    def test_Antipaths(self):
        S = Causet.NewFence(6)
        i: int = 0
        first: CausetEvent = S.find('1')
        between: Set[CausetEvent] = S.findAll('2', '3', '5', '6')
        last: CausetEvent = S.find('4')
        for ap in S.Antipaths(first, last, along=S.PastInf):
            i += 1
            self.assertEqual(len(ap), 4)
            self.assertEqual(ap[0], first)
            self.assertIn(ap[1], between)
            self.assertIn(ap[2], between)
            self.assertEqual(ap[3], last)
        self.assertEqual(i, 2)
        i = 0
        for ap in S.Antipaths(first, first, along=S.PastInf):
            i += 1
            self.assertEqual(len(ap), 1)
            self.assertEqual(ap[0], first)
        self.assertEqual(i, 1)

    def test_plotDiagram(self):
        S: Causet = Causet.NewSimplex(3)
        eventList: List[CausetEvent] = list(S._events)
        # S.plotDiagram(set([*eventList[1:12], *eventList[13:15]]))
        # S.plotDiagram(set([*eventList[2:8], eventList[0]]))
        S = EmbeddedCauset.NewPermutation([8, 5, 7, 3, 4, 2, 6, 1])
        S = EmbeddedCauset.NewPermutation([6, 8, 4, 5, 3, 7, 2, 1])
        S = EmbeddedCauset.NewPermutation(
            [3, 8, 10, 11, 14, 7, 1, 5, 12, 9, 6, 2, 13, 4])
        S = Causet.NewFence(5)
#         events = S.sortedByLabels(S.PastInf)
#         S._print_eventlist(events)
#         print(S.DistanceMatrix(events, False))
#         print(S.DistanceMatrix(events))
        # S._print_eventlists(S.disjoint())
#         S = Causet.NewPermutation(
#             [1, 6, 8, 7, 9, 5, 12, 3, 2, 10, 11, 4])
        # eventList: List[CausetEvent] = list(S._events)
        # S.plotDiagram(set([*eventList[0:11], *eventList[13:14]]))
        # print('| '.join(', '.join(str(e)
        #                           for e in layer)
        #                 for layer in S.layered()))
        # S = Causet.NewAntichain(11)
#         S.plotDiagram()
#         plt.show()


if __name__ == '__main__':
    unittest.main()
