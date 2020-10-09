#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
import unittest
from sprinkledcauset import SprinkledCauset
from test_causet import CausetTestCase
from matplotlib import pyplot as plt
import color_schemes as colors
import causet_plotting as cplt
from spacetimes import deSitterSpacetime


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

    def test_plot(self):
        cplt.setDefaultColors('UniYork')
        C = SprinkledCauset(card=100,
                            spacetime=deSitterSpacetime(3))
        dims = [1, 2, 0]
        e = set(C).copy().pop()
        events_Cone = e.Cone
        colors.setGlobalColorScheme('UniYork')
        if len(dims) > 2:
            plt.figure(figsize=(8.0, 8.0))
        C.plot(dims=dims,
               events={'alpha': 0.05},
               links=False, labels=False)
        C.plot(eventList=list(events_Cone), dims=dims,
               events={'markerfacecolor': 'cs:darkblue'},
               links=False, labels=False)
        C.plot(eventList=[e], dims=dims,
               events={'markerfacecolor': 'cs:red'},
               links=False, labels=False,
               pastcones={'alpha': 1.0,
                          'linewidth': 2.0, 'facecolor': 'none'},
               futurecones={'alpha': 1.0,
                            'linewidth': 2.0, 'facecolor': 'none'},
               time=[-1.0, 1.0])
        C.Shape.plot(dims)
        plt.gca().grid(colors.getColor('cs:green'))
        plt.show()


if __name__ == '__main__':
    unittest.main()
