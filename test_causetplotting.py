#!/usr/bin/env python
'''
Created on 24 Oct 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
import unittest
from causets.embeddedcauset import EmbeddedCauset  # @UnresolvedImport
from causets.sprinkledcauset import SprinkledCauset  # @UnresolvedImport
from causets.spacetimes import BlackHoleSpacetime  # @UnresolvedImport
from causets.shapes import CoordinateShape  # @UnresolvedImport
from matplotlib import pyplot as plt
import causets.causetplotting as cplt  # @UnresolvedImport


class TestCausetplotting(unittest.TestCase):

    def setUp(self):
        cplt.setDefaultColors('UniYork')
        plt.figure(figsize=(8.0, 8.0))

    def tearDown(self):
        pass

    def test_plotGeneric(self):
        C: EmbeddedCauset = EmbeddedCauset(
            shape=CoordinateShape(3, 'cuboid', edges=[4.0, 2.0, 2.0]),
            coordinates=[[-1.5, 0.0, -0.3], [-0.5, 0.0, -0.1],
                         [0.5, 0.0, 0.1], [1.5, 0.0, 0.3]])
        dims: List[int] = [1, 2, 0]
        P = cplt.Plotter(C, dims=dims,
                         pastcones={'facecolor': 'none', 'alpha': 0.8},
                         futurecones={'facecolor': 'none', 'alpha': 0.8})
        P([0.7])
        ax = plt.gca()
        if len(dims) > 2:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        plt.show()

    def test_plotSprinkle(self):
        C: SprinkledCauset = SprinkledCauset(card=200,
                                             spacetime=BlackHoleSpacetime(2))
        dims = [1, 0]
        e = C.PastInf.copy().pop()
        events_Cone = e.Cone
        cplt.plot(C, dims=dims,
                  events={'alpha': 0.05},
                  links=False, labels=False)
        cplt.plot(list(events_Cone), dims=dims, spacetime=C.Spacetime,
                  events={'markerfacecolor': 'cs:darkblue'},
                  links=True, labels=False)
        cplt.plot(e, dims=dims, spacetime=C.Spacetime,
                  events={'markerfacecolor': 'cs:red'},
                  pastcones={'alpha': 1.0,
                             'linewidth': 2.0},
                  futurecones={'alpha': 1.0,
                               'linewidth': 2.0},
                  time=[-1.0, 1.0])
        C.Shape.plot(dims)
        plt.show()


if __name__ == '__main__':
    unittest.main()
