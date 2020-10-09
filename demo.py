#!/usr/bin/env python
'''
Created on 9 Oct 2020

@author: Christoph Minz
'''
from __future__ import annotations
from sprinkledcauset import SprinkledCauset
from spacetimes import deSitterSpacetime
from shapes import CoordinateShape
from matplotlib import pyplot as plt
import causet_plotting as cplt

# Create a sprinkle from de Sitter spacetime with cosmological horizon at
# radius 1.0. Coordinates range over a hollow cylinder with height 3.0.
# 30% of the cylinder interior is hollow.
S: CoordinateShape = CoordinateShape(3, 'cylinder', duration=3.0, hollow=0.3)
C: SprinkledCauset = SprinkledCauset(intensity=250.0,
                                     spacetime=deSitterSpacetime(3), shape=S)
e: CausetEvent = C.CentralAntichain().pop()  # pick one event

# Plotting setup:
cplt.setDefaultColors('UniYork')  # University of York brand colours
dims: List[int] = [1, 2, 0]  # choose the (order of) plot dimensions
if len(dims) > 2:
    plt.figure(figsize=(8.0, 8.0))
S.plot(dims)  # plot the embedding shape
# Add causet plots and show result:
C.plot(dims=dims,
       events={'alpha': 0.05},
       links={'alpha': 0.1, 'linewidth': 0.5}, labels=False)
C.plot(eventList=list(e.Cone), dims=dims,
       events={'markerfacecolor': 'cs:darkblue'},
       links=False, labels=False)
# end times for past and future light-cone:
timeslices: Tuple[float, float] = S.Limits(0)
C.plot(eventList=[e], dims=dims,
       events={'markerfacecolor': 'cs:red'},
       links=False, labels=False,
       pastcones={'alpha': 1.0},
       futurecones={'alpha': 1.0},
       time=timeslices)
plt.show()
