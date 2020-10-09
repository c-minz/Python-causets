#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
from typing import Set, List, Iterable, Union
import numpy as np
import math
from numpy.random import default_rng
from event import CausetEvent
from embeddedcauset import EmbeddedCauset
from shapes import CoordinateShape
from spacetimes import Spacetime


class SprinkledCauset(EmbeddedCauset):
    '''
    Handles a causal set that is embedded in a subset of a manifold.
    '''

    _intensity: float

    def __init__(self,
                 card: int = 0, intensity: float = 0.0,
                 dim: int = 2,
                 spacetime: Spacetime = None,
                 shape: Union[str, CoordinateShape] = None) -> None:
        '''
        Generates a sprinkled causal set by sprinkling in a 
        spacetime subset with dimension `dim` of at least 1. 

        The arguments `dim`, `shape` and `spacetime` are handled by the 
        super class `EmbeddedCauset` before event are sprinkled.

        'card': int
        Number of sprinkled event.

        'intensity': float
        Sprinkling intensity parameter, the expected number of 
        sprinkled event.
        '''
        # initialise shape and spacetime with super class:
        super().__init__(dim, spacetime, shape)
        # sprinkle:
        self._intensity = 0.0
        if card > 0:
            self.sprinkle(card)
        else:
            self.intensify(intensity)

    @property
    def Intensity(self) -> float:
        '''
        Returns the sprinkling intensity, which is the expected 
        number of sprinkled event. The exact number of sprinkled 
        event is given by the property 'Card'.
        '''
        return self._intensity

    @property
    def Density(self) -> float:  # overwrites superclass
        return self._intensity / self.Shape.Volume

    @property
    def LengthScale(self) -> float:  # overwrites superclass
        return (self.Shape.Volume / self._intensity)**(1.0 / self.Dim)

    def _sprinkle_coords(self, count: int, shape: CoordinateShape,
                         rng) -> np.ndarray:
        if count < 0:
            raise ValueError('The sprinkle cardinality has to ' +
                             'be a non-negative integer.')
        coords: np.ndarray = np.empty((count, self.Dim),
                                      dtype=np.float32)
        if shape.Name in ('cube', 'cuboid'):
            # Create rectangle based sprinkle:
            low: np.ndarray
            high: np.ndarray
            if shape.Name == 'cuboid':
                low = shape.Center - \
                    shape.Parameter('edges') / 2
                high = shape.Center + \
                    shape.Parameter('edges') / 2
            else:
                low = shape.Center - \
                    shape.Parameter('edge') / 2
                high = shape.Center + \
                    shape.Parameter('edge') / 2
            for i in range(count):
                coords[i, :] = rng.uniform(low, high)
        elif shape.Name in ('ball', 'cylinder', 'diamond'):
            # Create circle based sprinkle:
            isCylindrical: bool = 'cylinder' in shape.Name
            isDiamond: bool = 'diamond' in shape.Name
            d: int = self.Dim
            b_r: float = shape.Parameter('radius')
            if (d == 2) and isDiamond:
                # pick `count` random coordinate tuples uniformly:
                uv: np.ndarray = rng.uniform(low=-1.0, high=1.0,
                                             size=(count, 2))
                coords[:, 0] = uv[:, 0] + uv[:, 1]
                coords[:, 1] = uv[:, 0] - uv[:, 1]
                coords *= b_r / 2
            else:
                b_dstart: int = 0 if shape.Name == 'ball' else 1
                b_d: int = d - b_dstart
                if isCylindrical:
                    # set time coordinate:
                    time_low: float = shape.Center[0] - \
                        shape.Parameter('duration') / 2
                    time_high: float = shape.Center[0] + \
                        shape.Parameter('duration') / 2
                    coords[:, 0] = rng.uniform(time_low, time_high,
                                               size=(count,))
                # pick `count` random coordinate tuples uniformly:
                r_low: float = shape.Parameter('hollow')**b_d
                for i in range(count):
                    # get coordinates on sphere using normal distribution:
                    coord: np.ndarray = rng.standard_normal(size=(b_d,))
                    r: float = np.sqrt(sum(np.square(coord)))
                    r_scaling: float
                    r_scaling = rng.uniform(low=r_low)**(1.0 / b_d)
                    if isDiamond:
                        # set time coordinate:
                        h_squeeze: float = rng.uniform()**(1.0 / d)
                        h_sign: float = np.sign(
                            rng.uniform(low=-1.0, high=1.0))
                        coords[i, 0] = h_sign * (1 - h_squeeze) * b_r
                        # adjust scaling:
                        r_scaling *= h_squeeze
                    coords[i, b_dstart:] = shape.Center[b_dstart:] + \
                        (r_scaling * b_r / r) * coord
        return coords

    def sprinkle(self, count: int, rng=default_rng(),
                 shape: CoordinateShape = None) -> Set[CausetEvent]:
        '''
        Creates a fixed number of new event by sprinkling into `shape` 
        (by default the entire embedding region).
        '''
        if count < 0:
            raise ValueError('The sprinkle cardinality has to ' +
                             'be a non-negative integer.')
        self._intensity += float(count)
        if shape is None:
            shape = self.Shape
        coords: np.ndarray = self._sprinkle_coords(count, shape, rng)
        return super().create(coords)

    def intensify(self, intensity: float, rng=default_rng(),
                  shape: CoordinateShape = None) -> Set[CausetEvent]:
        '''
        Creates an expected number of new events by sprinkling into 
        `shape` (by default the entire embedding region). The expected 
        number is determined by the Poisson distribution with the 
        given `intensity` parameter.
        '''
        if intensity < 0.0:
            raise ValueError('The intensity parameter has to ' +
                             'be a non-negative float.')
        self._intensity += intensity
        count: int = int(rng.poisson(lam=intensity))
        if shape is None:
            shape = self.Shape
        coords: np.ndarray = self._sprinkle_coords(count, shape, rng)
        return super().create(coords)

    def create(self, coords: Union[Iterable[List[float]],
                                   Iterable[np.ndarray],
                                   np.ndarray],
               labelFormat: str = None,
               relate: bool = True) -> Set[CausetEvent]:
        card_old: float = float(self.Card)
        eventSet: Set[CausetEvent] = super().create(
            coords, labelFormat, relate)
        self._intensity += (float(self.Card) - card_old)
        return eventSet

    def add(self, eventSet: Iterable, unlink: bool = False) -> None:
        card_old: float = float(self.Card)
        super().add(eventSet, unlink)
        self._intensity += (float(self.Card) - card_old)

    def discard(self, eventSet: Iterable, unlink: bool = False) -> None:
        card_old: float = float(self.Card)
        super().discard(eventSet, unlink)
        self._intensity *= (float(self.Card) / card_old)
