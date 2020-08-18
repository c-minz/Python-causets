#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
from typing import Set, List, Iterable, Union
import numpy as np
from numpy.random import default_rng
from events import CausetEvent
from embeddedcausets import EmbeddedCauset


class SprinkledCauset(EmbeddedCauset):
    '''
    Handles a causal set that is embedded in a subset of a manifold.
    '''

    _intensity: float

    def __init__(self, dim: int, **kwargs) -> None:
        '''
        Generates a sprinkled causal set by sprinkling in a 
        spacetime subset with dimension 'dim' of at least 1. 

        Keywords 'shape' and 'spacetime' are handled by the super 
        class EmbeddedCauset before events are sprinkled.

        'card': int
        Number of sprinkled events.

        'intensity': float
        Sprinkling intensity parameter, the expected number of 
        sprinkled events.
        '''
        # initialise shape and spacetime with super class:
        super_kwargs = {}
        try:
            super_kwargs['shape'] = kwargs['shape']
        except KeyError:
            pass
        try:
            super_kwargs['spacetime'] = kwargs['spacetime']
        except KeyError:
            pass
        super().__init__(dim, **super_kwargs)
        # sprinkle:
        self._intensity = 0.0
        try:
            self.sprinkle(kwargs['card'])
        except KeyError:
            try:
                self.intensify(kwargs['intensity'])
            except KeyError:
                self.intensify(0.0)

    @property
    def Intensity(self) -> float:
        '''
        Returns the sprinkling intensity, which is the expected 
        number of sprinkled events. The exact number of sprinkled 
        events is given by the property 'Card'.
        '''
        return self._intensity

    @property
    def Density(self) -> float:
        '''
        Returns the sprinkling density, which is measured in SI 
        units and has the 'Dim'-fold power of the in inverse 
        physical length dimension [1 / L].
        '''
        return self._intensity / self.Volume

    @property
    def LengthScale(self) -> float:
        '''
        Returns the fundamental length scale. It is the inverse 
        of the 'Dim'-th root of the sprinkling density.
        '''
        return (self.Volume / self._intensity)**(1.0 / self.Dim)

    def _sprinkle_coords(self, count: int,
                         rng=default_rng()) -> np.ndarray:
        if count < 0:
            raise ValueError('The sprinkle cardinality has to ' +
                             'be a non-negative integer.')
        coords: np.ndarray = np.empty((count, self._dim),
                                      dtype=np.float32)
        if self._shape_name in ('cube', 'cuboid'):
            # Create rectangle based sprinkle:
            low: np.ndarray
            high: np.ndarray
            if self._shape_name == 'cuboid':
                low = self._shape_center - \
                    self._shape_params['edges'] / 2
                high = self._shape_center + \
                    self._shape_params['edges'] / 2
            else:
                low = self._shape_center - \
                    self._shape_params['edge'] / 2
                high = self._shape_center + \
                    self._shape_params['edge'] / 2
            for i in range(count):
                coords[i, :] = rng.uniform(low, high)
        elif self._shape_name in ('ball', 'cylinder', 'bicone'):
            # Create circle based sprinkle:
            isCylinder: bool = self._shape_name == 'cylinder'
            isBicone: bool = self._shape_name == 'bicone'
            d: int = self.Dim
            b_r: float = self._shape_params['radius']
            b_dstart: int = 0 if self._shape_name == 'ball' else 1
            b_d: int = d - b_dstart
            if isCylinder:
                time_low: float = self._shape_center[0] - \
                    self._shape_params['duration'] / 2
                time_high: float = self._shape_center[0] + \
                    self._shape_params['duration'] / 2
            # pick 'count' random coordinate tuples uniformly:
            for i in range(count):
                # get coordinates on sphere using normal distribution:
                coord: np.ndarray = rng.standard_normal(size=(1, b_d))
                r: float = np.sqrt(sum(np.square(coord)))
                r_scaling: float = rng.uniform()**(1.0 / b_d)
                if isBicone:
                    # set time coordinate:
                    h_squeeze: float = rng.uniform()**(1.0 / d)
                    h_sign: float = np.sign(
                        rng.uniform(low=-1.0, high=1.0))
                    coords[i, 0] = h_sign * (1 - h_squeeze) * b_r
                    # adjust scaling:
                    r_scaling *= h_squeeze
                elif isCylinder:
                    # set time coordinate:
                    coords[i, 0] = rng.uniform(time_low, time_high)
                coords[i, b_dstart:] = (r_scaling * b_r / r) * coord
        return coords

    def sprinkle(self, count: int,
                 rng=default_rng()) -> Set[CausetEvent]:
        '''
        Creates a fixed number of new events by sprinkling 
        into the shape.
        '''
        if count < 0:
            raise ValueError('The sprinkle cardinality has to ' +
                             'be a non-negative integer.')
        self._intensity += float(count)
        coords: np.ndarray = self._sprinkle_coords(count, rng)
        return super().create(coords)

    def intensify(self, intensity: float,
                  rng=default_rng()) -> Set[CausetEvent]:
        '''
        Creates an expected number of new events by sprinkling 
        into the shape. The expected number is determined by the 
        Poisson process with the given 'intensity' parameter.
        '''
        if intensity < 0.0:
            raise ValueError('The intensity parameter has to ' +
                             'be a non-negative float.')
        self._intensity += intensity
        count: int = int(rng.poisson(lam=intensity))
        coords: np.ndarray = self._sprinkle_coords(count, rng)
        return super().create(coords)

    def create(self, coords: Union[Iterable[List[float]],
                                   Iterable[np.ndarray],
                                   np.ndarray],
               relate: bool = True) -> Set[CausetEvent]:
        card_old: float = float(self.Card)
        eventSet: Set[CausetEvent] = super().create(coords, relate)
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
