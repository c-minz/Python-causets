#!/usr/bin/env python
'''
Created on 20 Jul 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import Set, List, Iterable, Tuple  # @UnusedImport
from typing import Callable, Union, Optional  # @UnusedImport
import numpy as np
from causets.causetevent import CausetEvent  # @UnresolvedImport
from causets.causet import Causet  # @UnresolvedImport
from causets.shapes import CoordinateShape  # @UnresolvedImport
from causets import spacetimes  # @UnresolvedImport
from causets.spacetimes import Spacetime  # @UnresolvedImport


class EmbeddedCauset(Causet):
    '''
    Handles a causal set that is embedded in a subset of a spacetime manifold.
    '''

    _shape: CoordinateShape
    _spacetime: Spacetime

    @staticmethod
    def __raiseDimValueError__(argument: str):
        e: str = 'EmbeddedCauset: The dimension of `%s` is ' + \
                 'not compatible with the other arguments.'
        raise ValueError(e % argument)

    def __init__(self,
                 spacetime: Optional[Union[Spacetime, str]] = None,
                 shape: Optional[Union[str, CoordinateShape]] = None,
                 coordinates: Optional[Union[List[List[float]],
                                             List[np.ndarray],
                                             np.ndarray]] = None,
                 dim: int = -1) -> None:
        '''
        Generates an embedded causal set in a spacetime subset of a specified 
        (coordinate) shape and with the events specified by `causet` and 
        `coordinates`.

        Optional parameters
        -------------------
        spacetime: Spacetime, str
            A spacetime object (including parameters) that determines the 
            causality, or name of spacetime to initialise with default 
            parameters. Supported values for the name are 'flat', 'Minkowski', 
            'dS', 'de Sitter', 'AdS', 'Anti-de Sitter', 'black hole', 
            'Schwarzschild'.
            Default: `spacetimes.FlatSpacetime` of the determined dimension.
        shape: str, CoordinateShape
            (The name of) a coordinate shape that describes the embedding 
            region of the events. 
            Default: `DefaultShape()` of the spacetime object.
        coordinates: np.ndarray
            List of coordinates, a row of coordinates for event to be created.
        dim: int
            Dimension for the default spacetime.
            Default: 2
        Note: A `ValueError` is raised if the dimensions of any of the four 
        optional arguments are not compatible.
        '''
        # initialise base class (Causet):
        super().__init__()
        # initialise dimension:
        if dim <= 0:
            if (spacetime is not None) and isinstance(spacetime, Spacetime):
                dim = spacetime.Dim
            elif (shape is not None) and isinstance(shape, CoordinateShape):
                dim = shape.Dim
            elif coordinates is not None:
                dim = len(coordinates[0])
            else:
                dim = 2  # default
        # initialise spacetime:
        M: Spacetime
        if spacetime is None:
            M = spacetimes.FlatSpacetime(dim)
        elif isinstance(spacetime, str):
            if spacetime in {'flat', 'Minkowski'}:
                M = spacetimes.FlatSpacetime(dim)
            elif spacetime in {'dS', 'de Sitter'}:
                M = spacetimes.deSitterSpacetime(dim)
            elif spacetime in {'AdS', 'Anti-de Sitter'}:
                M = spacetimes.AntideSitterSpacetime(dim)
            elif spacetime in {'black hole', 'Schwarzschild'}:
                M = spacetimes.BlackHoleSpacetime(dim)
            else:
                raise ValueError(
                    'The spacetime name "%s" is not supported.' % spacetime)
        else:
            M = spacetime
        if M.Dim != dim:
            self.__raiseDimValueError__('spacetime')
        self._spacetime = M
        defaultShape = M.DefaultShape()
        # initialise shape:
        _shape: CoordinateShape
        if shape is None:
            _shape = defaultShape
        elif isinstance(shape, str):
            _shape = CoordinateShape(dim, shape)
        else:
            _shape = shape
        if _shape.Dim != defaultShape.Dim:
            self.__raiseDimValueError__('shape')
        self._shape = _shape
        # create new events:
        if coordinates is not None:
            # add labelled events with coordinates:
            if isinstance(coordinates, np.ndarray) and \
                    ((coordinates.ndim != 2) or
                     (coordinates.shape[1] != defaultShape.Dim)):
                self.__raiseDimValueError__('coordinates')
            self.create(coordinates)

    @staticmethod
    def _Permutation_Coords(P: List[int], radius: float) -> np.ndarray:
        '''
        Returns a matrix of (t, x) coordinates with `len(P)` rows, a pair of 
        coordinates for each element in the permutation integer list (integers 
        from 1 to `len(P)`).
        '''
        count: int = len(P)
        coords: np.ndarray = np.empty((count, 2))
        if count > 0:
            cellscale: float = radius / float(count)
            for i, p in enumerate(P):
                crd_u: float = (p - 0.5) * cellscale
                crd_v: float = (i + 0.5) * cellscale
                coords[i, 0] = crd_u + crd_v - radius
                coords[i, 1] = crd_u - crd_v
        return coords

    @staticmethod
    def FromPermutation(P: List[int], labelFormat: Optional[str] = None,
                        radius: float=1.0) -> 'EmbeddedCauset':
        '''
        Generates a causal set from the permutation P of integers from 1 to 
        `len(P)` - that can be embedded in an Alexandrov subset of Minkowski 
        spacetime.
        '''
        C: EmbeddedCauset = EmbeddedCauset(
            shape=CoordinateShape(2, 'diamond', radius=radius))
        C.create(EmbeddedCauset._Permutation_Coords(P, radius), labelFormat)
        return C

    @property
    def Shape(self) -> CoordinateShape:
        '''
        Returns the CoordinateShape object of the embedding region.
        '''
        return self._shape

    @property
    def Spacetime(self) -> Spacetime:
        '''
        Returns the Spacetime object of the embedding.
        '''
        return self._spacetime

    @property
    def Dim(self) -> int:
        '''
        Returns the coordinate dimension of the embedding region.
        '''
        return self.Shape.Dim

    @property
    def Density(self) -> float:
        '''
        Returns the density of events as ratio of the set cardinality to the 
        embedding shape's volume.
        '''
        return float(self.Card) / self.Shape.Volume

    @property
    def LengthScale(self) -> float:
        '''
        Returns the fundamental length scale as inverse d-root of 
        `self.Density` if `card > 0`, else 0.0. 
        '''
        return 0.0 if self.Card == 0 else self.Density**(1.0 / self.Shape.Dim)

    def create(self, coordinates: Union[List[List[float]],
                                        List[np.ndarray],
                                        np.ndarray],
               labelFormat: Optional[str] = None, relate: bool = True) -> \
            Set[CausetEvent]:
        '''
        Creates new events with the specified coordinates, adds them to 
        this instance and returns the new set of events.
        The argument 'coordinates' has to be List[List[float]], 
        List[np.ndarray] or np.ndarray (matrix with a coordinate row for 
        each event).
        '''
        n: int = self.Card + 1
        eventSet: Set[CausetEvent] = {
            CausetEvent(label=(n + i) if labelFormat is None else
                        labelFormat.format(n + i), coordinates=c)
            for i, c in enumerate(coordinates)}
        self._events.update(eventSet)
        if relate:
            self.relate()
        return eventSet

    def relate(self, link: bool = True) -> None:
        '''
        Resets the causal relations between all events based on their 
        embedding in the given spacetime manifold.
        '''
        _iscausal: Callable[[np.ndarray, np.ndarray],
                            Tuple[bool, bool]] = self._spacetime.Causality()
        for e in self._events:
            e._prec = set()
            e._succ = set()
        eventList: List[CausetEvent] = list(self._events)
        eventList_len: int = len(eventList)
        for i, a in enumerate(eventList):
            for j in range(i + 1, eventList_len):
                b = eventList[j]
                isAB, isBA = _iscausal(a.Coordinates, b.Coordinates)
                if isAB:  # A in the past of B:
                    a._succ.add(b)
                    b._prec.add(a)
                if isBA:  # A in the future of B:
                    b._succ.add(a)
                    a._prec.add(b)
        if link:
            self.link()
        else:
            self.unlink()

    def relabel(self, dim: int = 0, descending: bool = False) -> None:
        '''
        Resets the labels of all events to ascending (default) or descending 
        integers (converted to str) corresponding to the coordinate component 
        in dimension 'dim'.
        '''
        eventList = list(self._events)
        sorted_idx = np.argsort(np.array(
            [e.Coordinates[dim] for e in eventList]))
        if descending:
            sorted_idx = np.flip(sorted_idx)
        for i, idx in enumerate(sorted_idx):
            eventList[idx].Label = i + 1
