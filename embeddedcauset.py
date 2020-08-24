#!/usr/bin/env python
'''
Created on 20 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
from typing import Set, List, Iterable, Dict, Tuple, Callable, Any, Union
import math
import numpy as np
from event import CausetEvent
from causet import Causet
from spacetimes import Causality
import causet_plotting as cs_plt
from matplotlib import axes as plta


class EmbeddedCauset(Causet):
    '''
    Handles a causal set that is embedded in a subset of a manifold.
    '''

    _dim: int
    _shape_name: str
    _shape_center: List[float]
    _shape_params: Dict[str, Any]
    __coords: np.ndarray

    def __init__(self, dim: int, **kwargs) -> None:
        '''
        Generates an embedded causal set in a spacetime subset 
        with dimension 'dim' of at least 1.

        `shape`: str
        For supported shapes and their optional parameters, see 
        setShape(...).
        Default: 'diamond'

        `spacetime`: str
        Name of the spacetime causality that must have been implemented 
        in the function `Causality` of the spacetime module.
        The keyword arguments are passed to the `Causality` function so that 
        spacetime parameters can be stored. 
        Default: 'Minkowski'

        `causet`: Causet or Set[CausetEvent]
        Causal set of event to initialise this instance - 
        super class call.
        Default: set()

        `coords`: np.ndarray
        Matrix with a row for the coordinates for event to be added 
        to this instance.
        Default: matrix with no rows
        '''
        # initialise dimension and shape:
        if dim < 1:
            raise ValueError('The dimension must be an ' +
                             'integer of at least 1.')
        name: str
        try:
            name = kwargs.pop('shape')
            self.setShape(dim, name, **kwargs)
        except KeyError:
            self.setShape(dim, 'diamond', **kwargs)
        # initialise spacetime:
        self._iscausal: Callable[[np.ndarray, np.ndarray], Tuple[bool, bool]]
        try:
            name = kwargs.pop('spacetime')
            self._iscausal = Causality(name, self._dim, **kwargs)
        except KeyError:
            self._iscausal = Causality('Minkowski', self._dim)
        # initialises the super class (with events from a causal set):
        S: Set[CausetEvent]
        try:
            S = set(kwargs['causet'])
        except KeyError:
            S = set()
        super().__init__(S)
        # add labelled event with coordinates:
        coords: np.ndarray
        try:
            coords = kwargs['coords']
            if coords.ndim != 2:
                raise ValueError
            self.create(coords)
        except (KeyError, ValueError):
            pass

    @staticmethod
    def FromPermutation(P: List[int], radius: float=1.0) -> 'EmbeddedCauset':
        '''
        Generates a causal set from the permutation P of integers from 1 
        to len(P) - that can be embedded in an Alexandrov subset of 
        Minkowski spacetime.
        '''
        return EmbeddedCauset(2, shape='diamond', radius=radius,
                              coords=Causet._Permutation_Coords(P, radius))

    @property
    def Dim(self) -> int:
        '''
        Returns the dimension of the spacetime (region).
        '''
        return self._dim

    @property
    def ShapeName(self) -> str:
        '''
        Returns the shape name of the spacetime region.
        '''
        return self._shape_name

    @property
    def ShapeCenter(self) -> List[float]:
        '''
        Returns the shape center of the spacetime region.
        '''
        return self._shape_center

    def ShapeParam(self, key: str) -> Any:
        '''
        Returns a parameter for the shape of the spacetime region.
        '''
        return self._shape_params[key]

    def ShapeMaxEdgeHalf(self, dims: List[int]) -> float:
        '''
        Returns the half of the largest shape edge of the spacetime region.
        '''
        if self._shape_name == 'cube':
            return self._shape_params['edge'] / 2
        elif self._shape_name in ['diamond', 'ball']:
            return self._shape_params['radius']
        elif self._shape_name == 'cylinder':
            if dims.count(0) > 0:
                return max([self._shape_params['radius'],
                            self._shape_params['duration'] / 2])
            else:
                return self._shape_params['radius']
        else:  # cuboid
            return max(self._shape_params['edges'][dims]) / 2

    def ShapeLimits(self, dim: int) -> Tuple[float, float]:
        '''
        Returns a tuple of floats for the minimum (left) and maximum (right) 
        of a shape along coordinate dimension 'dim' (0-indexed).
        '''
        if (dim < 0) or (dim >= self.Dim):
            raise ValueError('The argument d is out of range, ' +
                             f'{dim} not in [0, {self.Dim}).')
        if (dim == 0) and (self._shape_name == 'cylinder'):
            l = self._shape_params['duration'] / 2
        elif self._shape_name == 'cube':
            l = self._shape_params['edge'] / 2
        elif self._shape_name == 'cuboid':
            l = self._shape_params['edges'][dim] / 2
        else:
            l = self._shape_params['radius']
        shift: float = self._shape_center[dim]
        return (-l + shift, l + shift)

    @staticmethod
    def _checkShape(dim: int, name: str,
                    **kwargs) -> Tuple[int, str, np.ndarray,
                                       Dict[str, np.float32]]:
        # set dimension:
        if dim < 1:
            raise ValueError('The value "dim" is out of range. ' +
                             'Must be an integer of at least 1.')
        # check name:
        if name == 'bicone':
            name = 'diamond'
        elif name not in ['ball', 'diamond', 'cylinder',
                          'cube', 'cuboid']:
            raise ValueError(f'A shape with name "{name}" is not supported.')
        # check center point:
        center: np.ndarray
        try:
            center = np.array(kwargs['center'], dtype=np.float32)
            if center.shape != (dim,):
                raise TypeError
        except KeyError:
            center = np.zeros(dim, dtype=np.float32)
        except TypeError:
            raise ValueError('The value for the key "center" has to be an ' +
                             f'Iterable of {dim} float values.')
        # check shape parameters:
        params = {}
        if name in ('cylinder'):
            try:
                value = kwargs['duration']
            except KeyError:
                value = 1.0
            params['duration'] = np.array(
                value, dtype=np.float32)
        if name in ('ball', 'diamond', 'cylinder'):
            try:
                value = kwargs['radius']
            except KeyError:
                value = 1.0
            params['radius'] = np.array(
                value, dtype=np.float32)
        elif name == 'cube':
            try:
                value = kwargs['edge']
            except KeyError:
                value = 1.0
            params['edge'] = np.array(
                value, dtype=np.float32)
        elif name == 'cuboid':
            try:
                value = kwargs['edges']
                params['edges'] = np.array(
                    value, dtype=np.float32)
            except KeyError:
                value = np.ones(dim, dtype=np.float32)
                params['edges'] = value
            if params['edges'].shape != (dim,):
                raise ValueError('The value for the key "edges" has ' +
                                 f'to be an Iterable of {dim} ' +
                                 'float values.')
            if any(params['edges'] <= 0.0):
                raise ValueError('At least one edge length is out of range.' +
                                 'Each must be a float greater than 0.')
        # return tuple
        return (dim, name, center, params)

    def setShape(self, dim: int, name: str, **kwargs) -> None:
        '''
        Sets the embedding shape, but does not adjust event coordinates.
        The dimension parameter dim must be an integer greater than zero.

        Accepted shape names with parameters:
        'ball' and 'diamond' ('bicone'), opt. keyword arg. 'radius' (float)
        'cylinder', opt. keyword args. 'radius' and 'duration' (floats)
        'cube', opt. keyword arg. 'edge' (float)
        'cuboid', opt. keyword arg. 'edges' (List[float, ...])
        Every shape also has an optional keyword arg. 'center' 
        (List[float, ...]) to set the central point with the zero vector 
        as default value. All other shape parameters default to 1.0. 

        Raises a ValueError if some input value is invalid.
        '''
        shape = EmbeddedCauset._checkShape(dim, name, **kwargs)
        self._dim = shape[0]
        self._shape_name = shape[1]
        self._shape_center = shape[2]
        self._shape_params = shape[3]
        # delete previously computed volume:
        if hasattr(self, '_volume'):
            delattr(self, '_volume')

    def calcVolume(self) -> float:
        '''
        Computes and returns the volume from the shape parameter.
        '''
        d: float = float(self.Dim)
        V: float = 0.0
        if self.ShapeName == 'ball':
            V = math.pi**(d / 2.0)
            V *= self.ShapeParam('radius')**d
            V /= math.gamma(d / 2.0 + 1.0)
        elif self.ShapeName in {'diamond', 'cylinder'}:
            r = self.ShapeParam('radius')
            V = math.pi**(d / 2.0 - 0.5)
            V *= r**(d - 1.0)
            V /= math.gamma(d / 2.0 + 0.5)
            if self.ShapeName == 'cylinder':
                V *= self.ShapeParam('duration')
            else:
                V *= 2 * r
                V /= d
        elif self.ShapeName == 'cube':
            V = self.ShapeParam('edge')**d
        elif self.ShapeName == 'cuboid':
            V = math.prod(self.ShapeParam('edges'))
        return V

    @property
    def Volume(self) -> float:
        '''
        Returns the volume of the shape.
        '''
        try:
            return self._volume
        except AttributeError:
            self._volume: float = self.calcVolume()
            return self._volume

    @property
    def Density(self) -> float:
        '''
        Returns the density of event in the shape volume, which is 
        measured in SI units and has the 'Dim'-fold power of the in 
        inverse physical length dimension [1 / L].
        '''
        return float(self.Card) / self.Volume

    @property
    def LengthScale(self) -> float:
        '''
        Returns the fundamental length scale. It is the inverse 
        of the 'Dim'-th root of the density.
        '''
        return (self.Volume / float(self.Card))**(1.0 / self.Dim)

    def create(self, coords: Union[Iterable[List[float]],
                                   Iterable[np.ndarray],
                                   np.ndarray],
               relate: bool = True) -> Set[CausetEvent]:
        '''
        Creates new event with the specified coordinates, adds them to 
        this instance and returns the new set of event.
        The argument 'coords' has to be List[List[float]], List[np.ndarray] 
        or np.ndarray (matrix with a coordinate row for each event).
        '''
        n: int = self.Card + 1
        eventSet: Set[CausetEvent] = {CausetEvent(label=i + n, coord=c)
                                      for i, c in enumerate(coords)}
        self._events.update(eventSet)
        if relate:
            self.relate()
        return eventSet

    def relate(self, link: bool=True) -> None:
        '''
        Resets the causal relations between all event based on their 
        embedding in the given spacetimes manifold.
        '''
        for e in self._events:
            e._prec = set()
            e._succ = set()
        eventList: List[CausetEvent] = list(self._events)
        eventList_len: int = len(eventList)
        for i, a in enumerate(eventList):
            for j in range(i + 1, eventList_len):
                b = eventList[j]
                isAB, isBA = self._iscausal(a.Coord, b.Coord)
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
        self.__coords = None

    def _set__coords(self, events: List[CausetEvent]) -> None:
        if hasattr(self, '__coords') and not (self.__coords is None):
            return
        coords = np.empty((len(events), self.Dim))
        for i, e in enumerate(events):
            coords[i, :] = e.Coord
        self.__coords = coords

    def relabel(self, dim: int = 0, descending: bool = False) -> None:
        '''
        Resets the labels of all event to ascending (default) or 
        descending integers (converted to str) corresponding to the 
        coordinate component in dimension 'dim'.
        '''
        eventList = list(self._events)
        self._set__coords(eventList)
        sorted_idx = np.argsort(self.__coords[:, dim])
        if descending:
            sorted_idx = np.flip(sorted_idx)
        for i, idx in enumerate(sorted_idx):
            eventList[idx].Label = i + 1

    def _init_plotting(self, plotArgs: Dict[str, Any],
                       plotEvents: List[CausetEvent]=None) -> \
            Tuple[List[CausetEvent], np.ndarray, Dict[str, Any]]:
        '''
        Handles the plot keyword parameter 'axislim' == 'shape' (Default) 
        and initialises the plotting coordinates.
        '''
        # handle keyword parameter axislim='shape' (default)
        dims: List[int]
        try:
            dims = plotArgs['dims']
        except KeyError:
            plotArgs.update({'dims': [1, 0]})
            dims = [1, 0]
        if ('axislim' not in plotArgs) or (plotArgs['axislim'] == 'shape'):
            if len(dims) > 2:
                edgehalf: float = self.ShapeMaxEdgeHalf(dims)
                center: List[float] = self.ShapeCenter
                center0: float = center[dims[0]]
                center1: float = center[dims[1]]
                center2: float = center[dims[2]]
                plotArgs.update({'axislim': {
                    'xlim': (center0 - edgehalf, center0 + edgehalf),
                    'ylim': (center1 - edgehalf, center1 + edgehalf),
                    'zlim': (center2 - edgehalf, center2 + edgehalf)}})
            else:
                plotArgs.update({'axislim': {
                    'xlim': self.ShapeLimits(dims[0]),
                    'ylim': self.ShapeLimits(dims[1])}})
        # get event list and copy coordinates to memory:
        eventList = list(self._events)
        self._set__coords(eventList)
        plotCoords: np.ndarray
        if plotEvents is None:
            plotEvents = eventList
            plotCoords = self.__coords
        else:
            plotCoords = self.__coords[plotEvents, :]
        return (plotEvents, plotCoords, plotArgs)

    def TimeslicePlotter(self, ax: plta.Axes=None,
                         eventList: List[CausetEvent]=None,
                         **kwargs) -> Callable[[np.ndarray], Dict[str, Any]]:
        '''
        Returns the core plotter function handle that requires an array of 
        one or two time values (as past and future time). The plotter function 
        creates a plot of this instance (or the list of event if eventList 
        is not None) into the Axes ax with plot parameters given by keyword 
        arguments. To fit the axis limits to the embedding shape, use the 
        keyword argument 'axislim' == 'shape' (Default).
        For all other plot options, see doc of 
        causet_plotting.plot_parameters.
        '''
        pEvents, pCoords, pArgs = self._init_plotting(kwargs, eventList)
        return cs_plt.Plotter(pEvents, pCoords, ax, **pArgs)

    def plot(self, ax: plta.Axes=None, eventList: List[CausetEvent]=None,
             **kwargs) -> Dict[str, Any]:
        '''
        Creates a plot of this instance (or the list of event if eventList 
        is not None) into the Axes ax with plot parameters given by keyword 
        arguments. To fit the axis limits to the embedding shape, use the 
        keyword argument 'axislim' == 'shape' (Default).
        For all other plot options, see doc of 
        causet_plotting.plot_parameters.
        '''
        pEvents, pCoords, pArgs = self._init_plotting(kwargs, eventList)
        return cs_plt.plot(pEvents, pCoords, ax, **pArgs)
