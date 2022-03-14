#!/usr/bin/env python
'''
Created on 1 Oct 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List, Dict, Tuple, Any, Union, Optional
import math
import numpy as np
from matplotlib import patches
from matplotlib.pyplot import gca
from matplotlib.axes import Axes
from matplotlib.patches import Patch

default_samplingsize: int = 128  # default number of edges for curved objects


class CoordinateShape(object):
    '''
    Handles a coordinate shape for embedding (and sprinkling) regions.
    '''

    _dim: int
    _name: str
    _center: np.ndarray
    _params: Dict[str, Any]
    _volume: float

    def __init__(self, dim: int, name: str, **kwargs) -> None:
        '''
        Sets the embedding shape, but does not adjust event coordinates.
        The dimension parameter dim must be an integer greater than zero.

        Accepted shape names (and parameters)
        -------------------------------------
        'ball' ('radius': float, default: 1.0, must be > 0.0,
                'hollow': float, default: 0.0, must be >= 0.0 and < 1.0)
            Ball shape in all spacetime coordinates. The parameter 'hollow' is 
            the fraction (between 0.0 and 1.0) of the radius of the ball to 
            give an empty interior.
        'bicone' ('radius': float, default: 1.0, must be > 0.0,
                  'hollow': float, default: 0.0, must be >= 0.0 and < 1.0)
            Ball shape in all space coordinates and conical to the past and 
            future. The parameter 'hollow' is the fraction (between 0.0 and 
            1.0) of the radius of the bicone to give an empty interior.
            (alternative shape name: 'diamond')
        'cylinder' ('radius': float, default: 1.0, must be > 0.0,
                    'duration': float, default: 2.0 * radius, must be > 0.0,
                    'hollow': float, default: 0.0, must be >= 0.0 and < 1.0)
            Ball shape in all space coordinates and straight along the time 
            coordinate for the length 'duration'. The parameter 'hollow' is 
            the fraction (between 0.0 and 1.0) of the radius and duration of 
            the cylinder to give an empty interior.
        'cube' ('edge': float, default: 1.0, must be > 0.0)
            Cube shape with the same edge length 'edge' in all spacetime 
            coordinates.
        'cuboid' ('edges': Iterable[float], default: [1.0, 1.0, ...], 
                                            must all be > 0.0)
            Cuboid shape with distinct edge lengths 'edges' in the respective 
            spacetime coordinates. The default edges yield a cube.
        All shapes have a further keyword argument, 'center' (Iterable[float]) 
        to specify the central point, with the zero vector as default value.

        Raises a ValueError if some input value is invalid.
        '''
        # set dimension:
        if dim < 1:
            raise ValueError('The value ''dim'' is out of range. ' +
                             'It must be an integer of at least 1.')
        self._dim = dim
        # set name:
        if name == 'diamond':
            name = 'bicone'
        elif name not in ['ball', 'bicone', 'cylinder',
                          'cube', 'cuboid']:
            raise ValueError('A shape with name ''' +
                             f'{name}'' is not supported.')
        self._name = name
        # set centre point:
        try:
            self._center = np.array(kwargs['center'], dtype=np.float32)
            if self._center.shape != (dim,):
                raise TypeError
        except KeyError:
            self._center = np.zeros(dim, dtype=np.float32)
        except TypeError:
            raise ValueError('The value for the key ''center'' has to be ' +
                             f'an Iterable of {dim} float values.')

        def param_rangecheck(p: str, maxValue: float = math.nan,
                             canBeZero: bool = False):
            isTooLow: bool = (canBeZero and (self._params[p] < 0.0)) or \
                (not canBeZero and self._params[p] <= 0.0)
            errorStr: str = 'greater than or equal to 0' if canBeZero else \
                'greater than 0'
            if math.isnan(maxValue):
                if isTooLow:
                    raise ValueError('The parameter ''' +
                                     f'{p}'' is out of range. ' +
                                     f'It must be a float {errorStr}.')
            elif isTooLow or (self._params[p] >= maxValue):
                raise ValueError('The parameter ''' +
                                 f'{p}'' is out of range. ' +
                                 f'It must be a float {errorStr} and '
                                 f'smaller than {maxValue}.')

        # set shape parameters:
        value: float
        self._params = {}
        if name in {'ball', 'bicone', 'cylinder'}:
            self._params['radius'] = np.array(kwargs.get('radius', 1.0),
                                              dtype=np.float32)
            param_rangecheck('radius')
            self._params['hollow'] = np.array(kwargs.get('hollow', 0.0),
                                              dtype=np.float32)
            param_rangecheck('hollow', 1.0, True)
            if name == 'cylinder':
                self._params['duration'] = np.array(
                    kwargs.get('duration', 2.0 * self._params['radius']),
                    dtype=np.float32)
                param_rangecheck('duration')
        elif name == 'cube':
            self._params['edge'] = np.array(kwargs.get('edge', 1.0),
                                            dtype=np.float32)
            param_rangecheck('edge')
        elif name == 'cuboid':
            self._params['edges'] = np.array(
                kwargs.get('edges', np.ones((dim,))), dtype=np.float32)
            if self._params['edges'].shape != (dim,):
                raise ValueError('The value for the key "edges" has ' +
                                 f'to be an Iterable of {dim} float values.')
            if any(x <= 0.0 for x in self._params['edges']):
                raise ValueError('At least one edge length is out of range.' +
                                 'Each must be a float greater than 0.')

    def __str__(self):
        if ('hollow' in self._params) and \
                (self._params['hollow'] > 0.0):
            return f'hollow {self._dim}-{self._name}'
        else:
            return f'{self._dim}-{self._name}'

    def __repr__(self):
        return f'{self._class__._name__}(' + \
            f'{self._dim}, {self._name}, center={self._center}, ' + \
            f'**{self._params})'

    @property
    def Dim(self) -> int:
        '''
        Returns the dimension of the spacetime (region).
        '''
        return self._dim

    @property
    def Name(self) -> str:
        '''
        Returns the shape name of the spacetime region.
        '''
        return self._name

    @property
    def Center(self) -> np.ndarray:
        '''
        Returns the shape center of the spacetime region.
        '''
        return self._center

    def Parameter(self, key: str) -> Any:
        '''
        Returns a parameter for the shape of the spacetime region.
        '''
        return self._params[key]

    @property
    def Volume(self) -> float:
        '''
        Returns the volume of the shape.
        (On the first call of this property, the volume is computed and stored 
        in an internal variable.)
        '''
        if not hasattr(self, '_volume'):
            V: float = 0.0
            r: float = self._params.get('radius', 0.0)
            isHollow: bool = self._params.get('hollow', 0.0) > 0.0
            if self._name == 'ball':
                V = r**self._dim
                if isHollow:
                    V -= (self._params['hollow'] * r)**self._dim
                V *= math.pi**(self._dim / 2.0) / \
                    math.gamma(self._dim / 2.0 + 1.0)
            elif self._name == 'cylinder':
                V = r**(self._dim - 1)
                if isHollow:
                    V -= (self._params['hollow'] * r)**(self._dim - 1)
                V *= math.pi**(self._dim / 2.0 - 0.5) / \
                    math.gamma(self._dim / 2.0 + 0.5)
                V *= self._params['duration']
            elif self._name == 'bicone':
                V = r**self._dim
                if isHollow:
                    V -= (self._params['hollow'] * r)**self._dim
                V *= math.pi**(self._dim / 2.0 - 0.5) / \
                    math.gamma(self._dim / 2.0 + 0.5)
                V *= 2.0 / self._dim
            elif self._name == 'cube':
                V = self._params['edge']**self._dim
            elif self._name == 'cuboid':
                V = math.prod(self._params['edges'])
            self._volume = V
        return self._volume

    def MaxEdgeHalf(self, dims: List[int]) -> float:
        '''
        Returns the half of the largest shape edge of the spacetime region.
        '''
        if self.Name == 'cube':
            return self._params['edge'] / 2
        elif self.Name in {'bicone', 'ball'}:
            return self._params['radius']
        elif self.Name == 'cylinder':
            if dims.count(0) > 0:
                return max([self._params['radius'],
                            self._params['duration'] / 2])
            else:
                return self._params['radius']
        else:  # cuboid
            return max(self._params['edges'][dims]) / 2

    def Limits(self, dim: int) -> Tuple[float, float]:
        '''
        Returns a tuple of floats for the minimum (left) and maximum (right) 
        of a shape along coordinate dimension 'dim' (0-indexed).
        '''
        if (dim < 0) or (dim >= self.Dim):
            raise ValueError('The argument d is out of range, ' +
                             f'{dim} not in [0, {self.Dim}).')
        if (dim == 0) and (self.Name == 'cylinder'):
            l = self._params['duration'] / 2
        elif self.Name == 'cube':
            l = self._params['edge'] / 2
        elif self.Name == 'cuboid':
            l = self._params['edges'][dim] / 2
        else:
            l = self._params['radius']
        shift: float = self._center[dim]
        return (-l + shift, l + shift)

    def plot(self, dims: List[int], plotAxes: Optional[Axes] = None,
             **kwargs) -> Union[Patch,
                                List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        '''
        Plots a cut through the center of the shape showing `dims` that 
        can be a list of 2 integers (for a 2D plot) or 3 integers (for a 3D 
        plot). The argument `plotAxes` specifies the plot axes object to be 
        used (by default the current axes of `matplotlib`). 
        As optional keyword arguments, the plotting options can be specified 
        that will overwrite the defaults that are for 2D:
        {'edgecolor': 'black', 'facecolor': 'black', 'alpha': 0.05}
        and for 3D:
        {'edgecolor': None, 'color': 'black', 'alpha': 0.05}
        The plotting options are passed to the Poly3DCollection object if it 
        is plotting in 3D or to the Patch object if it is plotting in 2D. 
        The Patch object is returned for a 2D plot and a list of surfaces is 
        returned for a 3D plot.
        '''
        is3d: bool = len(dims) == 3
        if plotAxes is None:
            plotAxes = gca(projection='3d') if is3d else \
                gca(projection=None)
        if is3d:
            plotoptions = {'edgecolor': None,
                           'color': 'black',
                           'alpha': 0.05}
        else:
            plotoptions = {'edgecolor': 'black',
                           'facecolor': 'black',
                           'alpha': 0.05}
        plotoptions.update(kwargs)
        timeaxis: int
        try:
            timeaxis = dims.index(0)
        except ValueError:
            timeaxis = -1
        hollow: float = 0.0
        if 'hollow' in self._params:
            hollow = self._params['hollow']
        if is3d:
            S: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            r: float
            if self.Name == 'cube':
                S = CubeSurface(self.Center[dims],
                                self._params['edge'])
            elif self.Name == 'ball' or \
                    ((timeaxis < 0) and (self.Name in {'bicone', 'cylinder'})):
                r = self._params['radius']
                S = BallSurface(self.Center[dims], r)
                if hollow > 0.0:
                    S = BallSurface(self.Center[dims],
                                    hollow * r) + S
            elif self.Name == 'cylinder':
                r = self._params['radius']
                h: float = self._params['duration']
                S = CylinderSurface(self.Center[dims],
                                    r, h, hollow, timeaxis)
            elif self.Name == 'bicone':
                r = self._params['radius']
                conecenter: np.ndarray = self.Center[dims]
                tip: np.ndarray = conecenter.copy()
                top: np.ndarray = conecenter.copy()
                tip[timeaxis] = conecenter[timeaxis] - r
                top[timeaxis] = conecenter[timeaxis] + r
                S = OpenConeSurface(tip, r, r, timeaxis) + \
                    OpenConeSurface(top, r, -r, timeaxis)
                if hollow > 0.0:
                    r *= hollow
                    tip[timeaxis] = conecenter[timeaxis] - r
                    top[timeaxis] = conecenter[timeaxis] + r
                    S = OpenConeSurface(tip, r, r, timeaxis) + \
                        OpenConeSurface(top, r, -r, timeaxis) + S
            else:  # cuboid
                S = CuboidSurface(self.Center[dims],
                                  self._params['edges'][dims])
            for XYZ in S:
                plotAxes.plot_surface(*XYZ, **plotoptions)
            return S
        else:
            p: Patch
            a: float
            b: float
            if self.Name == 'cube':
                a = self._params['edge']
                p = patches.Rectangle(
                    self.Center[dims] - 0.5 * np.array([a, a]), width=a,
                    height=a, **plotoptions)
            elif self.Name == 'ball' or \
                    ((timeaxis < 0) and (self.Name in {'bicone', 'cylinder'})):
                if hollow == 0.0:
                    p = patches.Circle(
                        self.Center[dims], self._params['radius'],
                        **plotoptions)
                else:
                    p = patches.Polygon(
                        CircleEdge(self.Center[dims], self._params['radius'],
                                   hollow),
                        **plotoptions)
            elif self.Name == 'cylinder':
                cyl: np.ndarray = CylinderCutEdge(
                    self.Center[dims], self._params['radius'],
                    self._params['duration'], hollow)
                if timeaxis == 0:
                    cyl = cyl[:, [1, 0]]
                p = patches.Polygon(cyl, **plotoptions)
            elif self.Name == 'bicone':
                p = patches.Polygon(
                    BiconeEdge(self.Center[dims], self._params['radius'],
                               hollow),
                    **plotoptions)
            else:  # cuboid
                edges: np.ndarray = self._params['edges'][dims]
                p = patches.Rectangle(
                    self.Center[dims] - 0.5 * edges, width=edges[0],
                    height=edges[1], **plotoptions)
            plotAxes.add_patch(p)
            return p


def RectangleEdge(left: float, bottom: float, width: float, height: float) -> \
        np.ndarray:
    '''
    Returns a matrix of size 5 x 2 that describes the edge of a rectangle.
    '''
    return np.array([[left, bottom],
                     [left + width, bottom],
                     [left + width, bottom + height],
                     [left, bottom + height],
                     [left, bottom]])


def CircleEdge(center: np.ndarray, radius: float, hollow: float = 0.0,
               samplingsize: int = -1) -> np.ndarray:
    '''
    Returns a matrix of size m x 2 that describes the edge of a circle.
    If the circle is hollow, `m = 2 * samplingsize + 3`, else it is 
    `samplingsize + 1`, where `samplingsize` is set to `default_samplingsize` 
    if non-positive.
    '''
    if samplingsize <= 0:
        samplingsize = default_samplingsize
    samplingsize = samplingsize + 1
    edge: np.ndarray = np.empty((samplingsize if hollow == 0.0
                                 else 2 * samplingsize + 1, 2))
    phi = np.linspace(0.0, 2.0 * np.pi, samplingsize)
    edge[:samplingsize, 0] = center[0] + radius * np.cos(phi)
    edge[:samplingsize, 1] = center[1] + radius * np.sin(phi)
    if hollow > 0.0:
        edge[samplingsize, :] = np.array([[np.nan, np.nan]])
        edge[(samplingsize + 1):, 0] = center[0] + \
            hollow * radius * np.cos(-phi)
        edge[(samplingsize + 1):, 1] = center[1] + \
            hollow * radius * np.sin(-phi)
    return edge


def EllipseEdge(center: np.ndarray, radii: np.ndarray, hollow: float = 0.0,
                samplingsize: int = -1) -> np.ndarray:
    '''
    Returns a matrix of size m x 2 that describes the edge of an ellipse.
    If the ellipse is hollow, `m = 2 * samplingsize + 3`, else it is 
    `samplingsize + 1`, where `samplingsize` is set to `default_samplingsize` 
    if non-positive.
    '''
    if samplingsize <= 0:
        samplingsize = default_samplingsize
    samplingsize = samplingsize + 1
    edge: np.ndarray = np.empty((samplingsize if hollow == 0.0
                                 else 2 * samplingsize + 1, 2))
    phi = np.linspace(0.0, 2.0 * np.pi, samplingsize)
    edge[:samplingsize, 0] = center[0] + radii[0] * np.cos(phi)
    edge[:samplingsize, 1] = center[1] + radii[1] * np.sin(phi)
    if hollow > 0.0:
        edge[samplingsize, :] = np.array([[np.nan, np.nan]])
        edge[(samplingsize + 1):, 0] = center[0] + \
            hollow * radii[0] * np.cos(-phi)
        edge[(samplingsize + 1):, 1] = center[1] + \
            hollow * radii[1] * np.sin(-phi)
    return edge


def BiconeEdge(center: np.ndarray, radius: float, hollow: float = 0.0) -> \
        np.ndarray:
    '''
    Returns a matrix of size m x 2 that describes the edge of a 2D bicone.
    If the bicone is hollow, `m = 11`, else it is `5`.
    '''
    edge: np.ndarray = np.array([[center[0], center[1] + radius],
                                 [center[0] - radius, center[1]],
                                 [center[0], center[1] - radius],
                                 [center[0] + radius, center[1]],
                                 [center[0], center[1] + radius]])
    if hollow > 0.0:
        radius = hollow * radius
        edge = np.concatenate(
            (edge, np.array([[np.nan, np.nan]]),
             np.array([[center[0] + radius, center[1]],
                       [center[0], center[1] - radius],
                       [center[0] - radius, center[1]],
                       [center[0], center[1] + radius],
                       [center[0] + radius, center[1]]])))
    return edge


def CylinderCutEdge(center: np.ndarray, radius: float, height: float,
                    hollow: float = 0.0) -> np.ndarray:
    '''
    Returns a matrix of size m x 2 that describes the (x, z) coordinates 
    of the edge of the y=0 cut through a cylinder aligned along the z-axis.
    If the cylinder is hollow, `m = 11`, else it is `5`.
    '''
    height_half: float = height / 2
    if hollow == 0.0:
        return RectangleEdge(center[0] - radius, center[1] - height_half,
                             2 * radius, height)
    else:
        return np.concatenate(
            (RectangleEdge(center[0] - radius, center[1] - height_half,
                           (1 - hollow) * radius, height),
             np.array([[np.nan, np.nan]]),
             RectangleEdge(center[0] + hollow * radius, center[1] - height_half,
                           (1 - hollow) * radius, height)))


def CuboidSurface(center: np.ndarray, edges: np.ndarray) -> \
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Returns the XYZ data that describes a cuboid surface.
    Each data matrix has size `5` x `4`.
    '''
    xed: float = edges[0]
    yed: float = edges[1]
    zed: float = edges[2]
    corner: np.ndarray = center - edges / 2
    S: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    S.append((np.array([[0.0]]) + corner[0],
              np.array([[0.0, 0.0, yed, yed]]) + corner[1],
              np.array([[0.0, zed, zed, 0.0]]).transpose() + corner[2]))
    S.append((np.array([[0.0, xed, xed, 0.0]]).transpose() + corner[0],
              np.array([[0.0]]) + corner[1],
              np.array([[0.0, 0.0, zed, zed]]) + corner[2]))
    S.append((np.array([[0.0, 0.0, xed, xed]]) + corner[0],
              np.array([[0.0, yed, yed, 0.0]]).transpose() + corner[1],
              np.array([[0.0]]) + corner[2]))
    S.append((np.array([[xed]]) + corner[0],
              np.array([[0.0, yed, yed, 0.0]]) + corner[1],
              np.array([[0.0, 0.0, zed, zed]]).transpose() + corner[2]))
    S.append((np.array([[0.0, 0.0, xed, xed]]).transpose() + corner[0],
              np.array([[yed]]) + corner[1],
              np.array([[0.0, zed, zed, 0.0]]) + corner[2]))
    S.append((np.array([[0.0, xed, xed, 0.0]]) + corner[0],
              np.array([[0.0, 0.0, yed, yed]]).transpose() + corner[1],
              np.array([[zed]]) + corner[2]))
    return S


def CubeSurface(center: np.ndarray, edge: float) -> \
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Returns a `CuboidSurface` object with the same `edge` length in all 
    dimensions.
    '''
    return CuboidSurface(center, np.array([edge, edge, edge]))


def BallSurface(center: np.ndarray, radius: float, samplingsize: int = -1) -> \
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Returns the XYZ data that describes a ball surface.
    Each data matrix has size `samplingsize + 1` x `(samplingsize + 1)/2`, 
    where `samplingsize` is set to `default_samplingsize` if non-positive.
    '''
    if samplingsize <= 0:
        samplingsize = default_samplingsize
    samplingsize = samplingsize + 1
    phi: np.ndarray = np.linspace(0.0, 2.0 * np.pi, samplingsize)
    theta: np.ndarray = np.linspace(0.0, np.pi, round(samplingsize / 2))
    return [(np.outer(np.cos(phi), np.sin(theta)) * radius + center[0],
             np.outer(np.sin(phi), np.sin(theta)) * radius + center[1],
             np.outer(np.ones(samplingsize), np.cos(theta)) * radius
             + center[2])]


def CylinderSurface(center: np.ndarray, radius: float, height: float,
                    hollow: float = 0.0, zaxis: int = 2,
                    samplingsize: int = -1) -> \
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Returns the XYZ data that describes a (hollow) cylinder surface.
    Each data matrix has size `4` x `samplingsize + 1` if `hollow == 0.0`, 
    or `5` x `samplingsize + 1` if `1.0 > hollow > 0.0`, where 
    `samplingsize` is set to `default_samplingsize` if non-positive.
    '''
    if samplingsize <= 0:
        samplingsize = default_samplingsize
    samplingsize = samplingsize + 1
    phi: np.ndarray = np.linspace(0.0, 2.0 * np.pi, samplingsize)
    radii: np.ndarray
    relZ: np.ndarray
    sampling: np.ndarray = np.ones(samplingsize)
    if hollow == 0.0:
        radii = radius * np.array([0.0, 1.0, 1.0, 0.0])
        relZ = np.array([-sampling, -sampling, sampling, sampling])
    else:
        radii = radius * np.array([hollow, 1.0, 1.0, hollow, hollow])
        relZ = np.array([-sampling, -sampling, sampling, sampling, -sampling])
    XYZ: Tuple[np.ndarray, np.ndarray, np.ndarray] = \
        (np.outer(radii, np.cos(phi)) + center[0],
         np.outer(radii, np.sin(phi)) + center[1],
         (height / 2) * relZ + center[2])
    # rotate:
    if zaxis == 0:
        XYZ = (XYZ[2], XYZ[0], XYZ[1])
    elif zaxis == 1:
        XYZ = (XYZ[1], XYZ[2], XYZ[0])
    return [XYZ]


def OpenConeSurface(tip: np.ndarray, radius: float, height: float,
                    zaxis: int = 2, samplingsize: int = -1) -> \
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    '''
    Returns the XYZ data that describes an open cone surface.
    Each data matrix has size `2` x `samplingsize + 1`, where 
    `samplingsize` is set to `default_samplingsize` if non-positive.
    '''
    if samplingsize <= 0:
        samplingsize = default_samplingsize
    samplingsize = samplingsize + 1
    phi: np.ndarray = np.linspace(0.0, 2.0 * np.pi, samplingsize)
    radii: np.ndarray = radius * np.array([0.0, 1.0])
    XYZ: Tuple[np.ndarray, np.ndarray, np.ndarray] = \
        (np.outer(radii, np.cos(phi)) + tip[0],
         np.outer(radii, np.sin(phi)) + tip[1],
         np.array([np.zeros(samplingsize), np.ones(samplingsize)]) * height
         + tip[2])
    # rotate:
    if zaxis == 0:
        XYZ = (XYZ[2], XYZ[0], XYZ[1])
    elif zaxis == 1:
        XYZ = (XYZ[1], XYZ[2], XYZ[0])
    return [XYZ]
