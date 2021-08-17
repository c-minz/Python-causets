#!/usr/bin/env python
'''
Created on 1 Oct 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List, Dict, Tuple, Any, Union
import math
import numpy as np
from matplotlib import pyplot as plt, patches, axes as plt_axes

default_samplingsize: int = 128  # default number of edges for curved objects


class CoordinateShape(object):
    '''
    Handles a coordinate shape for embedding (and sprinkling) regions.
    '''

    __dim: int
    __name: str
    __center: np.ndarray
    __params: Dict[str, Any]
    __volume: float

    def __init__(self, dim: int, name: str, **kwargs) -> None:
        '''
        Sets the embedding shape, but does not adjust event coordinates.
        The dimension parameter dim must be an integer greater than zero.

        Accepted shape names (and parameters):
        'ball' with keyword argument 'radius' (float), 1.0 by default.
        Ball shape in all spacetime coordinates.

        'diamond' (or 'bicone') with keyword argument 'radius' (float), 
        1.0 by default.
        Ball shape in all space coordinates and conical to the past and 
        future.

        'cylinder' with keyword arguments 'radius' (float), 1.0 by default, 
        and 'duration' (float), twice of 'radius' by default. 
        Ball shape in all space coordinates and straight along the time 
        coordinate for the length 'duration'.

        As additional keyword argument 'hollow' (float) can be specified 
        (0.0 by default) to get a hollow version of the shapes 'ball', 
        'diamond' or 'cylinder' with an empty interior such that the 
        parameter specifies the fraction of the radius that is hollow.

        'cube' with keyword argument 'edge' (float), 1.0 by default.
        Cube shape with the same edge length 'edge' in all spacetime 
        coordinates.

        'cuboid' with keyword arguments 'edges' (iterable of float), unit 
        vector by default.
        Cuboid shape with distinct edge lengths 'edges' in the respective 
        spacetime coordinates.

        A further keyword argument 'center' (iterable of float) specifies 
        the central point with the zero vector as default value. 

        Raises a ValueError if some input value is invalid.
        '''
        # set dimension:
        if dim < 1:
            raise ValueError('The value ''dim'' is out of range. ' +
                             'It must be an integer of at least 1.')
        self.__dim = dim
        # set name:
        if name == 'bicone':
            name = 'diamond'
        elif name not in ['ball', 'diamond', 'cylinder',
                          'cube', 'cuboid']:
            raise ValueError('A shape with name ''' +
                             f'{name}'' is not supported.')
        self.__name = name
        # set center point:
        try:
            self.__center = np.array(kwargs['center'], dtype=np.float32)
            if self.__center.shape != (dim,):
                raise TypeError
        except KeyError:
            self.__center = np.zeros(dim, dtype=np.float32)
        except TypeError:
            raise ValueError('The value for the key ''center'' has to be ' +
                             f'an Iterable of {dim} float values.')

        def param_rangecheck(p: str, maxValue: float = math.nan,
                             canBeZero: bool = False):
            isTooLow: bool = (canBeZero and (self.__params[p] < 0.0)) or \
                (not canBeZero and self.__params[p] <= 0.0)
            errorStr: str
            if canBeZero:
                errorStr = 'greater than or equal to 0'
            else:
                errorStr = 'greater than 0'
            if math.isnan(maxValue):
                if isTooLow:
                    raise ValueError('The parameter ''' +
                                     f'{p}'' is out of range. ' +
                                     f'It must be a float {errorStr}.')
            elif isTooLow or (self.__params[p] >= maxValue):
                raise ValueError('The parameter ''' +
                                 f'{p}'' is out of range. ' +
                                 f'It must be a float {errorStr} and '
                                 f'smaller than {maxValue}.')

        # set shape parameters:
        isHollow: bool = False
        value: float
        self.__params = {}
        if name in {'ball', 'diamond', 'cylinder'}:
            try:
                value = kwargs['radius']
            except KeyError:
                value = 1.0
            self.__params['radius'] = np.array(value, dtype=np.float32)
            param_rangecheck('radius')
            try:
                value = kwargs['hollow']
            except KeyError:
                value = 0.0
            isHollow = value > 0.0
            self.__params['hollow'] = np.array(value,
                                               dtype=np.float32)
            param_rangecheck('hollow', 1.0, True)
        if 'cylinder' in name:
            try:
                value = kwargs['duration']
            except KeyError:
                value = 2.0 * self.__params['radius']
            self.__params['duration'] = np.array(value, dtype=np.float32)
            param_rangecheck('duration')
        elif name == 'cube':
            try:
                value = kwargs['edge']
            except KeyError:
                value = 1.0
            self.__params['edge'] = np.array(value, dtype=np.float32)
            param_rangecheck('edge')
        elif name == 'cuboid':
            try:
                value = kwargs['edges']
                self.__params['edges'] = np.array(value, dtype=np.float32)
            except KeyError:
                value = np.ones(dim, dtype=np.float32)
                self.__params['edges'] = value
            if self.__params['edges'].shape != (dim,):
                raise ValueError('The value for the key "edges" has ' +
                                 f'to be an Iterable of {dim} ' +
                                 'float values.')
            if any(self.__params['edges'] <= 0.0):
                raise ValueError('At least one edge length is out of range.' +
                                 'Each must be a float greater than 0.')
        # set volume:
        V: float = 0.0
        r: float
        d_f: float = float(dim)
        if name == 'ball':
            r = self.__params['radius']
            V = r**dim
            if isHollow:
                V -= (self.__params['hollow'] * r)**dim
            V *= math.pi**(d_f / 2.0) / math.gamma(d_f / 2.0 + 1.0)
        elif name == 'cylinder':
            r = self.__params['radius']
            V = r**(dim - 1)
            if isHollow:
                V -= (self.__params['hollow'] * r)**(dim - 1)
            V *= math.pi**(d_f / 2.0 - 0.5) / math.gamma(d_f / 2.0 + 0.5)
            V *= self.__params['duration']
        elif name == 'diamond':
            r = self.__params['radius']
            V = r**dim
            if isHollow:
                V -= (self.__params['hollow'] * r)**dim
            V *= math.pi**(d_f / 2.0 - 0.5) / math.gamma(d_f / 2.0 + 0.5)
            V *= 2.0 / d_f
        elif name == 'cube':
            V = self.__params['edge']**dim
        elif name == 'cuboid':
            V = math.prod(self.__params['edges'])
        self.__volume = V

    def __str__(self):
        if ('hollow' in self.__params) and \
                (self.__params['hollow'] > 0.0):
            return f'hollow {self.__dim}-{self.__name}'
        else:
            return f'{self.__dim}-{self.__name}'

    def __repr__(self):
        return f'{self.__class__.__name__}(' + \
            f'{self.__dim}, {self.__name}, center={self.__center}, ' + \
            f'**{self.__params})'

    @property
    def Dim(self) -> int:
        '''
        Returns the dimension of the spacetime (region).
        '''
        return self.__dim

    @property
    def Name(self) -> str:
        '''
        Returns the shape name of the spacetime region.
        '''
        return self.__name

    @property
    def Center(self) -> np.ndarray:
        '''
        Returns the shape center of the spacetime region.
        '''
        return self.__center

    def Parameter(self, key: str) -> Any:
        '''
        Returns a parameter for the shape of the spacetime region.
        '''
        return self.__params[key]

    @property
    def Volume(self) -> float:
        '''
        Returns the volume of the shape.
        '''
        return self.__volume

    def MaxEdgeHalf(self, dims: List[int]) -> float:
        '''
        Returns the half of the largest shape edge of the spacetime region.
        '''
        if self.Name == 'cube':
            return self.__params['edge'] / 2
        elif self.Name in {'diamond', 'ball'}:
            return self.__params['radius']
        elif self.Name == 'cylinder':
            if dims.count(0) > 0:
                return max([self.__params['radius'],
                            self.__params['duration'] / 2])
            else:
                return self.__params['radius']
        else:  # cuboid
            return max(self.__params['edges'][dims]) / 2

    def Limits(self, dim: int) -> Tuple[float, float]:
        '''
        Returns a tuple of floats for the minimum (left) and maximum (right) 
        of a shape along coordinate dimension 'dim' (0-indexed).
        '''
        if (dim < 0) or (dim >= self.Dim):
            raise ValueError('The argument d is out of range, ' +
                             f'{dim} not in [0, {self.Dim}).')
        if (dim == 0) and (self.Name == 'cylinder'):
            l = self.__params['duration'] / 2
        elif self.Name == 'cube':
            l = self.__params['edge'] / 2
        elif self.Name == 'cuboid':
            l = self.__params['edges'][dim] / 2
        else:
            l = self.__params['radius']
        shift: float = self.__center[dim]
        return (-l + shift, l + shift)

    def plot(self, dims: List[int], axes: plt_axes.Axes = None,
             **kwargs) -> Union[patches.Patch,
                                List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        '''
        Plots a cut through the center of the shape showing `dims` that 
        can be a list of 2 integers (for a 2D plot) or 3 integers (for a 3D 
        plot). The argument `axes` specifies the plot axes object to be used 
        (by default the current axes of `matplotlib`). 
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
        if axes is None:
            if is3d:
                axes = plt.gca(projection='3d')
            else:
                axes = plt.gca(projection=None)
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
        if 'hollow' in self.__params:
            hollow = self.__params['hollow']
        if is3d:
            S: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            r: float
            if self.Name == 'cube':
                S = CubeSurface(self.Center[dims],
                                self.__params['edge'])
            elif self.Name == 'ball' or \
                    ((timeaxis < 0) and (self.Name in {'diamond', 'cylinder'})):
                r = self.__params['radius']
                S = BallSurface(self.Center[dims], r)
                if hollow > 0.0:
                    S = BallSurface(self.Center[dims],
                                    hollow * r) + S
            elif self.Name == 'cylinder':
                r = self.__params['radius']
                h: float = self.__params['duration']
                S = CylinderSurface(self.Center[dims],
                                    r, h, hollow, timeaxis)
            elif self.Name == 'diamond':
                r = self.__params['radius']
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
                                  self.__params['edges'][dims])
            for XYZ in S:
                axes.plot_surface(*XYZ, **plotoptions)
            return S
        else:
            p: patches.Patch
            a: float
            b: float
            if self.Name == 'cube':
                a = self.__params['edge']
                p = patches.Rectangle(self.Center[dims] -
                                      0.5 * np.array([a, a]),
                                      width=a, height=a, **plotoptions)
            elif self.Name == 'ball' or \
                    ((timeaxis < 0) and (self.Name in {'diamond', 'cylinder'})):
                if hollow == 0.0:
                    p = patches.Circle(self.Center[dims],
                                       self.__params['radius'], **plotoptions)
                else:
                    p = patches.Polygon(CircleEdge(self.Center[dims],
                                                   self.__params['radius'],
                                                   hollow),
                                        **plotoptions)
            elif self.Name == 'cylinder':
                cyl: np.ndarray = CylinderCutEdge(self.Center[dims],
                                                  self.__params['radius'],
                                                  self.__params['duration'],
                                                  hollow)
                if timeaxis == 0:
                    cyl = cyl[:, [1, 0]]
                p = patches.Polygon(cyl, **plotoptions)
            elif self.Name == 'diamond':
                p = patches.Polygon(DiamondEdge(self.Center[dims],
                                                self.__params['radius'], hollow),
                                    **plotoptions)
            else:  # cuboid
                edges: np.ndarray = self.__params['edges'][dims]
                p = patches.Rectangle(self.Center[dims] - 0.5 * edges,
                                      width=edges[0], height=edges[1],
                                      **plotoptions)
            axes.add_patch(p)
            return p


def RectangleEdge(left: float, bottom: float, width: float,
                  height: float) -> np.ndarray:
    '''
    Returns a matrix of size 5 x 2 that describes the edge of a rectangle.
    '''
    return np.array([[left, bottom],
                     [left + width, bottom],
                     [left + width, bottom + height],
                     [left, bottom + height],
                     [left, bottom]])


def CircleEdge(center: np.ndarray, radius: float,
               hollow: float = 0.0, samplingsize: int = -1) -> np.ndarray:
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


def EllipseEdge(center: np.ndarray, radii: np.ndarray,
                hollow: float = 0.0, samplingsize: int = -1) -> np.ndarray:
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


def DiamondEdge(center: np.ndarray, radius: float,
                hollow: float = 0.0) -> np.ndarray:
    '''
    Returns a matrix of size m x 2 that describes the edge of a 2D diamond.
    If the diamond is hollow, `m = 11`, else it is `5`.
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


def BallSurface(center: np.ndarray, radius: float,
                samplingsize: int = -1) -> \
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
