#!/usr/bin/env python
'''
Created on 02 Oct 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import Callable, Tuple, List, Dict, Any, Union, Optional
import numpy as np
import math
from matplotlib import pyplot as plt, patches, axes as plt_axes
from causets.shapes import CoordinateShape  # @UnresolvedImport
import causets.shapes as shp  # @UnresolvedImport
from causets.calculations import NewtonsMethod as Newton  # @UnresolvedImport

default_samplingsize: int = 128  # default value for sampling lightcones
causality_eps: float = 1e-12  # tolerance for causality rounding errors


class Spacetime(object):
    '''
    Super-class for the implementation of spacetimes.
    '''

    _dim: int
    _name: str
    _metricname: str
    _params: Dict[str, Any]

    def __init__(self) -> None:
        self._dim = 2
        self._name = ''
        self._metricname = 'unknown'
        self._params = {}

    def __str__(self):
        return f'{self._dim}-dimensional {self._name} spacetime'

    def __repr__(self):
        return f'{self.__class__.__name__}({self._dim}, **{self._params})'

    @property
    def Dim(self) -> int:
        '''
        Returns the dimension of the spacetime.
        '''
        return self._dim

    @property
    def Name(self) -> str:
        '''
        Returns the name of the spacetime.
        '''
        return self._name

    @property
    def MetricName(self) -> str:
        '''
        Returns the name of the coordinate representation of the metric.
        '''
        return self._metricname

    def Parameter(self, key: str) -> Any:
        '''
        Returns a parameter for the shape of the spacetime.
        '''
        return self._params[key]

    def DefaultShape(self) -> CoordinateShape:
        '''
        Returns the default coordinate shape of the embedding region in the 
        spacetime.
        '''
        return CoordinateShape(self.Dim, 'cylinder')

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:
        '''
        Returns a handle to a function to determine if two points x and y are 
        causally connected for the spacetime. 

        The function accepts coordinates x and y for two points and returns the 
        causality tuple (x <= y, x > y).
        '''
        # This is an example implementation for a spacetime.
        def isCausal(x: np.ndarray,
                     y: np.ndarray) -> Tuple[bool, bool]:
            t_delta: float = y[0] - x[0]
            return (t_delta >= 0.0, t_delta < 0.0)
        return isCausal

    def _T_slice_sampling(self, t: float, origin: np.ndarray,
                          samplingsize: int = -1) -> np.ndarray:
        '''
        Internal function for the time sampling array for a cone from `origin` 
        to time `t`.
        '''
        samplingsize = samplingsize if samplingsize >= 0 \
            else default_samplingsize
        return np.linspace(origin[0], t, samplingsize)

    def _XT_slice(self, t: float, origin: np.ndarray, xdim: int,
                  samplingsize: int = -1) -> np.ndarray:
        '''
        Internal function for the cone plotting from `origin` to time `t` 
        projected onto a X-T (space-time) plane with space dimension `xdim`. 
        '''
        raise NotImplementedError()

    def _XY_slice(self, t: float, origin: np.ndarray, dims: List[int],
                  samplingsize: int = -1) -> np.ndarray:
        '''
        Internal function for the cone plotting from `origin` to time `t` 
        projected onto a X-Y (space-space) plane with space dimensions `dims`.
        '''
        raise NotImplementedError()

    def _XYZ_slice(self, t: float, origin: np.ndarray, dims: List[int],
                   samplingsize: int = -1) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Internal function for the cone plotting from `origin` to time `t` 
        projected onto a X-Y-Z (space-space) plane with space dimensions `dims`.
        '''
        raise NotImplementedError()

    def ConePlotter(self, dims: List[int], plotting_params: Dict[str, Any],
                    timesign: float, axes: Optional[plt_axes.Axes] = None,
                    dynamicAlpha: Optional[Callable[[float], float]] = None,
                    samplingsize: int = -1) -> \
            Callable[[np.ndarray, float],
                     Union[patches.Patch,
                           List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        '''
        Returns a function handle to plot past (`timesign == -1`) or future 
        (`timesign == 1`) causal cones for the spacetime `self` into the axes 
        object `axes` (given by gca() by default, with projection='3d' if 
        len(dims) > 2) up to the coordinate time `timeslice` with plotting 
        parameters given in the dictionary `plotting_params`. The time 
        coordinate goes along the axis with index `timeaxis`. As optional 
        parameter `dynamicAlpha` a function (mapping float to float) can be 
        specified to compute the opacity of the cone from its size (radius). 

        The argument `dims` specifies the coordinate axes to be plotted. 
        It is a list of 2 or 3 integers, setting up a 2D or 3D plot.
        '''
        is3d: bool = len(dims) == 3
        _axes: plt_axes.Axes
        if axes is None:
            if is3d:
                _axes = plt.gca(projection='3d')
            else:
                _axes = plt.gca(projection=None)
        else:
            _axes = axes
        timeaxis: int
        try:
            timeaxis = dims.index(0)
        except ValueError:
            timeaxis = -1
        xaxis: int = (timeaxis + 1) % len(dims)
        yaxis: int = (timeaxis + 2) % len(dims)
        if samplingsize <= 0:
            samplingsize = default_samplingsize

        def cone(origin: np.ndarray, timeslice: float) -> \
                Union[patches.Patch,
                      List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
            '''
            Creates matplotlib surface plots for a 3D causal cone, or a patch 
            for a 2D causal cone added to the axes `axes`. The light emanates 
            from the coordinates `origin`, which has to be a `numpy` vector 
            with a length given by the coordinate dimensions of the spacetime.
            The lightcone reaches up to `timeslice`.

            The keyword argument `plotting_params` (with a dynamically 
            adjusted 'alpha' parameter) are passed to `plot_surface` methods if 
            it is 3D or to the Patch objects if it is 2D.

            The function returns `None` if no causal cone can be computed for 
            the respective input parameters.
            '''
            r: float = timesign * (timeslice - origin[0])
            if r <= 0.0:  # radius non-positive
                return None
            if dynamicAlpha is not None:
                conealpha = dynamicAlpha(r)
                if conealpha <= 0.0:
                    return None
                plotting_params.update({'alpha': conealpha})
            XY: np.ndarray = None
            T: np.ndarray
            samplesize_t: int
            if timeaxis >= 0:
                T = self._T_slice_sampling(timeslice, origin, samplingsize)
                samplesize_t = T.size
            if is3d:
                X: np.ndarray
                Y: np.ndarray
                Z: np.ndarray
                if timeaxis < 0:
                    X, Y, Z = self._XYZ_slice(timeslice, origin, dims,
                                              samplingsize)
                else:
                    for i, t in enumerate(T):
                        XY = self._XY_slice(t, origin,
                                            [dims[xaxis], dims[yaxis]],
                                            samplingsize)
                        if XY is None:
                            return None
                        elif i == 0:
                            s: Tuple[int, int] = (samplesize_t, XY.shape[0])
                            X, Y, Z = np.zeros(s), np.zeros(s), np.zeros(s)
                        X[i, :], Y[i, :], Z[i, :] = XY[:, 0], XY[:, 1], t
                    # rotate:
                    if timeaxis == 0:
                        X, Y, Z = Z, X, Y
                    elif timeaxis == 1:
                        X, Y, Z = Y, Z, X
                _axes.plot_surface(X, Y, Z, **plotting_params)
                return [(X, Y, Z)]
            else:
                if timeaxis < 0:
                    XY = self._XY_slice(timeslice, origin, dims,
                                        samplingsize)
                else:
                    XY = self._XT_slice(timeslice, origin, dims[xaxis],
                                        samplingsize)
                if XY is None:
                    return None
                # rotate:
                if timeaxis == 0:
                    XY = np.fliplr(XY)
                p: patches.Patch = patches.Polygon(XY, **plotting_params)
                _axes.add_patch(p)
                return p

        return cone


class FlatSpacetime(Spacetime):
    '''
    Initializes Minkowski spacetime for dim >= 1.
    As additional parameter, the spatial periodicity can be specified (using 
    the key 'period') as float (to be applied for all spatial directions 
    equally) or as tuple (with a float for each spatial dimension). A positive 
    float switches on the periodicity along the respective spatial direction, 
    using this value as period. The default is 0.0, no periodicity in any 
    direction. 
    '''

    def __init__(self, dim: int,
                 period: Union[float, Tuple[float, ...]] = 0.0) -> None:
        if dim < 1:
            raise ValueError('The spacetime dimension has to be at least 1.')
        super().__init__()
        self._dim = dim
        self._name = 'flat'
        self._metricname = 'Minkowski'

        _isPeriodic: bool
        _periods: np.ndarray = None
        if isinstance(period, float):
            _isPeriodic = period > 0.0
            if _isPeriodic:
                _periods = np.array([period] * (dim - 1))
        elif isinstance(period, tuple) and (len(period) == dim - 1):
            _isPeriodic = any(p > 0.0 for p in period)
            _periods = period
        else:
            raise ValueError('The parameter ''periodic'' has to be of ' +
                             'type float, or a tuple of float with the ' +
                             'same length as spatial dimensions.')
        self._params['isPeriodic'] = _isPeriodic
        if _isPeriodic:
            self._params['period'] = _periods

    def __repr__(self):
        _period: Tuple[float, ...] = self.Parameter('period')
        return f'{self.__class__.__name__}({self._dim}, period={_period})'

    def DefaultShape(self) -> CoordinateShape:
        return CoordinateShape(self.Dim, 'cube') \
            if self.Parameter('isPeriodic') \
            else CoordinateShape(self.Dim, 'diamond')

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:
        if self.Dim == 1:
            return super().Causality()
        if not self.Parameter('isPeriodic'):
            if self.Dim == 2:
                def isCausal_flat2D(x: np.ndarray,
                                    y: np.ndarray) -> Tuple[bool, bool]:
                    t_delta: float = y[0] - x[0]
                    isCausal: bool = abs(t_delta) >= \
                        abs(y[1] - x[1]) - causality_eps
                    return ((t_delta >= 0.0) and isCausal,
                            (t_delta < 0.0) and isCausal)
                return isCausal_flat2D
            else:
                def isCausal_flat(x: np.ndarray,
                                  y: np.ndarray) -> Tuple[bool, bool]:
                    t_delta: float = y[0] - x[0]
                    isCausal: bool = np.square(t_delta) >= \
                        sum(np.square(y[1:] - x[1:])) - causality_eps
                    return ((t_delta >= 0.0) and isCausal,
                            (t_delta < 0.0) and isCausal)
                return isCausal_flat
        else:
            _period: np.ndarray = self.Parameter('period')
            if self.Dim == 2:
                def isCausal_flat2Dperiodic(x: np.ndarray,
                                            y: np.ndarray) -> Tuple[bool, bool]:
                    t_delta: float = y[0] - x[0]
                    r_delta: float = abs(y[1] - x[1])
                    if _period[0] > 0.0:
                        r_delta = min(r_delta, _period[0] - r_delta)
                    isCausal: bool = abs(t_delta) >= \
                        abs(r_delta) - causality_eps
                    return ((t_delta >= 0.0) and isCausal,
                            (t_delta < 0.0) and isCausal)
                return isCausal_flat2Dperiodic
            else:
                def isCausal_flatperiodic(x: np.ndarray,
                                          y: np.ndarray) -> Tuple[bool, bool]:
                    t_delta: float = y[0] - x[0]
                    r2_delta: float = 0.0
                    for i in range(1, self.Dim):
                        r_delta_i: float = abs(y[i] - x[i])
                        if _period[i - 1] > 0.0:
                            r_delta_i = min(r_delta_i,
                                            _period[i - 1] - r_delta_i)
                        r2_delta += r_delta_i**2
                    isCausal: bool = np.square(t_delta) >= \
                        r2_delta - causality_eps
                    return ((t_delta >= 0.0) and isCausal,
                            (t_delta < 0.0) and isCausal)
                return isCausal_flatperiodic

    def ConePlotter(self, dims: List[int], plotting_params: Dict[str, Any],
                    timesign: float, axes: Optional[plt_axes.Axes] = None,
                    dynamicAlpha: Optional[Callable[[float], float]] = None,
                    samplingsize: int = -1) -> \
            Callable[[np.ndarray, float],
                     Union[patches.Patch,
                           List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        is3d: bool = len(dims) == 3
        _axes: plt_axes.Axes
        if axes is None:
            if is3d:
                _axes = plt.gca(projection='3d')
            else:
                _axes = plt.gca(projection=None)
        else:
            _axes = axes
        timeaxis: int
        try:
            timeaxis = dims.index(0)
        except ValueError:
            timeaxis = -1
        isPeriodic: bool = self.Parameter('isPeriodic')
        shifts: List[np.ndarray]
        k_x: float = 0.0
        x_s: List[float]
        k_y: float = 0.0
        y_s: List[float]
        if is3d:
            k_z: float = 0.0
            z_s: List[float]
            if isPeriodic:
                if timeaxis == 0:
                    k_y = self.Parameter('period')[dims[1] - 1]
                    k_z = self.Parameter('period')[dims[2] - 1]
                elif timeaxis == 1:
                    k_z = self.Parameter('period')[dims[2] - 1]
                    k_x = self.Parameter('period')[dims[0] - 1]
                elif timeaxis == 2:
                    k_x = self.Parameter('period')[dims[0] - 1]
                    k_y = self.Parameter('period')[dims[1] - 1]
                else:
                    k_x = self.Parameter('period')[dims[0] - 1]
                    k_y = self.Parameter('period')[dims[1] - 1]
                    k_z = self.Parameter('period')[dims[2] - 1]
                x_s = [-k_x, 0.0, k_x] if k_x > 0.0 else [0.0]
                y_s = [-k_y, 0.0, k_y] if k_y > 0.0 else [0.0]
                z_s = [-k_z, 0.0, k_z] if k_z > 0.0 else [0.0]
                shifts = [np.array([x, y, z])
                          for x in x_s for y in y_s for z in z_s]
            else:
                shifts = [np.array([0.0, 0.0, 0.0])]
        else:
            if isPeriodic:
                if timeaxis == 0:
                    k_y = self.Parameter('period')[dims[1] - 1]
                elif timeaxis == 1:
                    k_x = self.Parameter('period')[dims[0] - 1]
                else:
                    k_x = self.Parameter('period')[dims[0] - 1]
                    k_y = self.Parameter('period')[dims[1] - 1]
                x_s = [-k_x, 0.0, k_x] if k_x > 0.0 else [0.0]
                y_s = [-k_y, 0.0, k_y] if k_y > 0.0 else [0.0]
                shifts = [np.array([x, y])
                          for x in x_s for y in y_s]
            else:
                shifts = [np.array([0.0, 0.0])]
        if samplingsize <= 0:
            samplingsize = default_samplingsize

        def cone(origin: np.ndarray, timeslice: float) -> \
                Union[patches.Patch,
                      List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
            r: float = timesign * (timeslice - origin[0])
            if r <= 0.0:  # radius non-positive
                return None
            if dynamicAlpha is not None:
                conealpha = dynamicAlpha(r)
                if conealpha <= 0.0:
                    return None
                plotting_params.update({'alpha': conealpha})
            origin = origin[dims]
            if is3d:
                XYZ_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
                if timeaxis < 0:
                    for s in shifts:
                        XYZ_list = XYZ_list + shp.BallSurface(
                            origin - s, r, samplingsize)
                else:
                    for s in shifts:
                        XYZ_list = XYZ_list + shp.OpenConeSurface(
                            origin - s, r, timesign * r,
                            timeaxis, samplingsize)
                for XYZ in XYZ_list:
                    _axes.plot_surface(*XYZ, **plotting_params)
                return XYZ_list
            else:
                XY: np.array = None
                XYpart: np.array
                for i, s in enumerate(shifts):
                    if timeaxis == 0:
                        XYpart = np.array(
                            [origin - s,
                             np.array([timeslice, origin[1] - r]) - s,
                             np.array([timeslice, origin[1] + r]) - s,
                             origin - s])
                    elif timeaxis == 1:
                        XYpart = np.array(
                            [origin - s,
                             np.array([origin[0] + r, timeslice]) - s,
                             np.array([origin[0] - r, timeslice]) - s,
                             origin - s])
                    else:
                        XYpart = shp.CircleEdge(origin - s, radius=r,
                                                samplingsize=samplingsize)
                    XY = XYpart if i == 0 \
                        else np.concatenate(
                            (XY, np.array([[np.nan, np.nan]]), XYpart))
                p: patches.Patch = patches.Polygon(XY, **plotting_params)
                _axes.add_patch(p)
                return p

        return cone


class _dSSpacetime(Spacetime):
    '''
    Implementation of the base class for de Sitter and Anti-de Sitter 
    spacetimes.
    '''

    _alpha: float
    _alpha_sq: float

    def __init__(self, dim: int, alpha: float = 1.0) -> None:
        '''
        Initializes (Anti) de Sitter spacetime for dim >= 2.
        It is parametrized by `alpha` as float.
        '''
        if dim < 2:
            raise ValueError('The spacetime dimension has to be at least 2.')
        super().__init__()
        self._dim = dim
        self._metricname = 'static'
        self._alpha = alpha
        self._alpha_sq = alpha**2

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:
        raise NotImplementedError()

    def _XT_slice2(self, t: float, t0: float,
                   x0: float) -> Tuple[float, float]:
        raise NotImplementedError()

    def _XT_slice(self, t: float, origin: np.ndarray, xdim: int,
                  samplingsize: int = -1) -> np.ndarray:
        T: np.ndarray = self._T_slice_sampling(t, origin, samplingsize)
        XT: np.ndarray = np.zeros((2 * T.size - 1, 2))
        if origin.size == 2:
            x0: float = origin[1] / self._alpha
            if abs(x0) >= 1.0:
                return None
            for i, t in enumerate(T):
                r: Tuple[float, float] = self._XT_slice2(t, origin[0], x0)
                XT[-i, 0], XT[i, 0] = min(r), max(r)
                XT[-i, 1], XT[i, 1] = t, t
        else:
            t_X: np.ndarray
            for i, t in enumerate(T):
                x_min: float = np.PINF
                x_max: float = np.NINF
                for ydim in range(1, origin.size):
                    if ydim == xdim:
                        continue
                    t_X = self._XY_slice(t, origin, [xdim, ydim], samplingsize)
                    if t_X is None:
                        return None
                    x_min = np.min([x_min, np.min(t_X[:, 0])])
                    x_max = np.max([x_max, np.max(t_X[:, 0])])
                XT[-i, 0], XT[i, 0] = x_min, x_max
                XT[-i, 1], XT[i, 1] = t, t
        return XT


class deSitterSpacetime(_dSSpacetime):
    '''
    Implementation of de Sitter spacetimes, which are globally hyperbolic.
    '''

    def __init__(self, dim: int, r_dS: float = 1.0) -> None:
        '''
        Initializes de Sitter spacetime for dim >= 2.
        It is parametrized by the radius of the cosmological radius `r_dS` as 
        float.
        '''
        super().__init__(dim, r_dS)
        self._name = 'de Sitter'
        if r_dS > 0.0:
            self._params = {'r_dS': r_dS}
        else:
            raise ValueError('The cosmological radius ' +
                             'has to be positive.')

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:

        def isCausal_dS(x: np.ndarray,
                        y: np.ndarray) -> Tuple[bool, bool]:
            r2_x: float = sum(np.square(x[1:]))
            r2_y: float = sum(np.square(y[1:]))
            if (r2_x >= self._alpha_sq) or (r2_y >= self._alpha_sq):
                return (False, False)
            amp_x: float = math.sqrt(self._alpha_sq - r2_x)
            amp_y: float = math.sqrt(self._alpha_sq - r2_y)
            x0_x: float = amp_x * math.sinh(x[0] / self._alpha)
            x1_x: float = amp_x * math.cosh(x[0] / self._alpha)
            x0_y: float = amp_y * math.sinh(y[0] / self._alpha)
            x1_y: float = amp_y * math.cosh(y[0] / self._alpha)
            x0_delta: float = x0_y - x0_x
            isCausal: bool = x0_delta**2 >= \
                sum(np.square(y[1:] - x[1:])) + (x1_y - x1_x)**2 - \
                causality_eps
            return ((x0_delta >= 0.0) and isCausal,
                    (x0_delta < 0.0) and isCausal)
        return isCausal_dS

    def _XT_slice2(self, t: float, t0: float,
                   x0: float) -> Tuple[float, float]:
        return (self._alpha * np.tanh(np.arctanh(x0) - (t - t0) / self._alpha),
                self._alpha * np.tanh(np.arctanh(x0) + (t - t0) / self._alpha))

    def _XY_slice(self, t: float, origin: np.ndarray, dims: List[int],
                  samplingsize: int = -1) -> np.ndarray:
        # Define initial values `(t0, r0, phi0, Theta0)` from `origin`,
        # where Theta0 is the product of all remaining angular components (for
        # example, sin(theta0) in 4 dimensions) that is not yet implemented.
        if samplingsize <= 0:
            samplingsize = default_samplingsize
        r0_sq: float = np.sum(np.square(origin[1:])) / self._alpha_sq
        if r0_sq >= 1.0:
            return None
        r0: float = np.sqrt(r0_sq)
        t0: float = origin[0]
        phi0: float = np.arctan2(origin[dims[1]], origin[dims[0]])
        Theta0: float = 1.0
        # Computation of the ellipse in a x-y-coordinate system that is
        # rotated by -phi0 so that the center of the ellipse is on the x-axis:
        #   x_i: inner x-intercept
        #   x_o: outer x-intercept
        #   x_c: x-coordinate of the ellipse center
        #   a: semi-minor (along x-axis)
        #   b: semi-major (parallel to y-axis)
        delta_t: float = np.abs(t - t0) / self._alpha
        x_i: float = self._alpha * np.tanh(np.arctanh(r0) - delta_t)
        x_o: float = self._alpha * np.tanh(np.arctanh(r0) + delta_t)
        x_c: float = 0.5 * (x_o + x_i)
        a: float = 0.5 * (x_o - x_i)
        b: float = a
        if r0 > 0.0 and a > 0.0:
            # if actual ellipse then b is not a
            delta_t_tanh: float = np.tanh(delta_t)
            r_s: float = self._alpha * \
                np.sqrt((1 - r0_sq) * delta_t_tanh**2 + r0_sq)
            arg_s: float = np.arctan(
                np.sqrt(1 - r0_sq) * delta_t_tanh / r0) / Theta0
            b = np.sqrt((r_s * np.sin(arg_s))**2 /
                        (1 - ((r_s * np.cos(arg_s) - x_c) / a)**2))
        XY: np.ndarray = shp.EllipseEdge(
            np.array([x_c, 0.0]), np.array([a, b]))
        # Rotation of the ellipse to the angle phi0:
        R: np.ndarray = np.array([[np.cos(phi0), -np.sin(phi0)],
                                  [np.sin(phi0), np.cos(phi0)]])
        XY = np.matmul(XY, R.T)
        return XY


class AntideSitterSpacetime(_dSSpacetime):
    '''
    Implementation of anti-de Sitter spacetimes. Note that anti-de Sitter 
    spacetimes are not globally hyperbolic, so that infinite sprinkles on AdS 
    can break the finiteness axiom of causal sets.

    The past- and future- causal cone plotting is not implemented.
    '''

    def __init__(self, dim: int, r_AdS: float = 0.5) -> None:
        '''
        Initializes Anti-de Sitter spacetime for dim >= 2.
        It is parametrized by `r_AdS` as float.
        '''
        super().__init__(dim, r_AdS)
        self._name = 'Anti-de Sitter'
        if r_AdS > 0.0:
            self._params = {'r_AdS': r_AdS}
        else:
            raise ValueError('The Anti-de Sitter parameter ' +
                             'has to be positive.')

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:

        def isCausal_AdS(x: np.ndarray,
                         y: np.ndarray) -> Tuple[bool, bool]:
            amp_x: float = math.sqrt(self._alpha_sq + sum(np.square(x[1:])))
            amp_y: float = math.sqrt(self._alpha_sq + sum(np.square(y[1:])))
            x0_x: float = amp_x * math.sin(x[0] / self._alpha)
            x1_x: float = amp_x * math.cos(x[0] / self._alpha)
            x0_y: float = amp_y * math.sin(y[0] / self._alpha)
            x1_y: float = amp_y * math.cos(y[0] / self._alpha)
            x0_delta: float = x0_y - x0_x
            isCausal: bool = x0_delta**2 + (x1_y - x1_x)**2 >= \
                sum(np.square(y[1:] - x[1:])) - causality_eps
            return ((x0_delta >= 0.0) and isCausal,
                    (x0_delta < 0.0) and isCausal)
        return isCausal_AdS


class BlackHoleSpacetime(Spacetime):
    '''
    Implementation of black hole spacetimes, which are globally hyperbolic.
    '''

    _r_S: float

    def __init__(self, dim: int, r_S: float = 0.5,
                 metric: str = 'Eddington-Finkelstein') -> None:
        '''
        Initializes a black hole spacetime for dim == 2.
        It is parametrized by the radius of the event horizon `r_S` as float. 
        For the metric, 'Eddington-Finkelstein' (default) and 'Schwarzschild' 
        are implemented.
        '''
        if dim < 2:
            raise ValueError('The spacetime dimension has to be at least 2.')
        elif dim > 2:
            raise ValueError(f'The dimension {dim} is not implemented.')
        super().__init__()
        self._dim = dim
        self._name = 'black hole'
        if metric in {'Eddington-Finkelstein', 'Schwarzschild'}:
            self._metricname = metric
        else:
            raise ValueError(f'The metric {metric} is not implemented.')
        if r_S > 0.0:
            self._params = {'r_S': r_S}
            self._r_S = r_S
        else:
            raise ValueError(f'The Schwarzschild radius has to be positive.')

    def __repr__(self):
        return f'{self.__class__.__name__}({self._dim}, ' + \
            f'r_S={self._r_S}, metric={self._metricname})'

    def _light_EF(self, t0: float, r0: float, ingoing: bool = False,
                  derivative: int = 0) -> Callable[[Any], Any]:
        '''
        Returns the -cone function (and its derivatives) for `ingoing` and 
        outgoing radial lightrays starting at (t0, r0), for the 
        Eddington-Finkelstein metric.
        '''
        if derivative == 0:
            if ingoing:
                def _lightray_in(r: Any) -> Any:
                    return r0 - r + t0
                return _lightray_in
            else:
                def _lightray_out(r: Any) -> Any:
                    return r - r0 + t0 + 2.0 * self._r_S * \
                        np.log(np.abs((r - self._r_S) / (r0 - self._r_S)))
                return _lightray_out
        elif derivative == 1:
            if not ingoing:
                def _lightray_d1_out(r: Any) -> Any:
                    return 1.0 + 2.0 * self._r_S / (r - self._r_S)
                return _lightray_d1_out

        raise NotImplementedError()

    def _light_S(self, t0: float, r0: float, ingoing: bool = False,
                 derivative: int = 0) -> Callable[[Any], Any]:
        '''
        Returns the -cone function (and its derivatives) for `ingoing` and 
        outgoing radial lightrays starting at (t0, r0), for the Schwarzschild 
        metric.
        '''
        if derivative == 0:
            if ingoing:
                def _lightray_in(r: Any) -> Any:
                    return r0 - r + t0 - self._r_S * \
                        np.log(np.abs((r - self._r_S) / (r0 - self._r_S)))
                return _lightray_in
            else:
                def _lightray_out(r: Any) -> Any:
                    return r - r0 + t0 + self._r_S * \
                        np.log(np.abs((r - self._r_S) / (r0 - self._r_S)))
                return _lightray_out
        elif derivative == 1:
            if ingoing:
                def _lightray_d1_in(r: Any) -> Any:
                    return -1.0 - self._r_S / (r - self._r_S)
                return _lightray_d1_in
            else:
                def _lightray_d1_out(r: Any) -> Any:
                    return 1.0 + self._r_S / (r - self._r_S)
                return _lightray_d1_out

        raise NotImplementedError()

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:
        if self.Dim == 1:
            return super().Causality()
        isSchwarzschildMetric: bool = self._metricname == 'Schwarzschild'

        if self.Dim == 2:
            _func: Callable[[Any], Any] = self._light_S(0.0, 0.0) \
                if isSchwarzschildMetric \
                else self._light_EF(0.0, 0.0)

            def isCausal_BH2D(x: np.ndarray,
                              y: np.ndarray) -> Tuple[bool, bool]:
                if x[1] * y[1] < 0.0:
                    return (False, False)
                t_delta: float = y[0] - x[0]
                r_x: float = abs(x[1])
                r_y: float = abs(y[1])
                isSwapped: bool = False
                if isSchwarzschildMetric and ((r_x < self._r_S) or
                                              (r_y < self._r_S)):
                    # Schwarzschild metric and at least one is inside
                    isSwapped = r_x < r_y  # order s.t. r_y <= r_x
                else:  # EddFin metric, or both points are outside
                    isSwapped = t_delta < 0  # order s.t. t_y >= t_x
                if isSwapped:  # swap
                    x, y = y, x
                    r_x, r_y = r_y, r_x
                    t_delta = -t_delta
                isCausal: bool = False
                t_out: float = _func(r_y) - _func(r_x)
                t_in: float = -t_out if isSchwarzschildMetric \
                    else r_x - r_y
                if r_y <= r_x <= self._r_S:  # x is inside, y < x
                    isCausal = t_out >= t_delta >= t_in
                elif self._r_S <= r_x >= r_y:  # x is outside, y < x
                    isCausal = t_delta >= t_in
                elif self._r_S <= r_x <= r_y:  # x is outside, y > x
                    isCausal = t_delta >= t_out
                return (False, isCausal) if isSwapped else (isCausal, False)
            return isCausal_BH2D

        raise NotImplementedError()

    def _XT_slice(self, t: float, origin: np.ndarray, xdim: int,
                  samplingsize: int = -1) -> np.ndarray:
        if samplingsize <= 0:
            samplingsize = default_samplingsize
        r_0: float = abs(origin[xdim])
        r_out: float = r_0
        r_in: float = r_0
        XT: np.ndarray
        X: np.ndarray
        n: int
        f_out: Callable[[float], float]
        fd_out: Callable[[float], float]
        if self._metricname == 'Schwarzschild':
            f_in: Callable[[float], float] = \
                self._light_S(origin[0], r_0, True)
            fd_in: Callable[[float], float] = \
                self._light_S(origin[0], r_0, True, 1)
            f_out = self._light_S(origin[0], r_0, False)
            fd_out = self._light_S(origin[0], r_0, False, 1)
            if r_0 == self._r_S:  # on the horizon
                XT = np.zeros((4, 2))
                XT[0, :] = [origin[xdim], origin[0]]
                XT[1, :] = [origin[xdim], t]
                XT[2, :] = [0.0, t]
                XT[3, :] = [0.0, origin[0]]
            elif r_0 < self._r_S:  # inside the horizon
                XT = np.zeros((2 * samplingsize, 2))
                if t > origin[0]:  # pointing inside
                    X = np.linspace(r_0, 0.0, samplingsize)
                    XT[:samplingsize, 0] = np.copysign(X, origin[xdim])
                    XT[:samplingsize, 1] = f_out(X)
                    X = np.flip(X)
                    XT[-samplingsize:, 0] = np.copysign(X, origin[xdim])
                    XT[-samplingsize:, 1] = f_in(X)
                else:  # pointing outside
                    r_out = Newton(f_in, fd_in, r_0, t, xmin=self._r_S)
                    X = np.linspace(r_out, r_0, samplingsize)
                    XT[:samplingsize, 0] = np.copysign(X, origin[xdim])
                    XT[:samplingsize, 1] = f_in(X)
                    r_in = Newton(f_out, fd_out, r_0, t,
                                  xmin=0.0, xmax=self._r_S)
                    X = np.linspace(r_0, r_in, samplingsize)
                    XT[-samplingsize:, 0] = np.copysign(X, origin[xdim])
                    XT[-samplingsize:, 1] = f_out(X)
            else:  # outside the horizon
                XT = np.zeros((2 * samplingsize, 2))
                r_out = Newton(f_out, fd_out, r_0, t, xmin=self._r_S)
                X = np.linspace(r_0, r_out, samplingsize)
                XT[:samplingsize, 0] = np.copysign(X, origin[xdim])
                XT[:samplingsize, 1] = f_out(X)
                r_in = Newton(f_in, fd_in, r_0, t, xmin=self._r_S)
                X = np.linspace(r_in, r_0, samplingsize)
                XT[-samplingsize:, 0] = np.copysign(X, origin[xdim])
                XT[-samplingsize:, 1] = f_in(X)
                t_sing: float = f_in(0.0)
                if (t_sing < t) and (origin[0] < t):
                    r_in = Newton(f_in, fd_in, self._r_S / 2.0, t,
                                  xmin=0.0, xmax=self._r_S)
                    XT_inner: np.ndarray = np.zeros((samplingsize + 1, 2))
                    X = np.linspace(r_in, 0.0, samplingsize)
                    XT_inner[:samplingsize, 0] = np.copysign(X, origin[xdim])
                    XT_inner[:samplingsize, 1] = f_in(X)
                    XT_inner[-1, :] = [0.0, t]
                    XT = np.concatenate((XT, [[np.nan, np.nan]], XT_inner))
        else:  # Eddington-Finkelstein metric
            r_in = origin[0] - t + r_0
            n = 2 if r_in < 0.0 else 1
            if r_0 == self._r_S:
                XT = np.zeros((n + 2, 2))
                XT[0, :] = [origin[xdim], origin[0]]
                XT[1, :] = [origin[xdim], t]
            else:
                f_out = self._light_EF(origin[0], r_0, False)
                fd_out = self._light_EF(origin[0], r_0, False, 1)
                if r_0 < self._r_S:  # future -cone is limited
                    t = min(t, f_out(0.0))
                XT = np.zeros((samplingsize + n, 2))
                if r_0 < self._r_S:
                    r_out = Newton(f_out, fd_out, r_0, t,
                                   xmin=0.0, xmax=self._r_S)
                else:
                    r_out = Newton(f_out, fd_out, 1.5 * self._r_S, t,
                                   xmin=self._r_S)
                X = np.linspace(r_0, r_out, samplingsize)
                XT[:-n, 0] = np.copysign(X, origin[xdim])
                XT[:-n, 1] = f_out(X)
            if r_in < 0.0:
                XT[-2, :] = [0.0, t]
                XT[-1, :] = [0.0, origin[0] + r_0]
            else:
                XT[-1, :] = [np.copysign(1.0, origin[xdim]) * r_in, t]
        return XT
