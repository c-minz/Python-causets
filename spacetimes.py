#!/usr/bin/env python
'''
Created on 02 Oct 2020

@author: Christoph Minz
'''
from __future__ import annotations
from typing import Callable, Tuple, List, Dict, Any, Union
import numpy as np
import math
from matplotlib import patches, axes
from shapes import BallSurface, OpenConeSurface, CoordinateShape, CircleEdge
from builtins import int


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
        self._name = 'flat'
        self._metricname = 'Minkowski'
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

        The function accepts coordinates x and y for two points and returns 
        the causality tuple (x <= y, x > y).
        '''
        # This is an example implementation for a spacetime.
        def isCausal(x: np.ndarray,
                     y: np.ndarray) -> Tuple[bool, bool]:
            t_delta: float = y[0] - x[0]
            return (t_delta >= 0.0, t_delta < 0.0)
        return isCausal

    def LightconePlotter(self, ax: axes.Axes, dims: List[int],
                         plotting_params: Dict[str, Any],
                         timesign: float, timeslice: float,
                         dynamicAlpha: Callable[[float], float] = None) -> \
            Callable[[np.ndarray],
                     Union[patches.Patch,
                           List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        '''
        Returns a function handle to plot past (`timesign == -1`) or future 
        (`timesign == 1`) light-cones for the spacetime `self` into the axes 
        object `ax` up to the coordinate time `timeslice` with plotting 
        parameters given in the dictionary `plotting_params`. The time 
        coordinate goes along the axis with index `timeaxis`. As optional 
        parameter `dynamicAlpha` a function (mapping float to float) can be 
        specified to compute the opacity of the cone from its size (radius). 

        The argument `dims` specifies the coordinate axes to be plotted. 
        It is a list of 2 or 3 integers, setting up a 2D or 3D plot.
        '''
        is3d: bool = len(dims) == 3
        timeaxis: int
        try:
            timeaxis = dims.index(0)
        except ValueError:
            timeaxis = -1

        def lightcone(origin: np.ndarray) -> \
                Union[patches.Patch,
                      List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
            '''
            Creates matplotlib surface plots for a 3D lightcone, or a patch 
            for a 2D lightcone added to the axes `ax`. The light emanates 
            from the coordinates `origin`, which has to be a `numpy` vector 
            with a length given by the coordinate dimensions of the spacetime.
            The keyword argument `plotting_params` (with a dynamically 
            adjusted 'alpha' parameter) are passed to `plot_surface` methods 
            if it is 3D or to the Patch objects if it is 2D.

            The function returns `None` if no lightcone can be computed for 
            the respective input parameters.
            '''
            r: float = timesign * (timeslice - origin[0])
            if r <= 0.0:  # radius non-positive
                return None
            elif timeaxis > 0:  # time axis visible
                return None
            if dynamicAlpha is not None:
                conealpha = dynamicAlpha(r)
                if conealpha <= 0.0:
                    return None
                plotting_params.update({'alpha': conealpha})
            origin = origin[dims]
            if is3d:
                XYZ: Tuple[np.ndarray, np.ndarray, np.ndarray] = \
                    BallSurface(origin, r)
                ax.plot_surface(*XYZ, **plotting_params)
                return [XYZ]
            else:
                p: patches.Patch = patches.Circle(origin, radius=r,
                                                  **plotting_params)
                ax.add_patch(p)
                return p

        return lightcone


class FlatSpacetime(Spacetime):
    '''
    Initializes Minkowski spacetime for dim >= 1.
    As additional parameter, the spatial periodicity can be specified (by 
    the key 'period') as float (to be applied for all spatial 
    directions equally) or as tuple (with a float for each spatial 
    dimension). A positive float switches on the periodicity along the 
    respective spatial direction, using this value as period. 
    The default is 0.0, no periodicity in any direction. 
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
                    isConnected: bool = abs(t_delta) >= abs(y[1] - x[1])
                    return ((t_delta >= 0.0) and isConnected,
                            (t_delta < 0.0) and isConnected)
                return isCausal_flat2D
            else:
                def isCausal_flat(x: np.ndarray,
                                  y: np.ndarray) -> Tuple[bool, bool]:
                    t_delta: float = y[0] - x[0]
                    isConnected: bool = np.square(t_delta) >= \
                        sum(np.square(y[1:] - x[1:]))
                    return ((t_delta >= 0.0) and isConnected,
                            (t_delta < 0.0) and isConnected)
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
                    isConnected: bool = abs(t_delta) >= abs(r_delta)
                    return ((t_delta >= 0.0) and isConnected,
                            (t_delta < 0.0) and isConnected)
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
                    isConnected: bool = np.square(t_delta) >= r2_delta
                    return ((t_delta >= 0.0) and isConnected,
                            (t_delta < 0.0) and isConnected)
                return isCausal_flatperiodic

    def LightconePlotter(self, ax: axes.Axes, dims: List[int],
                         plotting_params: Dict[str, Any],
                         timesign: float, timeslice: float,
                         dynamicAlpha: Callable[[float], float] = None) -> \
            Callable[[np.ndarray],
                     Union[patches.Patch,
                           List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        is3d: bool = len(dims) == 3
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

        def lightcone(origin: np.ndarray) -> \
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
                XYZ_list: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = \
                    [BallSurface(origin - s, r) if timeaxis < 0
                     else OpenConeSurface(origin - s, r,
                                          timesign * r, timeaxis)
                     for s in shifts]
                for XYZ in XYZ_list:
                    ax.plot_surface(*XYZ, **plotting_params)
                return XYZ_list
            elif (timeaxis < 0) and (len(shifts) == 1):
                p_circle: patches.Patch = patches.Circle(origin, radius=r,
                                                         **plotting_params)
                ax.add_patch(p_circle)
                return p_circle
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
                        XYpart = CircleEdge(origin - s, radius=r)
                    XY = XYpart if i == 0 \
                        else np.concatenate(
                            (XY, np.array([[np.nan, np.nan]]), XYpart))
                p: patches.Patch = patches.Polygon(XY, **plotting_params)
                ax.add_patch(p)
                return p

        return lightcone


class deSitterSpacetime(Spacetime):
    '''
    Implementation of de Sitter spacetimes.
    '''

    def __init__(self, dim: int,
                 r_dS: float = 1.0) -> None:
        '''
        Initializes de Sitter spacetime for dim >= 2.
        It is parametrized by the radius of the cosmological radius `r_dS` 
        as float.
        '''
        if dim < 2:
            raise ValueError('The spacetime dimension has to be at least 2.')
        super().__init__()
        self._dim = dim
        self._name = 'de Sitter'
        self._metricname = 'static'
        if r_dS > 0.0:
            self._params = {'r_dS': r_dS}
        else:
            raise ValueError('The cosmological radius ' +
                             'has to be positive.')

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:
        _r_dS: float = self.Parameter('r_dS')
        _r_dS_2: float = _r_dS**2

        def isCausal_dS(x: np.ndarray,
                        y: np.ndarray) -> Tuple[bool, bool]:
            r2_x: float = sum(np.square(x[1:]))
            r2_y: float = sum(np.square(y[1:]))
            if (r2_x >= _r_dS_2) or (r2_y >= _r_dS_2):
                return (False, False)
            amp_x: float = math.sqrt(_r_dS_2 - r2_x)
            amp_y: float = math.sqrt(_r_dS_2 - r2_y)
            x0_x: float = amp_x * math.sinh(x[0] / _r_dS)
            x1_x: float = amp_x * math.cosh(x[0] / _r_dS)
            x0_y: float = amp_y * math.sinh(y[0] / _r_dS)
            x1_y: float = amp_y * math.cosh(y[0] / _r_dS)
            x0_delta: float = x0_y - x0_x
            isConnected: bool = x0_delta**2 >= \
                sum(np.square(y[1:] - x[1:])) + (x1_y - x1_x)**2
            return ((x0_delta >= 0.0) and isConnected,
                    (x0_delta < 0.0) and isConnected)
        return isCausal_dS

    def LightconePlotter(self, ax: axes.Axes, dims: List[int],
                         plotting_params: Dict[str, Any],
                         timesign: float, timeslice: float,
                         dynamicAlpha: Callable[[float], float] = None) -> \
            Callable[[np.ndarray],
                     Union[patches.Patch,
                           List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        is3d: bool = len(dims) == 3
        timeaxis: int
        xaxis: int
        yaxis: int
        try:
            timeaxis = dims.index(0)
        except ValueError:
            timeaxis = -1
        xaxis = dims[(timeaxis + 1) % len(dims)]
        yaxis = dims[(timeaxis + 2) % len(dims)]
        _r_dS: float = self.Parameter('r_dS')
        _r_dS_sq: float = _r_dS**2

        def XYtimeslice(t: float, origin: np.ndarray,
                        Theta: float = 1.0, samplesize: int = 30) -> np.ndarray:
            '''
            Returns a numpy matrix of (x, y) coordinate pairs that describe 
            the lightcone slice at time t, where `origin` is the embedding 
            coordinates of the cone tip. If there exists no coneslice at 
            time coordinate t, `None` is returned.
            '''
            # Define initial values `(t0, r0, phi0, Theta)` from `origin`,
            # where Theta is the product of all remaining angular components,
            # like sin(theta) in 4 dimensions.
            r0_sq: float = np.sum(np.square(origin[1:])) / _r_dS_sq
            if r0_sq == 0.0:
                r0_sq = 0.0001
            elif r0_sq >= 1.0:
                return None
            t0: float = origin[0]
            phi0: float = np.arctan2(origin[yaxis], origin[xaxis])
            Theta0: float = origin[xaxis] / (np.sqrt(r0_sq) * np.cos(phi0))
            if (Theta < 0.0) or (Theta > 1.0):
                Theta = Theta0
            # Define initial value range for the angular velocities
            # omega = d phi / d t, scaled by the initial radius, beta forwards
            # and beta backwards. The factor `(1.0 - 1.0 / samplesize**2)` is to
            # avoid a 90deg angle that could give a divergent term (and
            # duplicate data points).
            beta_fw: np.ndarray = np.linspace(0.001,
                                              ((1.0 - 1.0 / samplesize**2) *
                                               r0_sq / (1 - r0_sq))**0.75,
                                              samplesize)**(1.0 / 1.5)
            beta_bw: np.ndarray = -np.flip(beta_fw[1:])
            # Compute the solution in the 4 quadrants:
            XY: np.ndarray = np.empty((4 * samplesize - 1, 2))
            i_start: int = 0
            for s, beta in [(1, beta_fw), (-1, beta_bw),
                            (-1, beta_fw), (1, beta_bw)]:
                beta_sq: np.ndarray = np.square(beta)
                rho_beta: np.ndarray = np.sqrt(r0_sq * (1 + beta_sq) - beta_sq)
                rhot_tanh = np.tanh(np.arctanh(rho_beta) +
                                    np.copysign(t - t0, s) / _r_dS)
                r = np.sqrt((np.square(rhot_tanh) + beta_sq) / (1 + beta_sq))
                delta_phi = (np.arctan(rhot_tanh / beta) -
                             np.arctan(rho_beta / beta)) / Theta
                i_end: int = i_start + len(beta)
                XY[i_start:i_end, 0] = r * Theta0 * np.cos(phi0 + delta_phi)
                XY[i_start:i_end, 1] = r * Theta0 * np.sin(phi0 + delta_phi)
                i_start = i_end
            XY[-1, :] = XY[0, :]
            return XY

        def lightcone(origin: np.ndarray) -> \
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
            XY: np.ndarray = None
            if is3d:
                if timeaxis < 0:
                    return None
                samplesize_phi: int = 30
                samplesize_t: int = min(32 * round(r / _r_dS) + 2, 100)
                T: np.ndarray = np.linspace(origin[0], timeslice, samplesize_t)
                X: np.ndarray = np.zeros(
                    (samplesize_t, 4 * samplesize_phi - 1))
                Y: np.ndarray = np.zeros(
                    (samplesize_t, 4 * samplesize_phi - 1))
                Z: np.ndarray = np.zeros(
                    (samplesize_t, 4 * samplesize_phi - 1))
                for i, t in enumerate(T):
                    XY = XYtimeslice(t, origin, samplesize=samplesize_phi)
                    if XY is None:
                        return None
                    X[i, :] = XY[:, 0]
                    Y[i, :] = XY[:, 1]
                    Z[i, :] = t
                # rotate:
                if timeaxis == 0:
                    X, Y, Z = Z, X, Y
                elif timeaxis == 1:
                    X, Y, Z = Y, Z, X
                ax.plot_surface(X, Y, Z, **plotting_params)
                return [(X, Y, Z)]
            else:
                if timeaxis < 0:
                    XY = XYtimeslice(timeslice, origin)
                if XY is None:
                    return None
                p: patches.Patch = patches.Polygon(XY, **plotting_params)
                ax.add_patch(p)
                return p

        return lightcone


class AntideSitterSpacetime(Spacetime):
    '''
    Implementation of Anti-de Sitter spacetimes.
    '''

    def __init__(self, dim: int,
                 r_AdS: float = 0.5) -> None:
        '''
        Initializes Anti-de Sitter spacetime for dim >= 2.
        It is parametrized by `r_AdS` as float.
        '''
        if dim < 2:
            raise ValueError('The spacetime dimension has to be at least 2.')
        super().__init__()
        self._dim = dim
        self._name = 'Anti-de Sitter'
        self._metricname = 'static'
        if r_AdS > 0.0:
            self._params = {'r_AdS': r_AdS}
        else:
            raise ValueError('The Anti-de Sitter parameter ' +
                             'has to be positive.')

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:
        _r_AdS: float = self.Parameter('r_AdS')
        _r_AdS_2: float = _r_AdS**2

        def isCausal_AdS(x: np.ndarray,
                         y: np.ndarray) -> Tuple[bool, bool]:
            amp_x: float = math.sqrt(_r_AdS_2 + sum(np.square(x[1:])))
            amp_y: float = math.sqrt(_r_AdS_2 + sum(np.square(y[1:])))
            x0_x: float = amp_x * math.sin(x[0] / _r_AdS)
            x1_x: float = amp_x * math.cos(x[0] / _r_AdS)
            x0_y: float = amp_y * math.sin(y[0] / _r_AdS)
            x1_y: float = amp_y * math.cos(y[0] / _r_AdS)
            x0_delta: float = x0_y - x0_x
            isConnected: bool = x0_delta**2 + (x1_y - x1_x)**2 >= \
                sum(np.square(y[1:] - x[1:]))
            return ((x0_delta >= 0.0) and isConnected,
                    (x0_delta < 0.0) and isConnected)
        return isCausal_AdS

    def LightconePlotter(self, ax: axes.Axes, dims: List[int],
                         plotting_params: Dict[str, Any],
                         timesign: float, timeslice: float,
                         dynamicAlpha: Callable[[float], float] = None) -> \
            Callable[[np.ndarray],
                     Union[patches.Patch,
                           List[Tuple[np.ndarray, np.ndarray, np.ndarray]]]]:
        is3d: bool = len(dims) == 3
        timeaxis: int
        xaxis: int
        yaxis: int
        try:
            timeaxis = dims.index(0)
        except ValueError:
            timeaxis = -1
        xaxis = dims[(timeaxis + 1) % len(dims)]
        yaxis = dims[(timeaxis + 2) % len(dims)]
        _r_AdS: float = self.Parameter('r_AdS')
        _r_AdS_sq: float = _r_AdS**2

        def XYtimeslice(t: float, origin: np.ndarray,
                        Theta: float = 1.0, samplesize: int = 30) -> np.ndarray:
            '''
            Returns a numpy matrix of (x, y) coordinate pairs that describe 
            the lightcone slice at time t, where `origin` is the embedding 
            coordinates of the cone tip. If there exists no coneslice at 
            time coordinate t, `None` is returned.
            '''
            # Define initial values `(t0, r0, phi0, Theta)` from `origin`,
            # where Theta is the product of all remaining angular components,
            # like sin(theta) in 4 dimensions.
            r0_sq: float = np.sum(np.square(origin[1:])) / _r_AdS_sq
            if r0_sq == 0.0:
                r0_sq = 0.0001
            t0: float = origin[0]
            phi0: float = np.arctan2(origin[yaxis], origin[xaxis])
            Theta0: float = origin[xaxis] / (np.sqrt(r0_sq) * np.cos(phi0))
            if (Theta < 0.0) or (Theta > 1.0):
                Theta = Theta0
            # Define initial value range for the angular velocities
            # omega = d phi / d t, scaled by the initial radius, beta forwards
            # and beta backwards. The factor `(1.0 - 1.0 / samplesize**2)` is to
            # avoid a 90deg angle that could give a divergent term (and
            # duplicate data points).
            beta_fw: np.ndarray = np.linspace(0.001,
                                              ((1.0 - 1.0 / samplesize**2) *
                                               r0_sq / (1 + r0_sq))**0.75,
                                              samplesize)**(1.0 / 1.5)
            beta_bw: np.ndarray = -np.flip(beta_fw[1:])
            # Compute the solution in the 4 quadrants:
            XY: np.ndarray = np.empty((4 * samplesize - 1, 2))
            i_start: int = 0
            for s, beta in [(1, beta_fw), (-1, beta_bw),
                            (-1, beta_fw), (1, beta_bw)]:
                beta_sq: np.ndarray = np.square(beta)
                rho_beta: np.ndarray = np.sqrt(r0_sq * (1 - beta_sq) - beta_sq)
                rhot_tanh = np.tanh(np.arctanh(rho_beta) +
                                    np.copysign(t - t0, s) / _r_AdS)
                r = np.sqrt((np.square(rhot_tanh) + beta_sq) / (1 - beta_sq))
                delta_phi = (np.arctan(rhot_tanh / beta) -
                             np.arctan(rho_beta / beta)) / Theta
                i_end: int = i_start + len(beta)
                XY[i_start:i_end, 0] = r * Theta0 * np.cos(phi0 + delta_phi)
                XY[i_start:i_end, 1] = r * Theta0 * np.sin(phi0 + delta_phi)
                i_start = i_end
            XY[-1, :] = XY[0, :]
            return XY

        def lightcone(origin: np.ndarray) -> \
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
            XY: np.ndarray = None
            if is3d:
                if timeaxis < 0:
                    return None
                samplesize_phi: int = 30
                samplesize_t: int = min(32 * round(r / _r_AdS) + 2, 100)
                T: np.ndarray = np.linspace(origin[0], timeslice, samplesize_t)
                X: np.ndarray = np.zeros(
                    (samplesize_t, 4 * samplesize_phi - 1))
                Y: np.ndarray = np.zeros(
                    (samplesize_t, 4 * samplesize_phi - 1))
                Z: np.ndarray = np.zeros(
                    (samplesize_t, 4 * samplesize_phi - 1))
                for i, t in enumerate(T):
                    XY = XYtimeslice(t, origin, samplesize=samplesize_phi)
                    if XY is None:
                        return None
                    X[i, :] = XY[:, 0]
                    Y[i, :] = XY[:, 1]
                    Z[i, :] = t
                # rotate:
                if timeaxis == 0:
                    X, Y, Z = Z, X, Y
                elif timeaxis == 1:
                    X, Y, Z = Y, Z, X
                ax.plot_surface(X, Y, Z, **plotting_params)
                return [(X, Y, Z)]
            else:
                if timeaxis < 0:
                    XY = XYtimeslice(timeslice, origin)
                if XY is None:
                    return None
                p: patches.Patch = patches.Polygon(XY, **plotting_params)
                ax.add_patch(p)
                return p

        return lightcone


class BlackHoleSpacetime(Spacetime):
    '''
    Implementation of black hole spacetimes.
    '''

    def __init__(self, dim: int,
                 r_S: float = 0.5,
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
        else:
            raise ValueError(f'The Schwarzschild radius has to be positive.')

    def __repr__(self):
        _r_S: float = self.Parameter('r_S')
        return f'{self.__class__.__name__}({self._dim}, ' + \
            f'r_S={_r_S}, metric={self._metricname})'

    def Causality(self) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:
        if self.Dim == 1:
            return super().Causality()
        _r_S: float = self.Parameter('r_S')
        _isSchwarzschildMetric: bool = self._metricname == 'Schwarzschild'

        if self.Dim == 2:
            def isCausal_BH2D(x: np.ndarray,
                              y: np.ndarray) -> Tuple[bool, bool]:
                if x[1] * y[1] < 0.0:
                    return (False, False)
                t_delta: float = y[0] - x[0]
                r_x: float = abs(x[1])
                r_y: float = abs(y[1])
                isSwapped: bool = False
                if _isSchwarzschildMetric and ((r_x < _r_S) or (r_y < _r_S)):
                    # Schwarzschild metric and at least one is inside
                    isSwapped = r_x < r_y  # order s.t. r_y <= r_x
                else:  # EddFin metric, or both points are outside
                    isSwapped = t_delta < 0  # order s.t. t_y >= t_x
                if isSwapped:  # swap
                    x, y = y, x
                    r_x, r_y = r_y, r_x
                isConnected: bool = False
                t_out: float
                t_in: float
                if _isSchwarzschildMetric:
                    t_out = r_y - r_x + _r_S * \
                        math.log(abs((r_y - _r_S) / (r_x - _r_S)))
                    t_in = -t_out
                else:
                    t_out = r_y - r_x + 2 * _r_S * \
                        math.log(abs((r_y - _r_S) / (r_x - _r_S)))
                    t_in = r_x - r_y
                if r_y <= r_x <= _r_S:  # x is inside, y is further inside
                    isConnected = t_out >= t_delta >= t_in
                elif _r_S <= r_x >= r_y:  # x is outside, y is within radius x
                    isConnected = t_delta >= t_in
                elif _r_S <= r_x <= r_y:  # x is outside, y is further outside
                    isConnected = t_delta >= t_out
                if isSwapped:
                    return (False, isConnected)
                else:
                    return (isConnected, False)
            return isCausal_BH2D

        return NotImplemented
