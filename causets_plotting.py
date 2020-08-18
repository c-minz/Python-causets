#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
'''
from __future__ import annotations
from typing import List, Dict, Any, Callable, Union
from events import CausetEvent
import numpy as np
from matplotlib import patches, pyplot as plt, axes as plta

# This is the official colour scheme of the University of York.
# You may replace it by any other colour scheme.
color_scheme: Dict[str, str] = {'black':    '#262F38',
                                'darkblue': '#00546E',
                                'white':    '#C5C9CA',
                                'grey':     '#C7C1B0',
                                'blue':     '#1389D9',
                                'cyan':     '#00A29A',
                                'lime':     '#89CF00',
                                'green':    '#6BB201',
                                'yellow':   '#FFA900',
                                'orange':   '#FF5200',
                                'red':      '#DD001F',
                                'pink':     '#D50083',
                                'purple':   '#7C4DC4'}


def plot_parameters(**kwargs) -> Dict[str, Any]:
    '''
    Pre-sets the default plot parameters and overwrites them with any of 
    the user-defined values. For a full set of possible plot properties, 
    consult the matplotlib documentation for the respective plot objects.

    >> General plot parameters:
    'dims': List[int]
    List of 2 or 3 integers that set the coordinate dimensions along the 
    x- and y- (and z-)axis of the plot. The plot will be either in 2D or 3D.
    Default: [1, 0] (first space dimension along the x-axis, time dimension 
    along the y-axis)

    'axislim': Dict[str, Tuple(float, float)]
    Axis limits of the plot ranges with the keys 'xlim', 'ylim' (and 'zlim').
    Each entry is a tuple of the minimal and maximal value.
    Default: -unset- (axis limits are not handled automatically)

    'aspect': List[str]
    Aspect settings for a 2D plot. 
    (Note: Aspect settings for 3D are not yet supported by matplotlib. For 
    3D, use equal axis lengths - see 'axislim' - and set the window to equal 
    width and height.)
    Default: ['equal', 'box']


    >> Plot parameters for time slicing:
    'time': float, List[float]
    Either a single or two time parameters. The first time parameter is 
    slices the past lightcones, the last time parameter slices the future 
    lightcones.
    Default: 0.0 (all cones are sliced at time 0.0)

    'timedepth': float
    This parameter switches to the dynamic plot mode such that objects are 
    only visible within this time depth. For positive values, objects to the 
    future of the first 'time' value are visible; for negative values, objects 
    to the past of the first 'time' value are visible.
    Default: -unset- (no dynamic plot mode)

    'timefade': str
    Specifies the time fading type in dynamic plot mode. Implemented values 
    are 'linear' (scales linearly to 0.0 at 'timedepth'), 'exponential' 
    (scales exponentially with half-time 'timedepth'), 'intensity' (shows a 
    spacetime dimension depending intensity scaling of a constant power).
    Default: 'linear'


    >> Plot parameters for links, events, labels:
    'links': bool, Dict[str, Any]
    Switch on the plotting of links with True (Default). A non-empty 
    dictionary will also show links. The parameters have to be supported by 
    matplotlib.lines.Line2D. The marker properties are only visible in dynamic 
    plot mode (for which 'markevery' is set to 2 if links are partially 
    plotted).
    Default: 
    {'linewidth': 2.0,
     'linestyle': '-',
     'markevery': 3,
     'color': color_scheme['blue'],
     'marker': 'o',
     'markersize': 5.0,
     'markeredgecolor': color_scheme['blue'],
     'markerfacecolor': color_scheme['cyan']}

    'events': bool, Dict[str, Any]
    Switch on the plotting of events with True (Default). A non-empty 
    dictionary will also show events. The parameters have to be supported by 
    matplotlib.lines.Line2D.
    Default:
    {'linewidth': 2.0,
     'linestyle': '',
     'marker': 'o',
     'markersize': 7.0,
     'markeredgecolor': color_scheme['darkblue'],
     'markerfacecolor': color_scheme['black']}

    'labels': bool, Dict[str, Any]
    Switch on the plotting of event labels with True (Default). A non-empty 
    dictionary will also show labels. The parameters have to be supported by 
    matplotlib.text.
    Default:
    {'verticalalignment': 'top',
     'horizontalalignment': 'right'}


    >> Plot parameters for lightcones:
    'pastcones': bool, Dict[str, Any]
    'futurecones': bool, Dict[str, Any]
    Switch on the plotting of past or future lightcones with True (Default: 
    False - lightcones are omitted). A non-empty dictionary with keyword 
    parameters will also switch on the lightcones. The parameters have to be 
    supported by matplotlib.patches.Patch (2D plots) or 
    by mpl_toolkits.mplot3d.art3d.Poly3DCollection (3D plots).
    Default 2D:
    {'edgecolor': color_scheme['orange'], 
    'facecolor': color_scheme['yellow'], 'alpha': 0.1}
    Default 3D:
    {'edgecolor': None, 'color': color_scheme['yellow'], 'alpha': 0.1}

    'conetimefade': str
    Specifies the time fading type for the 'alpha' of the lightcones, which 
    is independent of the dynamic plot mode. The 'alpha' value is used as 
    maximum.
    Additionally to the options of 'timefade', the empty string '' switches 
    off the fading of lightcones.
    Default: 'intensity'

    'conetimedepth': float
    Similar to the 'timedepth' parameter, this parameter determines the time 
    depth, only now for plotting lightcones. Again, this parameter is 
    independent of the dynamic plot mode.
    Default: 0.0
    '''
    p: Dict[str, Any] = {}
    # axis parameters:
    p['dims'] = kwargs.pop('dims', [1, 0])
    p['3d'] = len(p['dims']) > 2
    try:
        p['axislim'] = kwargs.pop('axislim')
    except KeyError:
        pass
    p['aspect'] = kwargs.pop('aspect', ['equal', 'box'])
    # time slicing parameters:
    try:
        p['timeaxis'] = p['dims'].index(0)
    except ValueError:
        p['timeaxis'] = -1
    p['conetimefade'] = kwargs.pop('conetimefade', 'intensity')
    p['conetimedepth'] = kwargs.pop('conetimedepth', 0.0)
    try:
        p['timedepth'] = kwargs.pop('timedepth')
        p['timefade'] = kwargs.pop('timefade', 'linear')
    except KeyError:
        pass
    # pastcones parameters:
    if p['3d']:
        p_pcones = {'edgecolor': None,
                    'color': color_scheme['yellow'],
                    'alpha': 0.1}
    else:
        p_pcones = {'edgecolor': color_scheme['orange'],
                    'facecolor': color_scheme['yellow'],
                    'alpha': 0.1}
    p_args: Any = kwargs.pop('pastcones', False)
    if isinstance(p_args, bool):
        if p_args:
            p['pastcones'] = p_pcones
    else:
        p_pcones.update(p_args)
        p['pastcones'] = p_pcones
    # futurecones parameters:
    if p['3d']:
        p_fcones = {'edgecolor': None,
                    'color': color_scheme['yellow'],
                    'alpha': 0.1}
    else:
        p_fcones = {'edgecolor': color_scheme['orange'],
                    'facecolor': color_scheme['yellow'],
                    'alpha': 0.1}
    p_args = kwargs.pop('futurecones', False)
    if isinstance(p_args, bool):
        if p_args:
            p['futurecones'] = p_fcones
    else:
        p_fcones.update(p_args)
        p['futurecones'] = p_fcones
    # links parameters:
    p_links = {'linewidth': 2.0,
               'linestyle': '-',
               'markevery': 3,
               'color': color_scheme['blue'],
               'marker': 'o',
               'markersize': 5.0,
               'markeredgecolor': color_scheme['blue'],
               'markerfacecolor': color_scheme['cyan']}
    p_args = kwargs.pop('links', True)
    if isinstance(p_args, bool):
        if p_args:
            p['links'] = p_links
    else:
        p_links.update(p_args)
        p['links'] = p_links
    # events parameters:
    p_events = {'linewidth': 2.0,
                'linestyle': '',
                'marker': 'o',
                'markersize': 7.0,
                'markeredgecolor': color_scheme['darkblue'],
                'markerfacecolor': color_scheme['black']}
    p_args = kwargs.pop('events', True)
    if isinstance(p_args, bool):
        if p_args:
            p['events'] = p_events
    else:
        p_events.update(p_args)
        p['events'] = p_events
    # labels parameters:
    p_labels = {'verticalalignment': 'top',
                'horizontalalignment': 'right'}
    p_args = kwargs.pop('labels', True)
    if isinstance(p_args, bool):
        if p_args:
            p['labels'] = p_labels
    else:
        p_labels.update(p_args)
        p['labels'] = p_labels
    return p


def dynamic_parameter(function: str, dim: int, timedepth: float,
                      alpha_max: float) -> Callable[[Any], Any]:
    '''
    Returns a function handle to compute the 'alpha' parameter for 
    lightcones, and also in dynamic plot mode for links and events.
    '''
    _timefade: float
    if timedepth == 0.0:
        _timefade = -1.0e10
    else:
        _timefade = -1.0 / timedepth
    _timefade_sgn: float = np.sign(timedepth)
    _alpha_max: float = alpha_max
    _dimpower: int = dim - 1
    if function == 'linear':
        def linear(value: Any) -> Any:
            return np.heaviside(_timefade_sgn * value, 1.0) * \
                _alpha_max * (_timefade * value + 1.0)
        return linear
    elif function == 'exponential':
        def exponential(value: Any) -> Any:
            return np.heaviside(_timefade_sgn * value, 1.0) * \
                _alpha_max * np.exp(_timefade * value)
        return exponential
    elif function == 'intensity':
        def intensity(value: Any) -> Any:
            return np.heaviside(_timefade_sgn * value, 1.0) * \
                _alpha_max / np.power(np.maximum(value, 0.0) + 1.0, _dimpower)
        return intensity
    else:
        return NotImplemented


def cone_plotter(ax: plta.Axes, is3D: bool, dim: int, timeaxis: int,
                 plotting_params: Dict[str, Any],
                 timesign: float, timeslice: float,
                 timefade: str = '', timedepth: float = 0.0) -> \
    Callable[[float, Union[np.ndarray, List[float]]],
             Union[patches.Patch, np.ndarray]]:
    '''
    Returns a function handle to plot cones in 2D or 3D mode.
    '''
    _ax: plta.Axes = ax
    _timeaxis: int = timeaxis
    _plotting_cones: Dict[str, Any] = plotting_params
    _timesign: float = timesign
    _timeslice: float = timeslice
    _isDynamic = str != ''
    if _isDynamic:
        try:
            dyn_cones = dynamic_parameter(timefade, dim,
                                          np.abs(timedepth),
                                          _plotting_cones['alpha'])
        except KeyError:
            dyn_cones = dynamic_parameter(timefade, dim,
                                          np.abs(timedepth), 1)
    if is3D:
        _samplesize = 32
        if _timeaxis == 0:
            _xaxis, _yaxis = 1, 2
        elif _timeaxis == 1:
            _xaxis, _yaxis = 2, 0
        elif _timeaxis == 2:
            _xaxis, _yaxis = 0, 1

        def _cone3(t: float, coords: Union[np.ndarray, List[float]]) -> \
                Union[patches.Patch, np.ndarray]:
            '''
            Creates a matplotlib surface plot for a 3D lightcone at time 
            coordinate t and originating from the coordinate triple coords. 
            All keyword arguments are passed to the Poly3DCollection 
            object.
            '''
            p: np.ndarray = None
            r: float = _timesign * (_timeslice - t)
            if r <= 0.0:  # radius non-positive
                return None
            elif _timeaxis < 0:  # no time axis, cone is a ball
                p = np.empty((3, _samplesize, _samplesize))
                phi = np.linspace(0, 2 * np.pi, _samplesize)
                theta = np.linspace(0, np.pi, _samplesize)
                p[0, :, :] = coords[0] + \
                    r * np.outer(np.cos(phi), np.sin(theta))
                p[1, :, :] = coords[1] + \
                    r * np.outer(np.sin(phi), np.sin(theta))
                p[2, :, :] = coords[2] + \
                    r * np.outer(np.ones(_samplesize), np.cos(theta))
            else:  # no time axis, cone is a proper cone
                p = np.empty((3, 2, 100))
                phi = np.linspace(0, 2 * np.pi, 100)
                tscale = np.linspace(0, 1, 2)
                p[_xaxis, :, :] = coords[_xaxis] + \
                    r * np.outer(tscale, np.cos(phi))
                p[_yaxis, :, :] = coords[_yaxis] + \
                    r * np.outer(tscale, np.sin(phi))
                p[_timeaxis, :, :] = coords[_timeaxis] + \
                    _timesign * r * np.array([np.zeros(100), np.ones(100)])
            if p is not None:
                if _isDynamic:
                    conealpha = dyn_cones(r)
                    if conealpha <= 0.0:
                        return None
                    _plotting_cones.update({'alpha': conealpha})
                _ax.plot_surface(p[0, :, :], p[1, :, :],
                                 p[2, :, :], **_plotting_cones)
            return p

        return _cone3

    else:
        def _cone2(t: float, coords: Union[np.ndarray, List[float]]) -> \
                Union[patches.Patch, np.ndarray]:
            '''
            Creates a matplotlib patch for a 2D lightcone at time coordinate 
            t and originating from the coordinate pair coords. The patch is 
            added to the axes. All keyword arguments are passed to the Patch 
            object.
            '''
            p: patches.Patch = None
            r: float = _timesign * (_timeslice - t)
            if r <= 0.0:  # radius non-positive
                return None
            if _isDynamic:
                conealpha = dyn_cones(r)
                if conealpha <= 0.0:
                    return None
                _plotting_cones.update({'alpha': conealpha})
            if _timeaxis < 0:  # without time axis, cone is a circle
                p = patches.Circle(coords, radius=r, **_plotting_cones)
            else:  # with time axis, cone is a triangle
                if _timeaxis == 0:
                    # time is along x-axis:
                    p_arr = np.array([coords,
                                      [_timeslice, coords[1] - r],
                                      [_timeslice, coords[1] + r]])
                else:
                    # time is along y-axis:
                    p_arr = np.array([coords,
                                      [coords[0] - r, _timeslice],
                                      [coords[0] + r, _timeslice]])
                p = patches.Polygon(p_arr, **_plotting_cones)
            if p is not None:
                _ax.add_patch(p)
            return p

        return _cone2


def plotter(eventList: List[CausetEvent], coords: np.ndarray,
            plotAxes: plta.Axes=None, **kwargs) -> \
        Callable[[float], Dict[str, Any]]:
    '''
    Returns a plotter function handle that requires the 'time' parameters, 
    which has to be an list or np.ndarray of (one or two) float values.
    Call the returned function to plots the events in eventList and their 
    links (and further objects) to the Axes object ax. If ax is set to 
    None (default) it plots in the current axes. The function returns a 
    dictionary of plot object pointers. The keyword arguments are 
    explained in the doc of plot_parameters.
    '''
    plotting: Dict[str, Any] = plot_parameters(**kwargs)
    dim = coords.shape[1]
    is3d = plotting['3d']
    eventCount = len(eventList)
    if 'links' in plotting:
        linkCount = CausetEvent.LinkCountOf(set(eventList))
    _xy_z: List[int] = plotting['dims']
    _x: int = _xy_z[0]
    _y: int = _xy_z[1]
    ax: plta.Axes
    if is3d:
        _z: int = _xy_z[2]
        if plotAxes is None:
            ax = plt.gca(projection='3d')
    elif plotAxes is None:
        ax = plt.gca(projection=None)
    else:
        ax = plotAxes
    _h = {}
    isPlottingPastcones: bool = 'pastcones' in plotting
    isPlottingFuturecones: bool = 'futurecones' in plotting

    def _timeslice(time: np.ndarray) -> Dict[str, Any]:
        '''
        Core plot function that returns a dictionary of plot object 
        pointers.
        '''
        # plot cones:
        if isPlottingPastcones or isPlottingFuturecones:
            if isPlottingPastcones:
                _hpcn = [None] * eventCount
                plotpcone = cone_plotter(ax, is3d, dim, plotting['timeaxis'],
                                         plotting['pastcones'], -1, time[0],
                                         plotting['conetimefade'],
                                         plotting['conetimedepth'])
            if isPlottingFuturecones:
                _hfcn = [None] * eventCount
                plotfcone = cone_plotter(ax, is3d, dim, plotting['timeaxis'],
                                         plotting['futurecones'], 1, time[-1],
                                         plotting['conetimefade'],
                                         plotting['conetimedepth'])
            for i in range(eventCount):
                c: np.ndarray = coords[i, :]
                if isPlottingPastcones:
                    _hpcn[i] = plotpcone(c[0], c[_xy_z])
                if isPlottingFuturecones:
                    _hfcn[i] = plotfcone(c[0], c[_xy_z])
        # plot links, events, labels:
        l: int = -1
        if 'timedepth' in plotting:  # dynamic plots only
            t_depth = plotting['timedepth']
            t_dist = coords[:, 0] - time[0]
            if 'links' in plotting:
                plotting_links: Dict[str, Any] = plotting['links']
                _hlnk = [None] * linkCount
            else:
                plotting_links = {}
            if 'events' in plotting:
                plotting_events: Dict[str, Any] = plotting['events']
                _hvnt = [None] * eventCount
            else:
                plotting_events = {}
            if 'labels' in plotting:
                plotting_labels: Dict[str, Any] = plotting['labels']
                _hlbl = [None] * eventCount
            else:
                plotting_labels = {}
            try:
                dyn_links = dynamic_parameter(plotting['timefade'],
                                              dim, t_depth,
                                              plotting_links['alpha'])
            except KeyError:
                dyn_links = dynamic_parameter(plotting['timefade'],
                                              dim, t_depth, 1)
            try:
                dyn_events = dynamic_parameter(plotting['timefade'],
                                               dim, t_depth,
                                               plotting_events['alpha'])
            except KeyError:
                dyn_events = dynamic_parameter(plotting['timefade'],
                                               dim, t_depth, 1)
            eventFade = dyn_events(t_dist)
            for i, a in enumerate(eventList):
                if plotting_links:
                    for j in range(i + 1, eventCount):
                        if not a.isLinkedTo(eventList[j]):
                            continue
                        l += 1
                        if (eventFade[i] > 0) and (eventFade[j] > 0):
                            if np.abs(t_dist[i]) > np.abs(t_dist[j]):
                                i_in, i_out = j, i
                            else:
                                i_in, i_out = i, j
                            tau: float = 1.0
                            m = 3
                        elif ((eventFade[i] > 0) or (eventFade[j] > 0)) and \
                                (np.sign(t_dist[i]) != np.sign(t_dist[j])):
                            if t_depth * t_dist[i] > 0:
                                i_in, i_out = j, i
                            else:
                                i_in, i_out = i, j
                            tau = np.abs(t_dist[i_out] /
                                         (t_dist[i] - t_dist[j]))
                            m = 2
                        else:
                            continue
                        linkTarget = (1 - tau) * \
                            coords[i_out, :] + tau * coords[i_in, :]
                        linkWidth = dyn_links(t_dist[i_out])
                        if linkWidth > 0.0:
                            plotting_links.update({'alpha': linkWidth,
                                                   'markevery': m})
                            if is3d:
                                _hlnk[l] = ax.plot(
                                    [coords[i_out, _x], linkTarget[_x]],
                                    [coords[i_out, _y], linkTarget[_y]],
                                    [coords[i_out, _z], linkTarget[_z]],
                                    **plotting_links)
                            else:
                                _hlnk[l] = ax.plot(
                                    [coords[i_out, _x], linkTarget[_x]],
                                    [coords[i_out, _y], linkTarget[_y]],
                                    linewidth=linkWidth,
                                    **plotting_links)
                if eventFade[i] <= 0:
                    continue
                if plotting_events:
                    plotting_events.update({'alpha': eventFade[i]})
                    if is3d:
                        _hvnt[i] = ax.plot([coords[i, _x]], [coords[i, _y]],
                                           [coords[i, _z]],
                                           **plotting_events)
                    else:
                        _hvnt[i] = ax.plot([coords[i, _x]], [coords[i, _y]],
                                           **plotting_events)
                if plotting_labels:
                    if is3d:
                        _hlbl[i] = ax.text(coords[i, _x], coords[i, _y],
                                           coords[i, _z], f' {a.Label} ',
                                           **plotting_labels)
                    else:
                        _hlbl[i] = ax.text(coords[i, _x], coords[i, _y],
                                           f' {a.Label} ',
                                           **plotting_labels)
        else:  # static plots only
            if 'links' in plotting:
                _hlnk = [None] * linkCount
                for i, a in enumerate(eventList):
                    for j in range(i + 1, eventCount):
                        if not a.isLinkedTo(eventList[j]):
                            continue
                        l += 1
                        if is3d:
                            _hlnk[l] = ax.plot([coords[i, _x], coords[j, _x]],
                                               [coords[i, _y], coords[j, _y]],
                                               [coords[i, _z], coords[j, _z]],
                                               **plotting['links'])
                        else:
                            _hlnk[l] = ax.plot([coords[i, _x], coords[j, _x]],
                                               [coords[i, _y], coords[j, _y]],
                                               **plotting['links'])
            if 'events' in plotting:
                _hvnt = [None] * eventCount
                for i in range(eventCount):
                    if is3d:
                        _hvnt[i] = ax.plot([coords[i, _x]], [coords[i, _y]],
                                           [coords[i, _z]],
                                           **plotting['events'])
                    else:
                        _hvnt[i] = ax.plot([coords[i, _x]], [coords[i, _y]],
                                           **plotting['events'])
            if 'labels' in plotting:
                _hlbl = [None] * eventCount
                for i, e in enumerate(eventList):
                    if is3d:
                        _hlbl[i] = ax.text(coords[i, _x], coords[i, _y],
                                           coords[i, _z], f' {e.Label} ',
                                           **plotting['labels'])
                    else:
                        _hlbl[i] = ax.text(coords[i, _x], coords[i, _y],
                                           f' {e.Label} ',
                                           **plotting['labels'])
        # set axis parameters:
        try:
            ax.set(xlim=plotting['axislim']['xlim'],
                   ylim=plotting['axislim']['ylim'])
            if is3d:
                ax.set(zlim=plotting['axislim']['zlim'])
        except KeyError:
            pass
        if not is3d:
            ax.set_aspect(*plotting['aspect'])
        # return pointers:
        if isPlottingPastcones:
            _h['pastcones'] = _hpcn
        if isPlottingFuturecones:
            _h['futurecones'] = _hfcn
        if 'links' in plotting:
            _h['links'] = _hlnk
        if 'events' in plotting:
            _h['events'] = _hvnt
        if 'labels' in plotting:
            _h['labels'] = _hlbl
        return _h

    return _timeslice


def plot(eventList: List[CausetEvent], coords: np.ndarray,
         ax: plta.Axes=None, **kwargs) -> Dict[str, Any]:
    '''
    Plots the events in eventList and their links to the Axes object ax 
    (or current axes by default). It returns a dictionary of plot object 
    pointers. The keyword arguments are explained in the doc of 
    plot_parameters.
    '''
    time: np.ndarray = np.zeros(2)
    if 'time' in kwargs:
        if np.shape(kwargs['time']) == (2,):
            time = kwargs['time']
        else:
            time = np.array([kwargs['time'], kwargs['time']])
    return plotter(eventList, coords, ax, **kwargs)(time)
