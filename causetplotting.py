#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List, Dict, Any, Callable, Union, Optional
from causets.causetevent import CausetEvent  # @UnresolvedImport
from causets.embeddedcauset import EmbeddedCauset  # @UnresolvedImport
import numpy as np
import causets.colorschemes as colors  # @UnresolvedImport
from causets.spacetimes import Spacetime, FlatSpacetime  # @UnresolvedImport
from causets.shapes import CoordinateShape  # @UnresolvedImport @UnusedImport
from matplotlib import patches  # @UnusedImport
from matplotlib.pyplot import figure, gca, show  # @UnusedImport
from matplotlib.axes import Axes  # @UnusedImport
from matplotlib.patches import Patch  # @UnusedImport

default_colors: Dict[str, str] = {'links':       'cs:blue',
                                  'linksedge':   'cs:blue',
                                  'linksface':   'cs:cyan',
                                  'eventsedge':  'cs:black',
                                  'eventsface':  'cs:core',
                                  'conesedge':   'cs:yellow',
                                  'conesface':   'cs:yellow'}


def setDefaultColors(schemeName: str = 'matplotlib', **kwargs) -> None:
    '''
    Sets the scheme of the default colors for all plots to `schemeName`. 
    As optional keyword arguments the following colors can be set:
    links, linksedge, linksface, eventsedge, eventsface, conesedge, conesface
    (defaults see `default_colors`)
    '''
    global default_colors
    colors.setGlobalColorScheme(schemeName)
    default_colors.update(kwargs)


def plot_parameters(**kwargs) -> Dict[str, Any]:
    '''
    Pre-sets the default plot parameters and overwrites them with any of the 
    user-defined values. For a full set of possible plot properties, consult 
    the matplotlib documentation for the respective plot objects. The 
    `colorschemes` module adds support for local color schemes of research 
    institutes. To use colors of a scheme, precede the color by 'cs:', for 
    example 'cs:blue' for the blue defined in the respective color scheme.

    General plot parameters
    -----------------------
    dims: List[int]
        List of 2 or 3 integers that set the coordinate dimensions along the x- 
        and y- (and z-)axis of the plot. The plot will be either in 2D or 3D. 
        Default: [1, 0] (first space dimension along the x-axis, time dimension 
        along the y-axis)
    axislim: Dict[str, Tuple(float, float)]
        Axis limits of the plot ranges with the keys 'xlim', 'ylim' (and 
        'zlim'). Each entry is a tuple of the minimal and maximal value. Use 
        'shape' to automatically set the limits to the bounds of the shape 
        specified by the keyword argument 'shape', which is automatically set 
        to the embedding shape when plotting an `EmbeddedCauset` object.
        Default: -unset- (axis limits are not set by the plotting function)
    aspect: List[str]
        Aspect settings for a 2D plot. 
        (Note: Aspect settings for 3D are not yet supported by matplotlib. For 
        3D, use equal axis lengths - see 'axislim' - and set the window to 
        equal width and height.)
        Default: ['equal', 'box']

    Plot parameters for time slicing
    --------------------------------
    time: float, List[float]
        Either a single or two time parameters. The first time parameter is 
        slices the past cones, the last time parameter slices the future cones.
        Default: 0.0 (all cones are sliced at time 0.0)
    timedepth: float
        This parameter switches to the dynamic plot mode such that objects are 
        only visible within this time depth. For positive values, objects to 
        the future of the first 'time' value are visible; for negative values, 
        objects to the past of the first 'time' value are visible.
        Default: -unset- (no dynamic plot mode)
    timefade: str
        Specifies the time fading type in dynamic plot mode. Implemented values 
        are 'linear' (scales linearly to 0.0 at 'timedepth'), 'exponential' 
        (scales exponentially with half-time 'timedepth'), 'intensity' (shows a 
        spacetime dimension depending intensity scaling of a constant power).
        Default: 'linear'

    Plot parameters for links, events, labels
    -----------------------------------------
    links: bool, Dict[str, Any]
        Switch on the plotting of links with True (Default). A non-empty 
        dictionary will also show links. The parameters have to be supported by 
        matplotlib.lines.Line2D. The marker properties are only visible in 
        dynamic plot mode (for which 'markevery' is set to [1] if links are 
        partially plotted).
        Default: 
        {'linewidth': 2.0,
         'linestyle': '-',
         'markevery': [],
         'color': default_colors['links'],
         'marker': 'o',
         'markersize': 5.0,
         'markeredgecolor': default_colors['linksedge'],
         'markerfacecolor': default_colors['linksface']}
    linkgapsize: float
        Default: 0.0
    events: bool, Dict[str, Any]
        Switch on the plotting of events with True (Default). A non-empty 
        dictionary will also show events. The parameters have to be supported 
        by matplotlib.lines.Line2D.
        Default:
        {'linewidth': 2.0,
         'linestyle': '',
         'marker': 'o',
         'markersize': 7.0,
         'markeredgecolor': default_colors['eventsedge'],
         'markerfacecolor': default_colors['eventsface']}
    labels: bool, Dict[str, Any]
        Switch on the plotting of events labels with True (Default). A 
        non-empty dictionary will also show labels. The parameters have to be 
        supported by matplotlib.text.
        Default:
        {'verticalalignment': 'top',
         'horizontalalignment': 'right'}

    Plot parameters for causal cones
    --------------------------------
    pastcones: bool, Dict[str, Any]
    futurecones: bool, Dict[str, Any]
        Switch on the plotting of past or future causal cones with True 
        (Default: False - causal cones are omitted). A non-empty dictionary 
        with keyword parameters will also switch on the causal cones. The 
        parameters have to be supported by matplotlib.patches.Patch (2D plots) 
        or by mpl_toolkits.mplot3d.art3d.Poly3DCollection (3D plots).
        Default 2D:
        {'edgecolor': default_colors['conesedge'], 
         'facecolor': 'none', 
         'alpha': 0.1}
        Default 3D:
        {'edgecolor': 'none', 
         'color': default_colors['conesface'], 
         'alpha': 0.1}
    conetimefade: str
        Specifies the time fading type for the 'alpha' of the causal cones, 
        which is independent of the dynamic plot mode. The 'alpha' value is 
        used as maximum.
        Additionally to the options of 'timefade', the empty string '' switches 
        off the fading of causal cones.
        Default: 'intensity'
    conetimedepth: float
        Similar to the 'timedepth' parameter, this parameter determines the 
        time depth, only now for plotting causal cones. Again, this parameter 
        is independent of the dynamic plot mode.
        Default: 0.0
    '''
    p: Dict[str, Any] = {}
    # ====================
    # axis parameters:
    p['dims'] = kwargs.pop('dims', [1, 0])
    d: int = len(p['dims'])
    if (d < 2) or (d > 3):
        raise ValueError(('%d-dimensional plot are not implemented. The '
                          'argument `dims` for the plotting dimensions '
                          'must have length 2 or 3.') % d)
    p['3d'] = len(p['dims']) > 2
    try:
        p['axislim'] = kwargs.pop('axislim')
    except KeyError:
        pass
    if (('axislim' not in p) or (p['axislim'] == 'shape')) \
            and ('shape' in kwargs):
        shape: CoordinateShape = kwargs['shape']
        if len(p['dims']) > 2:
            edgehalf: float = shape.MaxEdgeHalf(p['dims'])
            center: np.ndarray = shape.Center
            center0: float = center[p['dims'][0]]
            center1: float = center[p['dims'][1]]
            center2: float = center[p['dims'][2]]
            p.update({'axislim': {
                'xlim': (center0 - edgehalf, center0 + edgehalf),
                'ylim': (center1 - edgehalf, center1 + edgehalf),
                'zlim': (center2 - edgehalf, center2 + edgehalf)}})
        else:
            p.update({'axislim': {
                'xlim': shape.Limits(p['dims'][0]),
                'ylim': shape.Limits(p['dims'][1])}})
    p['aspect'] = kwargs.pop('aspect', ['equal', 'box'])
    # ====================
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
    # ====================
    # pastcones parameters:
    p_pcones: Dict[str, Any]
    if p['3d']:
        p_pcones = {'edgecolor': 'none',
                    'color': default_colors['conesface'],
                    'alpha': 0.1}
    else:
        p_pcones = {'edgecolor': default_colors['conesedge'],
                    'facecolor': 'none',
                    'alpha': 0.1}
    p_args: Any = kwargs.pop('pastcones', False)
    if isinstance(p_args, bool):
        if p_args:
            p['pastcones'] = colors.convertColorsInDict(p_pcones)
    else:
        p_pcones.update(p_args)
        p['pastcones'] = colors.convertColorsInDict(p_pcones)
    # ====================
    # futurecones parameters:
    p_fcones: Dict[str, Any]
    if p['3d']:
        p_fcones = {'edgecolor': 'none',
                    'color': default_colors['conesface'],
                    'alpha': 0.1}
    else:
        p_fcones = {'edgecolor': default_colors['conesedge'],
                    'facecolor': 'none',
                    'alpha': 0.1}
    p_args = kwargs.pop('futurecones', False)
    if isinstance(p_args, bool):
        if p_args:
            p['futurecones'] = colors.convertColorsInDict(p_fcones)
    else:
        p_fcones.update(p_args)
        p['futurecones'] = colors.convertColorsInDict(p_fcones)
    # ====================
    # links parameters:
    p_links: Dict[str, Any] = {'linewidth': 2.0,
                               'linestyle': '-',
                               'markevery': [],
                               'color': default_colors['links'],
                               'marker': 'o',
                               'markersize': 5.0,
                               'markeredgecolor': default_colors['linksedge'],
                               'markerfacecolor': default_colors['linksface']}
    p_linkgapsize: float = kwargs.pop('linkgapsize', 0.0)
    p['linkgaps'] = (p_linkgapsize > 0.0)
    if p['linkgaps']:
        p_links.update({'markeredgecolor': [0.0, 0.0, 0.0, 0.0],
                        'markerfacecolor': [0.0, 0.0, 0.0, 0.0]})
    p_args = kwargs.pop('links', True)
    if isinstance(p_args, bool):
        if p_args:
            p['links'] = colors.convertColorsInDict(p_links)
    else:
        p_links.update(p_args)
        if p['linkgaps']:
            p_links.update({'markersize': p_linkgapsize})
        p['links'] = colors.convertColorsInDict(p_links)
    # ====================
    # events parameters:
    p_events: Dict[str, Any] = {'linewidth': 2.0,
                                'linestyle': '',
                                'marker': 'o',
                                'markersize': 7.0,
                                'markeredgecolor': default_colors['eventsedge'],
                                'markerfacecolor': default_colors['eventsface']}
    p_args = kwargs.pop('events', True)
    if isinstance(p_args, bool):
        if p_args:
            p['events'] = colors.convertColorsInDict(p_events)
    else:
        p_events.update(p_args)
        p['events'] = colors.convertColorsInDict(p_events)
    # ====================
    # labels parameters:
    p_labels: Dict[str, Any] = {'verticalalignment': 'top',
                                'horizontalalignment': 'right'}
    p_args = kwargs.pop('labels', True)
    if isinstance(p_args, bool):
        if p_args:
            p['labels'] = colors.convertColorsInDict(p_labels)
    else:
        p_labels.update(p_args)
        p['labels'] = colors.convertColorsInDict(p_labels)
    return p


def dynamic_parameter(function: str, dim: int, timedepth: float,
                      alpha_max: float) -> Callable[[float], float]:
    '''
    Returns a function handle to compute the `alpha` parameter for 
    causal cones, and also in dynamic plot mode for links and events.
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
        def linear(value: float) -> float:
            return np.heaviside(_timefade_sgn * value, 1.0) * \
                _alpha_max * (_timefade * value + 1.0)
        return linear
    elif function == 'exponential':
        def exponential(value: float) -> float:
            return np.heaviside(_timefade_sgn * value, 1.0) * \
                _alpha_max * np.exp(_timefade * value)
        return exponential
    elif function == 'intensity':
        def intensity(value: float) -> float:
            return np.heaviside(_timefade_sgn * value, 1.0) * \
                _alpha_max / np.power(np.maximum(value, 0.0) + 1.0, _dimpower)
        return intensity
    else:
        return NotImplemented


def Plotter(E: Union[CausetEvent, List[CausetEvent], EmbeddedCauset],
            plotAxes: Optional[Axes] = None,
            spacetime: Optional[Spacetime] = None, **kwargs) -> \
        Callable[[Union[float, List[float], np.ndarray]], Dict[str, Any]]:
    '''
    Returns a function handle to a plotting function that accepts the single 
    input `time`, which has to be a list or np.ndarray of one or two float 
    values. Call the returned function to plot the events of `E` (and their 
    links) the Axes object `plotAxes`. If `plotAxes` is set to None (default), 
    then the plots appear in the current matplotlib axes. A call of the 
    plotting function returns a dictionary of plot object pointers.

    Plotting parameters have to be specified as keyword arguments. 
    See help of plot_parameters.

    `E` is either a instance or a list of `CausetEvent` to be plotted in that 
    order or an `EmbeddedCauset` object.  
    `spacetime` is the spacetime for which the events and causal structure is 
    plotted. This parameter is automatically set if E is an embedded causet. If 
    `None` (default), then the events are expected to have a `Position` 
    attribute so that a Hasse diagram can be plotted. 
    If a spacetime is specified (by E or explicitly), then the events are 
    expected to have their `Coordinates` attribute set, for a plot of the 
    embedding.
    '''
    # ====================
    # get defaults and user-specified parameters:
    plotting: Dict[str, Any] = plot_parameters(**kwargs)
    _h = {}  # holds Dict of artists
    # ====================
    # get set of events:
    events: List[CausetEvent]
    if isinstance(E, EmbeddedCauset):
        events = list(E)
        spacetime = E.Spacetime
        kwargs.update({'shape': E.Shape})
    elif isinstance(E, CausetEvent):
        events = [E]
    else:
        events = E
    eventCount: int = len(events)
    linkCount: int = 0
    if 'links' in plotting:
        linkCount = CausetEvent.LinkCountOf(set(events))
    # ====================
    # set general parameters:
    is3d = plotting['3d']
    dim: int = 3 if is3d else 2
    _xy_z: List[int] = plotting['dims']
    _x: int = _xy_z[0]
    _y: int = _xy_z[1]
    ax: Axes = plotAxes
    if is3d:
        _z: int = _xy_z[2]
        if plotAxes is None:
            ax = gca(projection='3d')
    elif plotAxes is None:
        ax = gca(projection=None)
    # ====================
    # set spacetime and lightcone parameters:
    isPlottingPastcones: bool = 'pastcones' in plotting
    isPlottingFuturecones: bool = 'futurecones' in plotting
    plot_spacetime: Spacetime
    coordattr: str
    if spacetime is None:
        plot_spacetime = FlatSpacetime(max(_xy_z) + 1)
        coordattr = 'Position'
    else:
        plot_spacetime = spacetime
        coordattr = 'Coordinates'
    if isPlottingPastcones:
        pcn_alpha_max: float
        if plotting['conetimefade'] != '':
            try:
                pcn_alpha_max = plotting['pastcones']['alpha']
            except KeyError:
                pcn_alpha_max = 1.0
        plotpcone: Any = plot_spacetime.ConePlotter(
            _xy_z, plotting['pastcones'], timesign=-1, axes=ax,
            dynamicAlpha=dynamic_parameter(plotting['conetimefade'], dim,
                                           abs(plotting['conetimedepth']),
                                           pcn_alpha_max))
    if isPlottingFuturecones:
        fcn_alpha_max: float
        if plotting['conetimefade'] != '':
            try:
                fcn_alpha_max = plotting['futurecones']['alpha']
            except KeyError:
                fcn_alpha_max = 1.0
        plotfcone: Any = plot_spacetime.ConePlotter(
            _xy_z, plotting['futurecones'], timesign=1, axes=ax,
            dynamicAlpha=dynamic_parameter(plotting['conetimefade'], dim,
                                           abs(plotting['conetimedepth']),
                                           fcn_alpha_max))
    # ====================
    plotting_links: Dict[str, Any] = {}
    if 'links' in plotting:
        plotting_links = plotting['links']
    if 'timedepth' in plotting:
        # ====================
        # dynamic plots only
        t_depth = plotting['timedepth']
        plotting_events: Dict[str, Any] = {}
        plotting_labels: Dict[str, Any] = {}
        if 'events' in plotting:
            plotting_events = plotting['events']
        if 'labels' in plotting:
            plotting_labels = plotting['labels']
        dyn_links = dynamic_parameter(plotting['timefade'],
                                      dim, t_depth,
                                      plotting_links['linewidth'])
        dyn_events = dynamic_parameter(plotting['timefade'],
                                       dim, t_depth,
                                       plotting_events.get('alpha', 1.0))

    # ====================
    def _timeslice(time: Union[float, List[float], np.ndarray]) -> \
            Dict[str, Any]:
        '''
        Core plot function that returns a dictionary of plot object 
        pointers.
        '''
        if isinstance(time, float):
            time = [time, time]  # list of floats required
        # ====================
        # plot cones:
        if isPlottingPastcones or isPlottingFuturecones:
            temp_cone: Any
            _hpcn: List[Any] = []
            _hfcn: List[Any] = []
            for a in events:
                c: np.ndarray = np.array(getattr(a, coordattr))
                if isPlottingPastcones:
                    temp_cone = plotpcone(c, time[0])
                    if temp_cone is not None:
                        _hpcn.append(temp_cone)
                if isPlottingFuturecones:
                    temp_cone = plotfcone(c, time[-1])
                    if temp_cone is not None:
                        _hfcn.append(temp_cone)
        # ====================
        # plot links, events, labels:
        l: int = -1
        c_a: np.ndarray
        _hlnk = [None] * linkCount  # holds link artists
        _hvnt = [None] * eventCount  # holds event artists
        _hlbl = [None] * eventCount  # holds label artists
        if 'timedepth' in plotting:
            # ====================
            # dynamic plots only
            for i, a in enumerate(events):
                c_a = np.array(getattr(a, coordattr))  # plot coordinates of a
                i_t_dist = c_a[0] - time[0]  # distance from a to time
                i_fade = dyn_events(i_t_dist)
                if plotting_links:
                    for j in range(i + 1, eventCount):
                        b: CausetEvent = events[j]
                        if not a.isLinkedTo(b):
                            continue
                        j_t_dist = np.array(getattr(b, coordattr))[0] - time[0]
                        j_fade = dyn_events(j_t_dist)
                        l += 1
                        tau: float = 1.0
                        if (i_fade > 0) and (j_fade > 0):
                            if np.abs(i_t_dist) > np.abs(j_t_dist):
                                i_in, i_out = j, i
                                i_out_t_dist = i_t_dist
                                linkAlpha = i_fade
                            else:
                                i_in, i_out = i, j
                                i_out_t_dist = j_t_dist
                                linkAlpha = j_fade
                            m = []
                        elif ((i_fade > 0) or (j_fade > 0)) and \
                                (np.sign(i_t_dist) != np.sign(j_t_dist)):
                            if t_depth * i_t_dist > 0:
                                i_in, i_out = j, i
                                i_out_t_dist = i_t_dist
                                linkAlpha = i_fade
                            else:
                                i_in, i_out = i, j
                                i_out_t_dist = j_t_dist
                                linkAlpha = j_fade
                            tau = np.abs(i_out_t_dist /
                                         (i_t_dist - j_t_dist))
                            m = [1]
                        else:
                            continue
                        c_out: np.ndarray = \
                            np.array(getattr(events[i_out], coordattr))
                        linkTarget = (1 - tau) * c_out \
                            + tau * np.array(getattr(events[i_in], coordattr))
                        linkWidth = dyn_links(i_out_t_dist)
                        if linkWidth > 0.0:
                            plotting_links.update({'alpha': linkAlpha,
                                                   'linewidth': linkWidth,
                                                   'markevery': m})
                            if is3d:
                                _hlnk[l] = ax.plot(
                                    [c_out[_x], linkTarget[_x]],
                                    [c_out[_y], linkTarget[_y]],
                                    [c_out[_z], linkTarget[_z]],
                                    **plotting_links)
                            else:
                                _hlnk[l] = ax.plot(
                                    [c_out[_x], linkTarget[_x]],
                                    [c_out[_y], linkTarget[_y]],
                                    **plotting_links)
                if i_fade <= 0:
                    continue
                if plotting_events:
                    plotting_events.update({'alpha': i_fade})
                    if is3d:
                        _hvnt[i] = ax.plot([c_a[_x]], [c_a[_y]], [c_a[_z]],
                                           **plotting_events)
                    else:
                        _hvnt[i] = ax.plot([c_a[_x]], [c_a[_y]],
                                           **plotting_events)
                if plotting_labels:
                    if is3d:
                        _hlbl[i] = ax.text(c_a[_x], c_a[_y], c_a[_z],
                                           f' {a.Label} ',
                                           **plotting_labels)
                    else:
                        _hlbl[i] = ax.text(c_a[_x], c_a[_y],
                                           f' {a.Label} ',
                                           **plotting_labels)
        else:
            # ====================
            # static plots only
            if plotting_links:
                eps: float = 0.0000001
                for i, a in enumerate(events):
                    c_a = np.array(getattr(a, coordattr))
                    for j in range(i + 1, eventCount):
                        I: np.ndarray
                        c_e: np.ndarray
                        c_b: np.ndarray = \
                            np.array(getattr(events[j], coordattr))
                        if not a.isLinkedTo(events[j]):
                            continue
                        l += 1
                        X: List[float] = [c_a[_x]]
                        Y: List[float] = [c_a[_y]]
                        area: float
                        if is3d:
                            Z: List[float] = [c_a[_z]]
                            if 'linkgaps' in plotting:
                                for e in events:
                                    if e is a or e is events[j]:
                                        continue
                                    c_e = np.array(getattr(e, coordattr))
                                    if (c_a[_x] <= c_e[_x] <= c_b[_x] or
                                            c_a[_x] >= c_e[_x] >= c_b[_x]) and \
                                       (c_a[_y] <= c_e[_y] <= c_b[_y] or
                                            c_a[_y] >= c_e[_y] >= c_b[_y]) and \
                                       (c_a[_z] <= c_e[_z] <= c_b[_z] or
                                            c_a[_z] >= c_e[_z] >= c_b[_z]):
                                        d_ae: List[float] = [c_e[_x] - c_a[_x],
                                                             c_e[_y] - c_a[_y],
                                                             c_e[_z] - c_a[_z]]
                                        d_eb: List[float] = [c_b[_x] - c_e[_x],
                                                             c_b[_y] - c_e[_y],
                                                             c_b[_z] - c_e[_z]]
                                        area = np.sqrt((d_ae[1] * d_eb[2] -
                                                        d_ae[2] * d_eb[1])**2 +
                                                       (d_ae[2] * d_eb[0] -
                                                        d_ae[0] * d_eb[2])**2 +
                                                       (d_ae[0] * d_eb[1] -
                                                        d_ae[1] * d_eb[0])**2)
                                        if area < eps:
                                            X.append(c_e[_x])
                                            Y.append(c_e[_y])
                                            Z.append(c_e[_z])
                                X.append(c_b[_x])
                                Y.append(c_b[_y])
                                Z.append(c_b[_z])
                                if len(X) > 3:
                                    I = np.lexsort((X, Y, Z))
                                    X = [X[i] for i in I]
                                    Y = [Y[i] for i in I]
                                    Z = [Z[i] for i in I]
                                plotting_links.update(
                                    {'markevery': list(range(1, len(X) - 1))})
                            _hlnk[l] = ax.plot(X, Y, Z, **plotting_links)
                        else:
                            if 'linkgaps' in plotting:
                                for e in events:
                                    if e is a or e is events[j]:
                                        continue
                                    c_e = np.array(getattr(e, coordattr))
                                    if (c_a[_x] <= c_e[_x] <= c_b[_x] or
                                            c_a[_x] >= c_e[_x] >= c_b[_x]) and \
                                       (c_a[_y] <= c_e[_y] <= c_b[_y] or
                                            c_a[_y] >= c_e[_y] >= c_b[_y]):
                                        area = np.abs((c_e[_x] - c_a[_x]) *
                                                      (c_b[_y] - c_e[_y]) -
                                                      (c_e[_y] - c_a[_y]) *
                                                      (c_b[_x] - c_e[_x]))
                                        if area < eps:
                                            X.append(c_e[_x])
                                            Y.append(c_e[_y])
                                X.append(c_b[_x])
                                Y.append(c_b[_y])
                                if len(X) > 3:
                                    I = np.lexsort((X, Y))
                                    X = [X[i] for i in I]
                                    Y = [Y[i] for i in I]
                                plotting_links.update(
                                    {'markevery': list(range(1, len(X) - 1))})
                            _hlnk[l] = ax.plot(X, Y, **plotting_links)
            if 'events' in plotting:
                for i, a in enumerate(events):
                    c_a = np.array(getattr(a, coordattr))
                    if is3d:
                        _hvnt[i] = ax.plot([c_a[_x]], [c_a[_y]],
                                           [c_a[_z]],
                                           **plotting['events'])
                    else:
                        _hvnt[i] = ax.plot([c_a[_x]], [c_a[_y]],
                                           **plotting['events'])
            if 'labels' in plotting:
                for i, a in enumerate(events):
                    c_a = np.array(getattr(a, coordattr))
                    if is3d:
                        _hlbl[i] = ax.text(c_a[_x], c_a[_y], c_a[_z],
                                           f' {a.Label} ',
                                           **plotting['labels'])
                    else:
                        _hlbl[i] = ax.text(c_a[_x], c_a[_y],
                                           f' {a.Label} ',
                                           **plotting['labels'])
        # ====================
        # set axes properties:
        try:
            ax.set(xlim=plotting['axislim']['xlim'],
                   ylim=plotting['axislim']['ylim'])
            if is3d:
                ax.set(zlim=plotting['axislim']['zlim'])
        except KeyError:
            pass
        if not is3d:
            ax.set_aspect(*plotting['aspect'])
        # ====================
        # collect pointers in dictionary:
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


def plot(E: Union[CausetEvent, List[CausetEvent], EmbeddedCauset],
         plotAxes: Optional[Axes] = None,
         spacetime: Optional[Spacetime] = None, **kwargs) -> Dict[str, Any]:
    '''
    Generates a plotting function with `Plotter` and passes the `time` keyword 
    argument to the plotting function, the dictionary of plot handles is 
    returned. If the keyword `time` (a list of one or two float) is not 
    specified, then the default [0.0, 0.0] is used.
    '''
    time: np.ndarray = np.zeros(2)
    if 'time' in kwargs:
        if np.shape(kwargs['time']) == (2,):
            time = kwargs['time']
        else:
            time = np.array([kwargs['time'], kwargs['time']])
    return Plotter(E, plotAxes, spacetime, **kwargs)(time)


def plotDiagram(E: List[CausetEvent], permutation: List[int] = [],
                plotAxes: Optional[Axes] = None, **kwargs) -> \
        Dict[str, Any]:
    '''
    Plots a Hasse diagram of `E` such that every event is placed at the 
    point specified by its `Position` attribute. If `permutation` is specified 
    as an integer list with the same length as `E`, then the `Position` 
    attribute of the i-th element are set to the coordinates (i, permutation[i])
    for all i. 
    The plotting is executed by the `Plotter` routine.
    '''
    if len(permutation) == len(E):
        C: np.ndarray = EmbeddedCauset._Permutation_Coords(permutation, 1.0)
        for i, e in enumerate(E):
            e.Position = C[i, :]
    H: Dict[str, Any] = plot(E, plotAxes, **kwargs)
    if plotAxes is None:
        plotAxes = gca()
    plotAxes.set_axis_off()
    return H
