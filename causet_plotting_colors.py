#!/usr/bin/env python
'''
Created on 19 Aug 2020

@author: Christoph Minz
'''
from __future__ import annotations

# This is the list of supported color schemes.
# When adding a new entry, please specify a value for each color key that is
# used by `causet_plotting`.
# It is recommended to define at least the color-keys as listed in the scheme
# 'matplotlib' with the color 'core' as main brand core color and 'gray'
# identical to 'grey'.
ColorSchemes: Dict[str, Dict[str, str]] = {
    'matplotlib':            {'core':        'tab:blue',
                              'black':       'black',
                              'gray':        'tab:gray',
                              'grey':        'tab:gray',
                              'white':       'snow',
                              'purple':      'tab:purple',
                              'blue':        'tab:blue',
                              'cyan':        'tab:cyan',
                              'green':       'tab:green',
                              'lime':        'limegreen',
                              'yellow':      'gold',
                              'orange':      'tab:orange',
                              'red':         'tab:red',
                              'pink':        'tab:pink'},
    # ---------------------------------------------------------------
    # University of York, UK. Brand style added on 19/08/2020
    # https://www.york.ac.uk/staff/external-relations/brand/colours/
    'UniYork':               {'core':        '#00627D',
                              'black':       '#25303B',
                              'gray':        '#E7E2D3',
                              'grey':        '#E7E2D3',
                              'white':       '#E3E5E5',
                              'purple':      '#9067A9',
                              'darkblue':    '#00627D',
                              'blue':        '#0095D6',
                              'cyan':        '#00ABAA',
                              'green':       '#65B32E',
                              'lime':        '#CDD500',
                              'yellow':      '#FBB800',
                              'orange':      '#F18625',
                              'red':         '#E62A32',
                              'pink':        '#E2388D'},
    # ---------------------------------------------------------------
    # Imperial College London, UK. Brand style added on 19/08/2020
    # http://www.imperial.ac.uk/brand-style-guide/visual-identity/brand-colours/
    'ImperialLondon':        {'core':        '#002147',
                              'black':       '#002147',
                              'gray':        '#EBEEEE',
                              'grey':        '#EBEEEE',
                              'coolgrey':    '#9D9D9D',
                              'white':       '#D4EFFC',
                              'violet':      '#960078',
                              'iris':        '#751E66',
                              'purple':      '#653098',
                              'plum':        '#321E6D',
                              'navy':        '#002147',
                              'darkblue':    '#003E74',
                              'blue':        '#006EAF',
                              'cyan':        '#009CBC',
                              'green':       '#02893B',
                              'kermitgreen': '#66A40A',
                              'lime':        '#BBCEOO',
                              'yellow':      '#FFDD00',
                              'tangerine':   '#EC7300',
                              'orange':      '#D24000',
                              'cherry':      '#E40043',
                              'red':         '#DD2501',
                              'brick':       '#A51900',
                              'pink':        '#C81E78',
                              'raspberry':   '#9F004E'}
}
