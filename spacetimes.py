#!/usr/bin/env python
'''
Created on 22 Jul 2020

@author: Christoph Minz
'''
from typing import Callable, Tuple
import numpy as np


def Causality(name: str, dim: int,
              **kwargs) -> Callable[[np.ndarray, np.ndarray],
                                    Tuple[bool, bool]]:
    '''
    Returns a handle to a function to determine if two points x and y are 
    causally connected for the spacetime 'name' of dimension 'dim' (and 
    with parameters specified by keyword parameters).

    The function accepts coordinates x and y for two points and returns 
    the causality tuple (x <= y, x > y).

    Supported spacetimes:
    'Minkowski' or 'flat' for dim >= 1 (no parameters)

    Supported but not implemented spacetimes:
    'deSitter' for dim >= 2 
    (de Sitter radius parameter 'r_dS': float)
    'Schwarzschild' for dim >= 2 
    (Schwarzschild radius parameter 'r_S': float)
    '''
    if name in {'Minkowski', 'flat'}:
        if dim == 1:
            def isCausal_Minkowski1D(x: np.ndarray,
                                     y: np.ndarray) -> Tuple[bool, bool]:
                return (x[0] <= y[0], x[0] > y[0])
            return isCausal_Minkowski1D
        if dim == 2:
            def isCausal_Minkowski2D(x: np.ndarray,
                                     y: np.ndarray) -> Tuple[bool, bool]:
                isConnected: bool = abs(y[0] - x[0]) >= abs(y[1] - x[1])
                return ((x[0] <= y[0]) and isConnected,
                        (x[0] > y[0]) and isConnected)
            return isCausal_Minkowski2D
        elif dim > 2:
            def isCausal_Minkowski(x: np.ndarray,
                                   y: np.ndarray) -> Tuple[bool, bool]:
                isConnected: bool = \
                    (y[0] - x[0])**2 >= sum((y[1:] - x[1:])**2)
                return ((x[0] <= y[0]) and isConnected,
                        (x[0] > y[0]) and isConnected)
            return isCausal_Minkowski

    elif name == 'deSitter':
        _rdS = kwargs['r_dS']  # de Sitter radius
        if dim == 2:
            def isCausal_deSitter2D(x: np.ndarray,
                                    y: np.ndarray) -> Tuple[bool, bool]:
                isConnected: bool = abs(y[0] - x[0]) >= abs(y[1] - x[1])
                return ((x[0] <= y[0]) and isConnected,
                        (x[0] > y[0]) and isConnected)
            return isCausal_deSitter2D
        elif dim > 2:
            def isCausal_deSitter(x: np.ndarray,
                                  y: np.ndarray) -> Tuple[bool, bool]:
                isConnected: bool = \
                    (y[0] - x[0])**2 >= sum((y[1:] - x[1:])**2)
                return ((x[0] <= y[0]) and isConnected,
                        (x[0] > y[0]) and isConnected)
            return isCausal_deSitter

    elif name == 'Schwarzschild':
        _rS = kwargs['r_S']  # Schwarzschild radius
        if dim == 2:
            def isCausal_Schwarzschild2D(x: np.ndarray,
                                         y: np.ndarray) -> Tuple[bool, bool]:
                isConnected: bool = abs(y[0] - x[0]) >= abs(y[1] - x[1])
                return ((x[0] <= y[0]) and isConnected,
                        (x[0] > y[0]) and isConnected)
            return isCausal_Schwarzschild2D
        elif dim > 2:
            def isCausal_Schwarzschild(x: np.ndarray,
                                       y: np.ndarray) -> Tuple[bool, bool]:
                isConnected: bool = \
                    (y[0] - x[0])**2 >= sum((y[1:] - x[1:])**2)
                return ((x[0] <= y[0]) and isConnected,
                        (x[0] > y[0]) and isConnected)
            return isCausal_Schwarzschild

    raise ValueError(f'The {dim}-dimensional {name} spacetime ' +
                     'causality is not implemented.')
