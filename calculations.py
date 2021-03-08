'''
Created on 21 Aug 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import List, Any, Callable
from math import log
from fractions import Fraction
import numpy as np


@np.errstate(divide='ignore')
def HarmonicNumber(n: int, approximate: int = 100) -> float:
    '''
    Returns the floating point value of the n-th harmonic number. 
    For `n` greater or equal to `approximate`, the value is approximated by 
    the asymptotic expansion.
    '''
    if n < approximate:
        return sum(1. / k for k in range(n, 0, -1))
    else:
        h: float = log(n) + np.euler_gamma
        h += 0.5 / n
        h -= 0.25 / 3 / n**2
        h += 0.025 / 3 / n**4
        return h


@np.errstate(divide='ignore')
def HarmonicNumbers(n: int, approximate: int = 100) -> np.ndarray:
    '''
    Returns a vector of floating point values for all harmonic numbers from 0 
    to n.  
    For `n` greater than `approximate`, the value is approximated by the 
    asymptotic expansion.
    '''
    H: np.ndarray = np.empty((n + 1,))
    total: float = 0.
    for k in range(min(n + 1, approximate)):
        H[k] = total
        total += 1. / (k + 1)
    if n >= approximate:
        n_range: np.ndarray = np.arange(approximate, n + 1)
        H[n_range] = np.log(n_range) + np.euler_gamma
        H[n_range] += 0.5 / n_range
        H[n_range] -= 0.25 / 3 / np.square(n_range)
        H[n_range] += 0.025 / 3 / np.square(np.square(n_range))
    return H


def HarmonicNumberFraction(n: int) -> Fraction:
    '''
    Returns the exact rational number of the n-th harmonic number. 
    '''
    if n == 0:
        return Fraction(0, 1)
    else:
        return Fraction(sum(Fraction(1, k) for k in range(1, n + 1)))


def HarmonicNumberFractions(n: int) -> List[Fraction]:
    '''
    Returns a vector of exact rational numbers for all harmonic numbers from 
    0 to n.  
    '''
    H: List[Fraction] = [Fraction()] * (n + 1)
    total: Fraction = Fraction(0, 1)
    for k in range(n + 1):
        H[k] = total
        total += Fraction(1, k + 1)
    return H


def NewtonsMethod(f: Callable[[Any], Any],
                  fprime: Callable[[Any], Any], x0: float,
                  y: float = 0.0, xmin: float = np.NINF, xmax: float = np.PINF,
                  default: float = 0.0, precission: float = 1.0e-6,
                  maxIterations: int = 50) -> float:
    '''
    Uses Newton's method to find the x value at which the function `f` (with 
    its derivative `fprime`) reaches the value y with a given `precission` and 
    maximal number of iterations `maxIterations`. At each iteration, the 
    result will be restricted to the interval 
    [`xmin` + `precission`, `xmax` - `precission`]
    If the method does not converge, `default` is returned.
    '''
    x_n: float = x0
    for _ in range(maxIterations):
        yprime_n: float = fprime(x_n)
        if yprime_n == 0.0:
            break
        x_next: float = x_n - (f(x_n) - y) / yprime_n
        if x_next < xmin:
            x_next = xmin + precission
        elif x_next > xmax:
            x_next = xmax - precission
        if abs(x_n - x_next) < precission:
            return x_next
        x_n = x_next
    return default
