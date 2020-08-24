'''
Created on 21 Aug 2020

@author: Christoph Minz
'''
from __future__ import annotations
from typing import List
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
        return sum(Fraction(1, k) for k in range(1, n + 1))


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
