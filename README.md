# Introduction
This project contains Python modules to handle causal sets.

The original code was developed in MATLAB R2019a and R2020a. In order to make it accessible without a licence, I started to convert the code to Python 3.8.3 with mypy typing support.

The Python classes and functions are documented so that the help commands within Python give further information. 

# Causal Sets (Causets)
Causal set theory is an approach to quantum gravity that replaces the spacetime continuum by a locally finite partially ordered set. 

This project implements methods to create causal sets generically, create them by defining coordinates, or via the Poisson process of sprinkling. 

An instance of 'Causet' (causets.py) is a set of 'CausetEvent' (events.py) that has all the necessary functionality to deal with causets. After creation, two instances of 'CausetEvent' can be used in logical expressions, e.g. `a < b` is True if and only if a is in the causal past of b. The 'Causet' class can be used to get subsets like 'Layers' or 'Paths', find past and future infinities, sort events, and so on.

The class 'EmbeddedCauset' extends 'Causet' and is able to handle the coordinates of events (their embedding), and a region of embedding. Instances of this class can be plotted. A further extension is 'SprinkledCauset' that yields the functionality of sprinkling. 

# Future Development
## Currently under construction:
1. Visualization of generic causets as (Hasse) diagrams
2. Converting the causets to a causal or link matrix
3. Generating causets from a data file (importing)
4. Saving causets to a data file (exporting)

## Further development plans:
1. Methods for spacetime reconstruction
