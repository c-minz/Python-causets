# Introduction
This project contains the code base to handle causal sets.

The original code was developed in MATLAB R2019a and R2020a. In order to make it accessible without a licence, I started to convert the code to Python 3.8.3 with mypy typing support.

# Causal Sets (Causets)
Causal set theory is an approach to quantum gravity that repalces the spacetime continuum by a locally finite partially ordered set. 

This project implements methods to create causal sets from scratch, create them by defining coordinates, or via the Poisson process of sprinkling. 

An instance of 'Causet' (causets.py) is essentially a set of 'CausetEvent' (events.py) that contains all the necessary functionality. After creation, two instances of 'CausetEvent' can be used in logical expressions, e.g.
a < b is True if and only if a is in the causal past of b.

The class 'EmbeddedCauset' extends 'Causet' and is able to handle the coordinates of events (their embedding), and a region of embedding. Instances of this class can be plotted. A further extension is 'SprinkledCauset' that yields the functionality of sprinkling. 

# Future Plans
## Currently under construction:
- visualization of generic causets as (Hasse) diagrams
- generating causets from a data file (importing)
- storing causets to a data file (exporting)

## Further development ideas:
- methods for spacetime reconstruction
