# Introduction
This project contains Python modules for numerical investigations in causal set theory. Causal set theory is an approach to quantum gravity that replaces the spacetime continuum by a locally finite partially ordered set. 

This project implements methods to create causal sets generically, define them by embedding coordinates in a spacetime, or via the Poisson process (sprinkling) on any implemented spacetime (`spacetimes.py` currently supports flat spacetime, de Sitter, Anti-de Sitter, and some first developments of black hole spacetimes).

An instance of `Causet` (`causets.py`) is a set of `CausetEvent` (`events.py`) that has additional functionality. Each instance of `CausetEvent` can be used in logical expressions, e.g. `a < b` is True if and only if `a` is in the causal past of `b`. The `Causet` class can be used to get subsets like `Layers`, `Ranks`, futures and future and past of event (sets), future and past infinities, causal paths, and many more. The events of a causet can be sorted by different properties.

The class `EmbeddedCauset` extends `Causet` and is able to handle the coordinates of events (their embedding) in a given region of a spacetime. Instances of this class can be plotted (including support for light-cone plotting). The subclass `SprinkledCauset` adds the functionality to sprinkle new causets or to intensify a given spacetime region by more points.  

# Past Development
The original code was developed in MATLAB R2019a and R2020a. In order to make the code usable without a MATLAB license, I started to convert it to Python 3.8.3 with mypy typing support.

The Python classes and functions are documented so that the help commands within Python give further information. 

# Future Development
Further development ideas include: 
1. Visualization of generic causets as (Hasse) diagrams
2. Methods for spacetime reconstruction
