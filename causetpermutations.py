#!/usr/bin/env python
'''
Created on 25 Oct 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import Set, List, Tuple, Iterator, Dict
import numpy as np
from causets.causetevent import CausetEvent
from causets.causet import Causet


def AntichainPermutations(C: Causet, antichain: Set[CausetEvent],
                          orientationLayer: List[CausetEvent] = None,
                          includeLocalSymmetries: bool = False,
                          includeGlobalSymmetry: bool = False) -> \
        Iterator[List[CausetEvent]]:
    ac_list: List[CausetEvent] = list(antichain)
    D: np.ndarray = C.DistanceMatrix(ac_list, counting='intersection',
                                     recursive=True)
    for i, j in zip(*np.where(D == np.max(D))):
        if includeGlobalSymmetry or (i < j):
            ac_perm: List[CausetEvent]
            for p in C.Antipaths(ac_list[i], ac_list[j],
                                 along=ac_list, distances=D):
                ac_perm = p
                ac_perm
            yield ac_perm


def AntichainPermutations_old(C: Causet, eventSet: Set[CausetEvent],
                              pastLayer: Tuple[CausetEvent, ...] = ()) -> \
        Iterator[Tuple[CausetEvent, ...]]:
    '''
    Returns an iterator that yields all possible permutations of the 
    antichain given by `eventSet`.
    '''
    c: int = len(eventSet)
    events: np.ndarray = np.array(
        C.sortedByLabels(eventSet), dtype=CausetEvent)
    # create distance matrix:
    D: np.ndarray = np.zeros((c, c), dtype=int)
    for i in range(c):
        for j in range(i + 1, c):
            dist: int = C.SmallestIntervalCard(
                events[i], events[j], eventSet) - 1
            if dist < 0:
                dist = c
            D[i, j], D[j, i] = dist, dist
    print(D)

    # create index permutation extender:
    def __extend(subP: np.ndarray) -> Iterator[np.ndarray]:
        while subP.size < c:
            # find next closest index p:
            sel: np.ndarray = np.ones((c,), dtype=bool)
            sel[subP] = False
            p: int
            minvalue: int = c
            for i in np.where(sel)[0]:
                m: int = min(D[subP, i])
                if (m < minvalue) or (m == c):
                    p, minvalue = i, m
            # find sub-indices where to insert p:
            min_indices: np.ndarray
            if (minvalue == c) and (D[subP[0], p] == c):
                min_indices = np.array([subP.size - 1])
            elif (minvalue == c) and (D[subP[-1], p] == c):
                min_indices = np.array([0])
            else:
                min_indices = np.where(D[subP, p] == minvalue)[0]
            p_indices: np.ndarray = np.array([], dtype=int)
            for offset in [0, 1]:  # try left and then right appending
                for i in (min_indices + offset):  # insert index
                    if i in p_indices:
                        continue
                    p_has_past_in_layer: bool = False
                    for e in (pastLayer if offset == 0 else reversed(pastLayer)):
                        if e < events[p]:
                            p_has_past_in_layer = True
                        if e < events[subP[i]]:
                            if p_has_past_in_layer:
                                p_indices = np.append(p_indices, i)
                            break
                    else:
                        if (subP.size == 1) and (offset == 1):
                            p_indices = np.append(p_indices, i)
                        elif ((i - 1 < 0) or D[subP[i - 1], p] < c) and \
                                ((i >= subP.size) or D[subP[i], p] < c):
                            p_indices = np.append(p_indices, i)
            # insert p into subP at any of p_indices:
            if p_indices.size > 1:
                for i in p_indices:
                    try:
                        yield np.insert(subP, i, p)
                    except IndexError:
                        yield np.append(subP, p)
                break
            else:
                # There is only one insertion index.
                # No need for branching the output with `yield` and
                # the loop continues.
                try:
                    subP = np.insert(subP, p_indices, p)
                except IndexError:
                    subP = np.append(subP, p)
        else:
            yield subP

    # initialise start of index permutations:
    P: np.ndarray = np.array([], dtype=int)
    P_extension_steps: List[np.ndarray] = [P]
    cone_card: int = -1
    for i, e in enumerate(events):
        e_cone_card: int = e.ConeCard
        if e_cone_card > cone_card:
            P = np.array([i])
            cone_card = e_cone_card
    # extend and iterate through index permutations:
    P_extensions: List[Iterator[np.ndarray]] = [iter([P])]
    while P_extensions:
        try:
            P = next(P_extensions[-1])
        except StopIteration:
            P_extensions.pop()
            P_extension_steps.pop()
            continue
        if np.size(P) == c:
            yield tuple(events[P])
        else:
            P_extension_steps.append(P)
            P_extensions.append(__extend(P))


def layeredPermutations(C: Causet, eventSet: Set[CausetEvent] = None) -> \
        Iterator[List[List[CausetEvent]]]:
    '''
    Returns an iterator that yields all possible permutations of all 
    layers of the set `eventSet`.
    '''
    if eventSet is None:
        eventSet = C._events
    layer_sets: List[Set[CausetEvent]] = C.layered(eventSet)
    layer_count: int = len(layer_sets)
    layer_iterators: List[Iterator[List[CausetEvent]]] = \
        [iter([[CausetEvent()]])] * layer_count
    layers_perm: List[List[CausetEvent]] = [[]] * layer_count
    # initialise:
    for l, layer_set in enumerate(layer_sets):
        layer_iterators[l] = AntichainPermutations(C, layer_set,
                                                   layers_perm[0])
        layers_perm[l] = next(layer_iterators[l])
    # iterate:
    is_exhausted: bool = False
    while not is_exhausted:
        yield layers_perm
        for l in range(layer_count - 1, -1, -1):
            try:
                layers_perm[l] = next(layer_iterators[l])
            except StopIteration:
                is_exhausted = (l == 0)
            else:
                break
        if not is_exhausted:
            for k in range(l + 1, layer_count):
                layer_iterators[k] = AntichainPermutations(C, layer_sets[k],
                                                           layers_perm[0])
                layers_perm[k] = next(layer_iterators[k])


def permuted(C: Causet, eventSet: Set[CausetEvent] = None,
             maxIterations: int = 100000) -> \
        Tuple[List[CausetEvent], List[int]]:
    '''
    Searches for a permutation of integers from 1 to len(eventSet) + 1 
    such that the permutation determines a causet that is a 2D projected 
    version of this instance. This function provides can be used to 
    obtain a `flattened` causet to draw a Hasse diagram.
    If eventSet = None (default), all causet events are included.

    The optional parameter `maxIterations` sets the limit of iterations 
    in the optimisation. If this value is reaches without finding any 
    valid permutation, the function raises a ValueError.
    '''
    if eventSet is None:
        eventSet = C._events
    eventSet_len = len(eventSet)
    # Reserve buffer for result:
    L: List[CausetEvent] = []   # permutation of events
    P: List[int] = []           # integers to generate coordinates
    extension_degree: int = -1  # number of extended causal relations
    iteration: int = 0
    for layers in layeredPermutations(C, eventSet):
        iteration += 1
        if iteration > maxIterations:
            if extension_degree == -1:
                raise ValueError('Function \'permuted\' failed with ' +
                                 f'{maxIterations} permutations. ' +
                                 'Try to increase \'maxIterations\'.')
            break
        L_this = [CausetEvent()] * eventSet_len
        P_this = [0] * eventSet_len
        i_first: int
        i_last: int
        # Extend the future of each event by those events in future
        # layers that are between two other future events in the layers:
        layers_len = len(layers)
        ext_futures: Dict[CausetEvent, Set[CausetEvent]] = {}
        for l in range(layers_len - 1, -1, -1):
            layer = layers[l]
            for a in layer:
                ext_future: Set[CausetEvent] = a.Future & eventSet
                for m in range(l, layers_len):
                    future_layer: List[CausetEvent] = layers[m]
                    i_first = 0
                    i_last = -1
                    for i, b in enumerate(future_layer):
                        if a < b:
                            if i_first > i_last:
                                i_first = i
                            i_last = i
                    for i, b in enumerate(future_layer):
                        if i_first <= i <= i_last:
                            ext_future.add(b)
                            ext_future.update(ext_futures[b])
                ext_futures[a] = ext_future
        # Extend the past of each event by those events in past
        # layers that are between two other past events in the layers:
        ext_pasts: Dict[CausetEvent, Set[CausetEvent]] = {}
        try:
            for l, layer in enumerate(layers):
                right: Set[CausetEvent] = set()
                for j, b in enumerate(layer):
                    ext_past: Set[CausetEvent] = b.Past & eventSet
                    for k in range(l - 1, -1, -1):
                        past_layer: List[CausetEvent] = layers[k]
                        i_first = 0
                        i_last = -1
                        for i, a in enumerate(past_layer):
                            if a < b:
                                if i_first > i_last:
                                    i_first = i
                                i_last = i
                        for i, a in enumerate(past_layer):
                            if i_first <= i <= i_last:
                                ext_past.add(a)
                                ext_past.update(ext_pasts[a])
                        if (j == 0) and (i_first > 0):
                            # initialise right region for
                            # right-most event in `layer`:
                            for a in past_layer[:i_first]:
                                right.add(a)
                                right.update(ext_futures[a])
                    ext_pasts[b] = ext_past
                # Find u and v coordinates with the cardinalities of
                # the events in the past, right and left regions:
                for e in layer:
                    ext_cone: Set[CausetEvent] = \
                        ext_pasts[e] | ext_futures[e]
                    ext_cone.add(e)
                    right.difference_update(ext_cone)
                    left: Set[CausetEvent] = eventSet.copy()
                    left.difference_update(right)
                    left.difference_update(ext_cone)
                    past_card: int = len(ext_pasts[e])
                    u: int = past_card + len(left) + 1
                    v: int = past_card + len(right)
                    L_this[v] = e  # might raise IndexError if v too large
                    P_this[v] = u
                    right.update(ext_cone)
            if 0 in P_this:
                raise IndexError
        except IndexError:
            continue
        else:
            # Minimise number of extended causal relations:
            extended_causals: int = sum(len(ext_futures[e] - e.Future)
                                        for e in eventSet)
            if (extension_degree == -1) or \
                    (extended_causals < extension_degree):
                L = L_this
                P = P_this
                extension_degree = extended_causals
            if extension_degree == 0:
                break
            else:
                continue
    return (L, P)
