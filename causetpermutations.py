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


def permuted_old(C: Causet, eventSet: Set[CausetEvent] = None,
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

# new development (started December 2021) below:


def findClosedFences(PStart: List[CausetEvent],
                     FStart: List[CausetEvent],
                     PAchain: Set[CausetEvent],
                     FAchain: Set[CausetEvent]) -> \
        List[Tuple[List[CausetEvent], List[CausetEvent], int]]:
    '''
    Searches for causal, closed Fences between events of the 
    past antichain set `PAchain` and the future antichain set 
    `FAchain`. All closed Fences have to begin with the events 
    `PStart` in the past and `FStart` in the future where:
     `FStart`:  0  1  2         0  1  2  ...
                |\ |\ |\     or | /| /| /
                | \| \| \       |/ |/ |/
     `PStart`:  0  1  2  ...    0  1  2
    At least one of the lists has to be non-empty and the other 
    list has to have the same number of elements or exactly one 
    more or less, so that possible starts are:
     `FStart`:  0         0    0       0  1    0  1    0  1
                  or   or | or |\   or | /  or |\ | or | /| ...
                          |    | \     |/      | \|    |/ |
     `PStart`:       0    0    0  1    0       0  1    0  1

    The function returns a list of closed Fences each including 
    the events `PStart` and `FStart` and any events from the sets 
    `PAchain` and `FAchain` at most once among all returned values.
    Examples of closed Fences are:
     orientation:  -1/1         -1             1                1
     fence future: 0  1    0  2  1    0  1  3  2    0  1  4  2  3
                   |\/| or |\/ \/| or |\/ \/ \/| or |\/ \/ \/ \/| ...
                   |/\|    |/\ /\|    |/\ /\ /\|    |/\ /\ /\ /\|
     fence past:   0  1    0  1  2    0  3  1  2    0  4  1  3  2
     return: list of (fence past, fence future, orientation)
    '''
    # Check if the given fence (start) is valid:
    PStartSet: Set[CausetEvent] = set(PStart)
    FStartSet: Set[CausetEvent] = set(FStart)
    PStartCount: int = len(PStartSet)
    FStartCount: int = len(FStartSet)
    isStartCountsEqual: bool = PStartCount == FStartCount
    isValidStart: bool = (PStartCount > 0 or FStartCount > 0) and \
        PStartCount == len(PStart) and \
        FStartCount == len(FStart) and \
        abs(PStartCount - FStartCount) <= 1 and \
        PStartSet.issubset(PAchain) and \
        FStartSet.issubset(FAchain)
    isStartShapeN: bool = False  # Does it start in a Lambda shape?
    isStartShapeV: bool = False  # Does it start in a V shape?
    isClosedFenceN: bool = False
    isClosedFenceV: bool = False
    TerminalSet: Set[CausetEvent]
    if PStartCount >= FStartCount:
        isStartShapeN = isValidStart and \
            all(FStart[i].Past & PStartSet ==
                {PStart[i], PStart[i + 1]}
                for i in range(PStartCount - 1))
        if isStartCountsEqual:
            TerminalSet = FStart[-1].Past & PStartSet
            isClosedFenceN = (PStartCount > 1) and \
                (TerminalSet == {PStart[-1], PStart[0]})
            isStartShapeN = isStartShapeN and \
                (isClosedFenceN or (TerminalSet == {PStart[-1]}))
    if FStartCount >= PStartCount:
        isStartShapeV = isValidStart and \
            all(PStart[i].Future & FStartSet ==
                {FStart[i], FStart[i + 1]}
                for i in range(FStartCount - 1))
        if isStartCountsEqual:
            TerminalSet = PStart[-1].Future & FStartSet
            isClosedFenceV = (FStartCount > 1) and \
                (TerminalSet == {FStart[-1], FStart[0]})
            isStartShapeV = isStartShapeV and \
                (isClosedFenceV or (TerminalSet == {FStart[-1]}))
    if (isStartShapeN and isClosedFenceN) or \
            (isStartShapeV and isClosedFenceV):
        return [(PStart, FStart, -1 if isStartShapeN else 1)]
    elif not isStartShapeN and not isStartShapeV:
        return []
#         raise Exception('The specified lists `PStart` and `FStart` ' +
#                         'do not describe the start of a fence.')
    # Search and complete the fence start to closed fences:
    PAchain = PAchain - PStartSet
    FAchain = FAchain - FStartSet
    Fences: List[Tuple[List[CausetEvent], List[CausetEvent], int]] = []
    PFence: List[CausetEvent] = PStart
    FFence: List[CausetEvent] = FStart
    PFenceX: List[Set[CausetEvent]] = []
    FFenceX: List[Set[CausetEvent]] = []
    PFenceXIndex: int = 0
    FFenceXIndex: int = 0
    isPExtending: bool = (isStartShapeN and isStartCountsEqual) or \
        (isStartShapeV and (PStartCount < FStartCount))
    while (PFenceXIndex >= 0) or (FFenceXIndex >= 0):
        PFenceLen: int = len(PFence)
        FFenceLen: int = len(FFence)
        canPTerminate: bool = isStartShapeV and (FFenceLen < PFenceLen)
        canFTerminate: bool = isStartShapeN and (PFenceLen < FFenceLen)
        if canPTerminate:
            pass  # TODO
        elif canFTerminate:
            FFenceTerminating: Set[CausetEvent] = FAchain & \
                PFence[-1].Future & PFence[0].Future
            for i in range(1, FFenceLen - 1):
                FFenceTerminating -= PFence[i].Future
            while FFenceTerminating:
                e: CausetEvent = FFenceTerminating.pop()
                FAchain.remove(e)
                FFenceCopy: List[CausetEvent] = FFence.copy()
                FFenceCopy.append(e)
                Fences.append((PFence, FFenceCopy,
                               -1 if isStartShapeN else 1))
            FFence.pop(FFenceLen)
        if canPTerminate or \
                (isStartShapeN and (PFenceLen == FFenceLen)):
            pass  # TODO
        elif canFTerminate or \
                (isStartShapeV and (FFenceLen == PFenceLen)):
            if FFenceXIndex > len(FFenceX):
                FFenceX.append(FAchain & PFence[-1].Future)
                for i in range(1, FFenceLen - 1):
                    FFenceX[-1] -= PFence[i].Future
            if FFenceX[-1]:
                FFence.append(FFenceX[-1].pop())

        fFenceExtending: Set[CausetEvent]
    return Fences


def __perm_layeredsubset(C: Causet, E: Set[CausetEvent], s: int) -> \
        Tuple[List[CausetEvent], List[int]]:
    '''
    Internal function to find the permutation of a connected 
    subset `E` of a causet `C`. `s` is the start index of the 
    permutation.

    This function places the events of the connected subset `E` 
    that has past and future infinities both with more than one 
    event. It starts by placing all events of the past and 
    future infinities and then places the remaining events 
    element by element.
    '''
    # TODO: implement full version
    # placeholder code:
    L: List[CausetEvent]  # events sorted by u-coordinates
    P: List[int]  # permutation for v-coordinates
    PInf: Set[CausetEvent] = C.PastInfOf(E)
    FInf: Set[CausetEvent] = C.FutureInfOf(E)
    # 1. Start with one past infinity event such that the intersection
    # of its future and the future infinity is maximal.
    PInf_sorted: List[CausetEvent] = list(PInf)
    PInf_sorted.sort(key=lambda e: len(e.Future() & FInf), reverse=True)
    a: CausetEvent = PInf_sorted[0]
    # 2. Add one future infinity event in the future of event 1 such
    # that its past intersected with the past infinity is maximal.
    B: List[CausetEvent] = list(FInf & a.Future())
    B.sort(key=lambda e: len(e.Past() & PInf), reverse=True)
    b: CausetEvent = B[0]
    c: CausetEvent = (b.Past() & PInf - {a}).pop()
    L = [a, c, b]
    P = [2, 1, 3]

    return (L, P)


def __perm_jointsubset(C: Causet, E: Set[CausetEvent],
                       s: int) -> Tuple[List[CausetEvent], List[int]]:
    '''
    Internal function to find the permutation of a connected 
    subset `E` of a causet `C`. `s` is the start index of the 
    permutation.

    This function places the events of the past and future 
    infinities of `E` if they are single events, respectively, 
    and repeats the process until both, the past and future 
    infinities are not singleton anymore.
    '''
    L_start: List[CausetEvent] = []  # past events sorted by u-coordinates
    L_end: List[CausetEvent] = []    # future events sorted by u-coordinates
    L: List[CausetEvent] = []        # events sorted by u-coordinates
    P: List[int] = []                # permutation for v-coordinates
    E_count: int = len(E)
    Inf: Set[CausetEvent] = C.PastInfOf(E)
    while len(Inf) == 1:
        e: CausetEvent = Inf.pop()
        L_start.append(e)
        E.remove(e)
        Inf = C.PastInfOf(E)
    Inf = C.FutureInfOf(E)
    while len(Inf) == 1:
        e: CausetEvent = Inf.pop()
        L_end.insert(0, e)
        E.remove(e)
        Inf = C.FutureInfOf(E)
    start_len: int = len(L_start)
    end_len: int = len(L_end)
    central_subsets = C.disjoint(E)
    if len(central_subsets) == 1:
        L, P = __perm_layeredsubset(C, E, s + start_len)
    else:
        L, P = __perm_disjointsubsets(C, central_subsets, s + start_len)
    return (L_start + L + L_end,
            list(range(s, start_len)) + P +
            list(range(E_count - end_len + 1, end_len)))


def __perm_disjointsubsets(C: Causet, subsets: List[Set[CausetEvent]],
                           s: int = 1) -> Tuple[List[CausetEvent], List[int]]:
    '''
    Internal function to find the permutation of spacelike disjoint 
    `subsets` of a causet `C`. `s` is the start index of the 
    permutation (1 by default).

    This function places the disjoint `subsets` from left to right 
    sorted decreasingly by cardinality.
    '''
    L: List[CausetEvent] = []  # events sorted by u-coordinates
    P: List[int] = []          # permutation for v-coordinates
    subsets.sort(key=len, reverse=True)
    subset: Set[CausetEvent]
    for subset in subsets:
        subset_L: List[CausetEvent]
        subset_P: List[int]
        subset_L, subset_P = __perm_jointsubset(C, subset, s)
        L = L + subset_L
        P = subset_P + P
        s = s + len(subset)
    return (L, P)


def permuted(C: Causet, eventSet: Set[CausetEvent] = None) -> \
        Tuple[List[CausetEvent], List[int]]:
    '''
    Returns the events of `eventSet` (or all events of `C` if `eventSet` 
    is not specified) in an order such that the order index plus 1  
    corresponds to the u-coordinate and the permutation (second return 
    value) corresponds to the v-coordinate when plotted as a Hasse 
    diagram.
    '''
    if eventSet is None:
        eventSet = C._events
    return __perm_disjointsubsets(C, C.disjoint(eventSet))
