#!/usr/bin/env python
'''
Created on 15 Jul 2020

@author: Christoph Minz
@license: BSD 3-Clause
'''
from __future__ import annotations
from typing import Set, List, Iterable, Any
import math


class CausetEvent(object):
    '''
    Handles a single event (point) and its causal relations in a causal set.
    The attribute 'Label' can be used to assign a label, but does not need to 
    be unique (default: None).

    Instances of CausetEvent are comparable:
    a == b             is True if a is the same instance as b
    a < b              is True if a precedes b
    a <= b             is True if a precedes b or a is the same as b
    a > b              is True if a succeeds b
    a >= b             is True if a succeeds b or a is the same as b
    a.isSpacelikeTo(b) is True if a is spacelike separated to b
    '''

    Label: Any
    _prec: Set['CausetEvent']
    _succ: Set['CausetEvent']
    _coordinates: List[float]
    _position: List[float]

    def __init__(self, **kwargs) -> None:
        '''
        Initialise a CausetEvent.

        Keyword parameters:
        label: str
            Label for the event (does not need to be unique in a causet)
        past: Iterable[CausetEvent]
            Set of past events (that may or may not be linked). This instance 
            will automatically be added to their future.
        future: Iterable[CausetEvent]
            Set of future events (that may or may not be linked). This instance 
            will automatically be added to their past.
        coordinates: Iterable[float]
            Coordinates if the event shall be considered as embedded in a 
            spacetime region.
        position: Iterable[float]
            Coordinate pair of the event in a Hasse diagram if the Hasse 
            diagram is manually defined.
        '''
        self.Label = kwargs.get('label')
        self._prec = set()
        for e in kwargs.get('past', []):
            self._prec.update(e.PresentOrPast)
        self._succ = set()
        for e in kwargs.get('future', []):
            self._succ.update(e.PresentOrFuture)
        self._coordinates = list(kwargs.get('coordinates', []))
        self._position = list(kwargs.get('position', []))
        # Add this instance to its causal relatives:
        for e in self._prec:
            e._addToFuture(self)
        for e in self._succ:
            e._addToPast(self)

    def hasBeenLinked(self) -> bool:
        '''
        Tests if the method link() has been called (and unlink() has not been 
        called after).
        '''
        return hasattr(self, '_lprec') and hasattr(self, '_lsucc')

    def _addToPast(self, other: 'CausetEvent') -> bool:
        '''
        Adds an event to the past of this event.
        It returns False if the event is already in the past, otherwise it adds 
        the event and returns True.
        '''
        if other in self._prec:
            return False
        else:
            if self.hasBeenLinked() and CausetEvent.isLink(other, self):
                self._lprec -= other._prec
                self._lprec.add(other)
            self._prec.add(other)
            return True

    def _addToFuture(self, other: 'CausetEvent') -> bool:
        '''
        Adds an event to the future of this event.
        It returns False if the event is already in the future, otherwise it 
        adds the event and returns True.
        '''
        if other in self._succ:
            return False
        else:
            if self.hasBeenLinked() and CausetEvent.isLink(self, other):
                self._lsucc -= other._succ
                self._lsucc.add(other)
            self._succ.add(other)
            return True

    def _discard(self, other: 'CausetEvent') -> bool:
        '''
        Removes an event from the past and future of this event.
        It returns True if the event was in the past or future, otherwise False.
        '''
        if other in self._prec:
            if self.hasBeenLinked() and (other in self._lprec):
                self._lprec |= {
                    e for e in other._prec if CausetEvent.isLink(e, self)}
                self._lprec.remove(other)
            self._prec.discard(other)
            return True
        elif other in self._succ:
            if self.hasBeenLinked() and (other in self._lsucc):
                self._lsucc |= {
                    e for e in other._succ if CausetEvent.isLink(self, e)}
                self._lsucc.remove(other)
            self._succ.discard(other)
            return True
        else:
            return False

    def disjoin(self) -> None:
        '''
        Disjoins this event from the causal set (the set of events to the past 
        and future).
        '''
        self.unlink()
        for e in self.Cone:
            e._discard(self)
        self._prec = set()
        for e in self.Cone:
            e._discard(self)
        self._succ = set()

    def __str__(self):
        if self.Label:
            return f'#{self.Label}' if isinstance(self.Label, int) else \
                f'#\'{self.Label}\''
        else:
            return '#Event'

    def __repr__(self):
        P: str = ', '.join([str(e) for e in self.LinkPast])
        l: str = None
        if self.Label:
            l = str(self.Label) if isinstance(self.Label, int) else \
                f'\'{self.Label}\''
        try:
            if P:
                if l:
                    return f'{self.__class__.__name__}' + \
                           '(past={' + f'{P}' + '}, ' + \
                           f'label={l}, ' + \
                           f'coordinates={self._coordinates})'
                else:
                    return f'{self.__class__.__name__}' + \
                           '(past={' + f'{P}' + '}, ' + \
                           f'coordinates={self._coordinates})'
            elif l:
                return f'{self.__class__.__name__}' + \
                       f'(label={l}, ' + \
                       f'coordinates={self._coordinates})'
            else:
                return f'{self.__class__.__name__}' + \
                       f'(coordinates={self._coordinates})'
        except AttributeError:
            if P:
                if l:
                    return f'{self.__class__.__name__}' + \
                           '(past={' + f'{P}' + '}, ' + \
                           f'label={l})'
                else:
                    return f'{self.__class__.__name__}' + \
                           '(past={' + f'{P}' + '})'
            elif l:
                return f'{self.__class__.__name__}' + \
                       f'(label={l})'
            else:
                return f'{self.__class__.__name__}()'

    def __lt__(self, other):
        return self in other._prec

    def __le__(self, other):
        return (self is other) or (self in other._prec)

    def __gt__(self, other):
        return other in self._prec

    def __ge__(self, other):
        return (self is other) or (other in self._prec)

    def link(self) -> None:
        '''
        Computes the causal links between this event and all related events.

        Only call this method if it is necessary to increase performance of 
        this instance, when link requests with isPastLink(...) and 
        isFutureLink(...) are necessary. The first call of any of these two 
        methods will call link() if it was not called before.
        '''
        self._lprec: Set[CausetEvent] = {
            e for e in self._prec if CausetEvent.isLink(e, self)}
        self._lsucc: Set[CausetEvent] = {
            e for e in self._succ if CausetEvent.isLink(self, e)}

    def unlink(self) -> None:
        '''
        Force the CausetEvent instance to reset its link memory.
        '''
        delattr(self, '_lprec')
        delattr(self, '_lsucc')

    @classmethod
    def isLink(cls, a: 'CausetEvent', b: 'CausetEvent') -> bool:
        '''
        Tests if event a is linked to event b by intersecting a.Future() with 
        b.Past().

        This method is slow, but saves memory. Instead of calling this 
        method many times, faster access is achieved with the call of 
        a.isFutureLink(b).
        '''
        return not (a._succ & b._prec)

    def isPastLink(self, other: 'CausetEvent') -> bool:
        '''
        Tests if another event is linked in the past of this event.
        '''
        try:
            return other in self._lprec
        except AttributeError:
            self.link()
            return other in self._lprec

    def isFutureLink(self, other: 'CausetEvent') -> bool:
        '''
        Tests if another event is linked in the future of this event.
        '''
        try:
            return other in self._lsucc
        except AttributeError:
            self.link()
            return other in self._lsucc

    def isCausalTo(self, other: 'CausetEvent') -> bool:
        '''
        Tests if another event is causally related to this event.
        '''
        return (self <= other) or (self > other)

    def isLinkedTo(self, other: 'CausetEvent') -> bool:
        '''
        Tests if another event is linked in the past or future of this event.
        '''
        return self.isPastLink(other) or self.isFutureLink(other)

    def isSpacelikeTo(self, other: 'CausetEvent') -> bool:
        '''
        Tests if another event is spacelike separated to this event.
        '''
        return (other is not self) and (other not in self._prec) and \
            (other not in self._succ)

    @staticmethod
    def LinkCountOf(eventSet: Set['CausetEvent']) -> int:
        return sum([len(e.LinkPast & eventSet) for e in eventSet])

    @property
    def Past(self) -> Set['CausetEvent']:
        '''
        Returns a set of events (instances of CausetEvent) that are in the past 
        of this event.
        '''
        return self._prec

    @property
    def Future(self) -> Set['CausetEvent']:
        '''
        Returns a set of events (instances of CausetEvent) that are in the 
        future of this event.
        '''
        return self._succ

    @property
    def Cone(self) -> Set['CausetEvent']:
        '''
        Returns a set of events (instances of CausetEvent) that are in the 
        past, present or future of this event.
        '''
        return self._prec | {self} | self._succ

    @property
    def PresentOrPast(self) -> Set['CausetEvent']:
        '''
        Returns a set of events (instances of CausetEvent) that are in the past 
        of this event, including this event.
        '''
        return self._prec | {self}

    @property
    def PresentOrFuture(self) -> Set['CausetEvent']:
        '''
        Returns a set of events (instances of CausetEvent) that are in the 
        future of this event, including this event.
        '''
        return self._succ | {self}

    def Spacelike(self, eventSet: Set['CausetEvent']) -> Set['CausetEvent']:
        '''
        Returns the subset of events (instances of CausetEvent) of `eventSet` 
        that are spacelike separated to this event.
        '''
        return eventSet - self.Cone

    @property
    def PastCard(self) -> int:
        '''
        Returns the number of past events (set cardinality).
        '''
        return len(self._prec)

    @property
    def FutureCard(self) -> int:
        '''
        Returns the number of future events (set cardinality).
        '''
        return len(self._succ)

    @property
    def ConeCard(self) -> int:
        '''
        Returns the number of past and future events (set cardinality), 
        including this event (present).
        '''
        return len(self._prec) + len(self._succ) + 1

    def SpacelikeCard(self, eventSet: Set['CausetEvent']) -> int:
        '''
        Returns the number of events (instances of CausetEvent) of `eventSet` 
        that are spacelike separated to this event.
        '''
        if self in eventSet:
            return len(eventSet - self._prec - self._succ) - 1
        else:
            return len(eventSet - self._prec - self._succ)

    @property
    def LinkPast(self) -> Set['CausetEvent']:
        '''
        Returns a set of events (instances of CausetEvent) that are linked and 
        in the past of this event.
        '''
        try:
            return self._lprec
        except AttributeError:
            self.link()
            return self._lprec

    @property
    def LinkFuture(self) -> Set['CausetEvent']:
        '''
        Returns a set of events (instances of CausetEvent) that are linked and 
        in the future of this event.
        '''
        try:
            return self._lsucc
        except AttributeError:
            self.link()
            return self._lsucc

    @property
    def LinkCone(self) -> Set['CausetEvent']:
        '''
        Returns a set of events (instances of CausetEvent) that are linked and 
        in the past or future of this event.
        '''
        try:
            return self._lprec | self._lsucc
        except AttributeError:
            self.link()
            return self._lprec | self._lsucc

    @property
    def LinkPastCard(self) -> int:
        '''
        Returns the number of linked past events (set cardinality).
        '''
        try:
            return len(self._lprec)
        except AttributeError:
            self.link()
            return len(self._lprec)

    @property
    def LinkFutureCard(self) -> int:
        '''
        Returns the number of linked future events (set cardinality).
        '''
        try:
            return len(self._lsucc)
        except AttributeError:
            self.link()
            return len(self._lsucc)

    @property
    def LinkConeCard(self) -> int:
        '''
        Returns the number of linked past and linked future events (set 
        cardinality).
        '''
        try:
            return len(self._lprec) & len(self._lsucc)
        except AttributeError:
            self.link()
            return len(self._lprec) & len(self._lsucc)

    def copy(self) -> 'CausetEvent':
        '''
        Returns a shallow copy of the event without embedding.
        '''
        if self.Label is None:
            e = CausetEvent()
        elif isinstance(self.Label, int):
            e = CausetEvent(label=self.Label)
        else:
            e = CausetEvent(label=f'{self.Label}')
        e._prec = self._prec.copy()
        e._succ = self._succ.copy()
        if self.hasBeenLinked():
            e._lprec = self._lprec.copy()
            e._lsucc = self._lsucc.copy()
        return e

    def Rank(self, other: 'CausetEvent') -> float:
        '''
        Returns the rank of event other in the future of this event. If `other` 
        is not in the future of this event, math.inf is returned.
        '''
        if not self <= other:
            return math.inf
        elif self == other:
            return 0.0
        elif self.isFutureLink(other):
            return 1.0
        else:
            a_linked = self.LinkFuture
            b_linked = other.LinkPast
            if a_linked & b_linked:
                return 2.0
            r = math.inf
            for a_succ in a_linked:
                for b_prec in b_linked:
                    r = min(r, a_succ.Rank(b_prec) + 2.0)
                    if r == 3.0:
                        return 3.0
            return r

    @property
    def Position(self) -> List[float]:
        '''
        Returns the coordinates for the position in a Hasse diagram.
        '''
        try:
            return self._position
        except AttributeError:
            return [0.0, 0.0]

    @Position.setter
    def Position(self, value: Iterable[float]) -> None:
        '''
        Sets the coordinates for the position in a Hasse diagram.
        '''
        value = list(value)
        if len(value) != 2:
            raise ValueError(
                'The position has to be an iterable of exactly two values.')
        self._position = value

    @property
    def isEmbedded(self) -> bool:
        '''
        Returns True if the event has a coordinate tuple assigned to it.
        '''
        return hasattr(self, '_coordinates')

    def embed(self, coordinates: Iterable[float], reembed: bool = False) -> \
            None:
        '''
        Assigns coordinates to the event. If the event already has coordinates, 
        an `AttributeError` is raised. To avoid this error and overwrite the 
        value, use reembed.
        '''
        if not reembed and self.isEmbedded:
            raise AttributeError(f'The event {self} is already embedded. ' +
                                 'Use the `disembed` method first or use ' +
                                 'the `reembed` flag.')
        self._coordinates = list(coordinates)

    def disembed(self) -> None:
        '''
        Removes the embedding of the event.
        '''
        delattr(self, '_coordinates')

    @property
    def Coordinates(self) -> List[float]:
        '''
        Returns the embedding coordinates.

        Raises an AttributeError if the event is not embedded.
        '''
        try:
            return self._coordinates
        except AttributeError:
            raise AttributeError(f'The event {self} is not embedded. ' +
                                 'Use the `embed` method first.')

    @property
    def CoordinatesDim(self) -> int:
        '''
        Returns the dimension of its coordinates if any, otherwise 0.
        '''
        try:
            return len(self._coordinates)
        except AttributeError:
            return 0
