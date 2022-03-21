# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
from enum import Enum
from typing import Tuple, Union

from mabwiser.clusters import _Clusters
from mabwiser.configs.base import _EnumMeta
from mabwiser.greedy import _EpsilonGreedy
from mabwiser.linear import _LinGreedy, _LinTS, _LinUCB
from mabwiser.neighbors.approximate import _LSHNearest
from mabwiser.neighbors.fixed import _KNearest, _Radius
from mabwiser.popularity import _Popularity
from mabwiser.rand import _Random
from mabwiser.softmax import _Softmax
from mabwiser.thompson import _ThompsonSampling
from mabwiser.treebandit import _TreeBandit
from mabwiser.ucb import _UCB1
from mabwiser.utilities.types import _C, _T


class LPCall(Enum, metaclass=_EnumMeta):
    EpsilonGreedy = _EpsilonGreedy
    Popularity = _Popularity
    Random = _Random
    Softmax = _Softmax
    ThompsonSampling = _ThompsonSampling
    UCB1 = _UCB1
    LinGreedy = _LinGreedy
    LinTS = _LinTS
    LinUCB = _LinUCB

    def __call__(self, *args, **kwargs) -> _T:
        return self.value(*args, **kwargs)

    @classmethod
    def isinstance(cls, obj: _C, instance_name: Union[str, Tuple[str, ...]]):
        if isinstance(instance_name, (tuple, Tuple)):
            return any([cls._unit_isinstance(val, obj) for val in instance_name])
        else:
            return cls._unit_isinstance(obj, instance_name)

    @classmethod
    def _unit_isinstance(cls, obj: _C, instance_name: str):
        if instance_name not in cls.__members__:
            return False
        return isinstance(obj, cls.__members__[instance_name].value)


class NPCall(Enum, metaclass=_EnumMeta):
    LSHNearest = _LSHNearest
    Clusters = _Clusters
    KNearest = _KNearest
    Radius = _Radius
    TreeBandit = _TreeBandit

    def __call__(self, *args, **kwargs) -> _T:
        return self.value(*args, **kwargs)

    @classmethod
    def isinstance(cls, obj: _C, instance_name: Union[str, Tuple[str, ...]]):
        if isinstance(instance_name, (tuple, Tuple)):
            return any([cls._unit_isinstance(val, obj) for val in instance_name])
        else:
            return cls._unit_isinstance(instance_name, obj)

    @classmethod
    def _unit_isinstance(cls, obj: _C, instance_name: str):
        if instance_name not in cls.__members__:
            return False
        return isinstance(obj, cls.__members__[instance_name].value)
