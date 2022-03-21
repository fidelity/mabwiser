# SPDX-License-Identifier: Apache-2.0
from enum import EnumMeta
from typing import Tuple, Union

from mabwiser.utilities.types import _C, _T


class _EnumMeta(EnumMeta):
    def __getitem__(self, item: _T):
        return super(_EnumMeta, self).__getitem__(type(item).__name__)
