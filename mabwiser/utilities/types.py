# -*- coding: utf-8 -*-
# SPDX-License-Identifer: Apache-2.0

"""
This module provides types for type hinting
"""

from typing import TypeVar, Union

_T = TypeVar("_T")
_C = TypeVar("_C", bound=type)

Num = Union[int, float]
"""Num type is defined as integer or float."""
