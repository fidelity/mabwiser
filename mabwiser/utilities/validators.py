# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module provides various validators
"""

from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def validate_2d(data: Union[np.ndarray, List, pd.DataFrame, pd.Series], var_name: str):
    """
    Validates that  data is 2D
    """
    if isinstance(data, np.ndarray):
        check_true(
            data.ndim == 2,
            TypeError(
                f"Data named {var_name} should be given as 2D list, numpy array, pandas series or "
                f"data frames."
            ),
        )
    elif isinstance(data, list):
        check_true(
            np.array(data).ndim == 2,
            TypeError(
                f"Data named {var_name} should be given as 2D list, numpy array, pandas series or "
                f"data frames."
            ),
        )
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        check_true(
            isinstance(data, (pd.Series, pd.DataFrame)),
            TypeError(
                f"Data named {var_name} should be given as 2D list, numpy array, pandas series or "
                f"data frames."
            ),
        )
    else:
        raise TypeError(
            f"Data named {var_name} should be given as 2D list, numpy array, pandas series or "
            f"data frames."
        )


def check_false(expression: bool, exception: Exception) -> None:
    """
    Checks that given expression is false, otherwise raises the given exception.
    """
    if expression:
        raise exception


def check_true(expression: bool, exception: Exception) -> None:
    """
    Checks that given expression is true, otherwise raises the given exception.
    """
    if not expression:
        raise exception


def check_sum_to_unity(list_floats: List[float]):
    if isinstance(list_floats, (list, List)):
        if np.isclose(sum(list_floats), 1.0):
            raise ValueError(
                f"The list of values should sum up close to 1.0 -- currently `{sum(list_floats)}`"
            )


def check_sklearn_scaler(scaler: Optional[Callable]):
    if scaler is not None:
        if not hasattr(scaler, "transform"):
            raise TypeError(
                "Scaler must be a scaler object from sklearn.preprocessing with a transform method"
            )
        if not hasattr(scaler, "mean_") or not hasattr(scaler, "var_"):
            raise TypeError(
                "Scaler must be fit with calculated mean_ and var_ attributes"
            )


def check_in_arms(decisions: Union[List[str], np.ndarray, pd.Series], arms: List[str]):
    if isinstance(decisions, (list, List, np.ndarray)):
        tf_list = [v not in arms for v in decisions]
    else:
        tf_list = [v not in arms for v in decisions.values]
    if any(tf_list):
        raise ValueError(
            f"Arm `{decisions[tf_list.index(True)]}` is not within the set of defined arms `{repr(arms)}`"
        )


def check_fit_input(
    data: Union[
        Union[List[str], np.ndarray, pd.Series],
        Tuple[Union[List[str], np.ndarray, pd.Series], ...],
    ]
):
    if isinstance(data, (tuple, Tuple)):
        for v in data:
            _check_fit_input(v)
    else:
        _check_fit_input(data)


def _check_fit_input(data: Union[List[str], np.ndarray, pd.Series]):
    check_true(
        isinstance(data, (list, np.ndarray, pd.Series)),
        TypeError(
            "The decisions should be given as list, numpy array, or pandas series."
        ),
    )
