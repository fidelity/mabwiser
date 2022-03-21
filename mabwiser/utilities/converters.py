# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from mabwiser.configs.calls import LPCall, NPCall
from mabwiser.configs.learning import LearningPolicy, SpecialLinearPolicy
from mabwiser.utilities.types import _T


def convert_array(
    array_like: Union[np.ndarray, List, pd.DataFrame, pd.Series]
) -> np.ndarray:
    """
    Convert given array to numpy array for efficiency.
    """
    if isinstance(array_like, np.ndarray):
        return array_like
    elif isinstance(array_like, list):
        return np.asarray(array_like)
    elif isinstance(array_like, pd.Series):
        return array_like.values
    else:
        raise NotImplementedError(
            f"Unsupported data type `{type(array_like)}` -- currently supported "
            "np.ndarray, List, pd.DataFrame, pd.Series"
        )


def _handle_pd_series(pd_like: pd.Series, row: bool = False, **kwargs) -> np.ndarray:
    if row:
        return np.asarray(pd_like.values, order="C").reshape(1, -1)
    else:
        return np.asarray(pd_like.values, order="C").reshape(-1, 1)


def convert_matrix(
    matrix_like: Union[np.ndarray, List, pd.DataFrame, pd.Series],
    row: bool = False,
    handle_pd_fn: Callable = _handle_pd_series,
    pd_fn_kwargs: Dict = {},
) -> np.ndarray:
    """
    Convert contexts to numpy array for efficiency.
    For fit and partial fit, decisions must be provided.
    The numpy array need to be in C row-major order for efficiency.
    If the data is a series for a single row, set the row flag to True.
    """
    if isinstance(matrix_like, np.ndarray):
        if matrix_like.flags["C_CONTIGUOUS"]:
            return matrix_like
        else:
            return np.asarray(matrix_like, order="C")
    elif isinstance(matrix_like, list):
        return np.asarray(matrix_like, order="C")
    elif isinstance(matrix_like, pd.DataFrame):
        if matrix_like.values.flags["C_CONTIGUOUS"]:
            return matrix_like.values
        else:
            return np.asarray(matrix_like.values, order="C")
    elif isinstance(matrix_like, pd.Series):
        return handle_pd_fn(matrix_like, row=row, **pd_fn_kwargs)
    else:
        raise NotImplementedError(
            f"Unsupported data type `{type(matrix_like)}` -- currently supported "
            "np.ndarray, List, pd.DataFrame, pd.Series"
        )


def _handle_context_pd_series(
    contexts: pd.Series,
    lp: LearningPolicy,
    imp: _T,
    decisions: Optional[np.ndarray] = None,
    **kwargs,
):
    if decisions is not None:
        pd_row = not len(decisions) > 1
        _handle_pd_series(contexts, row=pd_row)
    else:  # For predictions, compare the shape to the stored context history
        # We need to find out the number of features (to distinguish Series shape)
        if SpecialLinearPolicy.has_value(lp):
            first_arm = imp.arms[0]
            if LPCall.isinstance(imp, ("LinUCB", "LinTS", "LinGreedy")):
                num_features = imp.arm_to_model[first_arm].beta.size
            else:
                num_features = imp.contexts.shape[1]
        elif NPCall.isinstance(imp, "TreeBandit"):
            # Even when fit() happened, the first arm might not necessarily have a fitted tree
            # So we have to search for a fitted tree
            # TODO: There is unreachable code here... num_features is not defined if not found -- need to raise
            for arm in imp.arms:
                try:
                    num_features = len(imp.arm_to_tree[arm].feature_importances_)
                except:
                    continue
        else:
            num_features = imp.contexts.shape[1]
        pd_row = num_features != 1
        _handle_pd_series(contexts, row=pd_row)


def convert_context(
    lp: LearningPolicy,
    imp: _T,
    contexts: Union[np.ndarray, List, pd.DataFrame, pd.Series],
    decisions: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert contexts to numpy array for efficiency.
    For fit and partial fit, decisions must be provided.
    The numpy array need to be in C row-major order for efficiency.
    """
    return convert_matrix(
        contexts,
        handle_pd_fn=_handle_context_pd_series,
        pd_fn_kwargs={"lp": lp, "imp": imp, "decisions": decisions},
    )
