# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0

"""
This module spock configuration definitions for Arm related operations
"""

from typing import Callable, Dict, List, Optional

from spock import spock
from spock.utils import within

from mabwiser.utilities.validators import check_sklearn_scaler


@spock
class ArmConfig:
    arm: str
    binarizer: Optional[Callable] = None
    scaler: Optional[Callable] = None

    def __post_hook__(self):
        try:
            check_sklearn_scaler(self.scaler)
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )


@spock
class WarmStartConfig:
    arm_to_features: Dict[str, List[float]]
    distance_quantile: float

    def __post_hook__(self):
        try:
            within(
                self.distance_quantile,
                low_bound=0.0,
                upper_bound=1.0,
                inclusive_upper=True,
                inclusive_lower=True,
            )
        except Exception as e:
            raise ValueError(
                f"`{self.__class__.__name__}` could not be instantiated -- spock message: {e}"
            )
